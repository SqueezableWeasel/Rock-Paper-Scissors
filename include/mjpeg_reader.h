#pragma once // Include the MJPEG parser declarations only once in each source file.

#include <array> // Provide fixed storage for a block read from the camera pipe.
#include <cerrno> // Handle interrupted operating system reads using errno and EINTR.
#include <cstddef> // Use platform-sized counters for buffer lengths and offsets.
#include <cstring> // Use memchr to locate the next marker prefix in a block.
#include <unistd.h> // Import the POSIX read call for the camera pipe.
#include <vector> // Collect the variable length compressed bytes of one frame.

// Retain unread bytes for the next frame and handle markers across reads.
class MjpegReader { // Parse consecutive JPEG images from a borrowed camera pipe descriptor.
public: // Expose construction and extraction of one complete frame.
    explicit MjpegReader(int fd) : fd_(fd) {} // Keep the caller's pipe descriptor without taking ownership of it.

    bool next(std::vector<unsigned char>& jpeg) { // Read one JPEG frame into the caller's reusable byte buffer.
        jpeg.clear(); // Remove the previous JPEG while retaining the allocated capacity.
        if (!scan(0xd8, nullptr)) return false; // Find the start marker and skip any bytes before it.
        jpeg.insert(jpeg.end(), {0xff, 0xd8}); // Restore the start marker consumed during the search.
        return scan(0xd9, &jpeg); // Append bytes through the end marker and return success only if it was found.
    } // End extraction of one compressed frame.

private: // Keep the marker loop and pending stream bytes internal.
    bool scan(unsigned char marker, std::vector<unsigned char>* output) { // Find the requested marker while optionally appending consumed bytes.
        constexpr std::size_t maxFrameBytes = 5 * 1024 * 1024; // Limit buffering when the stream is corrupt or a marker is missing.
        bool prefix = false; // Remember whether the preceding byte was the JPEG marker prefix.
        while (true) { // Continue searching until a marker is found or the stream ends.
            if (position_ == size_) { // Refill only after using the unread bytes in the current block.
                ssize_t length; // Hold the signed byte count because read returns negative errors.
                do { // Retry a read interrupted by a signal.
                    length = ::read(fd_, buffer_.data(), buffer_.size()); // Read a block without waiting for the entire buffer to fill.
                } while (length < 0 && errno == EINTR); // Repeat only for EINTR rather than for an ordinary read failure.
                if (length <= 0) return false; // Return failure on end of stream or an unrecoverable read error.
                position_ = 0; // Rewind the cursor to the beginning of the new block.
                size_ = static_cast<std::size_t>(length); // Record how many bytes are valid in the block.
            } // Finish refilling the input buffer.
            const auto* start = buffer_.data() + position_; // Remember the first byte consumed in this pass for the bulk append.
            bool found = false; // Clear the match flag before examining this part of the buffer.
            while (position_ < size_) { // Inspect the bytes that remain in this block.
                const auto* current = buffer_.data() + position_; // Get the address of the next unread byte.
                if (prefix && *current == marker) { // Match a marker code only when a prefix immediately precedes it.
                    ++position_; // Consume the marker code and leave later bytes for the next search.
                    found = true; // Record the match for the caller.
                    break; // Stop scanning the block at the matched marker.
                } // End the marker match branch.
                const auto* nextPrefix = static_cast<const unsigned char*>( // Find the next candidate prefix without inspecting every byte individually.
                    std::memchr(current, 0xff, size_ - position_)); // Search only the unread portion of the current block.
                prefix = nextPrefix != nullptr; // Carry the prefix flag across a block boundary if the prefix is the last byte.
                position_ = prefix ? static_cast<std::size_t>(nextPrefix - buffer_.data()) + 1 : size_; // Advance beyond the candidate prefix or exhaust the block.
            } // Finish this block search or continue at the next refill.
            if (output) { // Append bytes only when collecting a frame rather than seeking its start.
                const auto* end = buffer_.data() + position_; // Locate the end of the consumed range.
                if (static_cast<std::size_t>(end - start) > maxFrameBytes - output->size()) return false; // Reject a corrupt or oversized image before growing the frame buffer.
                output->insert(output->end(), start, end); // Append the consumed range in a single operation.
            } // End the optional frame append.
            if (found) return true; // Return success after writing any bytes through the marker.
        } // Continue searching in the next block if needed.
    } // End the marker scanner.

    int fd_; // Store the borrowed file descriptor for future camera reads.
    std::array<unsigned char, 32 * 1024> buffer_; // Keep one 32 KiB input block for bulk reads.
    std::size_t position_ = 0; // Track the next unread byte in the current block.
    std::size_t size_ = 0; // Track the number of valid bytes in the current block.
}; // End the MJPEG reader class.
