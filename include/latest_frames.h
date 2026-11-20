#ifndef RPS_LATEST_FRAMES_H_ // Prevent duplicate definitions of the threaded frame buffer.
#define RPS_LATEST_FRAMES_H_ // Mark the frame buffer header as included.
#include "mjpeg_reader.h" // Import the parser for JPEGs coming from the camera pipe.
#include <thread> // Run capture and decoding on a dedicated thread.
#include <mutex> // Guard the frame and its associated metrics during transfer.
#include <condition_variable> // Allow the consumer to sleep until a frame arrives or capture ends.
#include <atomic> // Make the stopping flag safe to read from the worker.
#include <cstdint> // Use fixed-width counters for capture statistics.
#include <utility> // Provide move and swap operations for recycling image buffers.

// Included after CameraProcess. Owns the single reader of the camera pipe.
// One decoded frame slot bounds latency and memory if inference falls behind.
class LatestFrames { // Transfer the newest decoded frame from camera capture to inference.
public: // Expose the metrics record to callers.
    struct Metrics { // Bundle timings and counters with each consumed frame.
        double decodeMs = 0; // Store the JPEG decode duration in milliseconds.
        double ageMs = 0; // Measure time since the compressed frame was read.
        std::uint64_t captured = 0; // Count successfully decoded frames.
        std::uint64_t dropped = 0; // Count unconsumed frames replaced by newer ones.
    }; // End the capture statistics record.

private: // Keep the camera pipe, worker, and synchronization state internal.
    CameraProcess& camera; // Borrow the open camera process for the lifetime of capture.
    std::mutex mutex; // Protect the frame slot, sequence flags, and metrics.
    std::condition_variable changed; // Notify the consumer when the frame slot or stream state changes.
    std::atomic<bool> stopping{false}; // Start with the stop request cleared.
    bool ended = false, pending = false; // Record whether the stream has ended and whether a frame is ready.
    cv::Mat latest; // Hold the newest decoded image until the consumer picks it up.
    double latestDecodeMs = 0; // Associate the decode duration with the current slot.
    std::chrono::steady_clock::time_point latestReceived; // Timestamp receipt of the JPEG represented by the current slot.
    std::uint64_t captured = 0, dropped = 0; // Maintain cumulative counts for successful decodes and overwritten frames.
    // Start the thread only after every object it accesses is initialized.
    std::thread worker; // Own the capture thread, constructed after the state it accesses.

    void capture() { // Read and decode JPEGs in the background.
        try { // Report unexpected decoding failures and signal end of capture.
            MjpegReader reader(camera.streamFd); // Read compressed frames from the camera process's pipe.
            std::vector<unsigned char> jpeg; // Allocate a reusable compressed frame buffer.
            jpeg.reserve(256 * 1024); // Reserve space for ordinary camera JPEGs before the loop starts.
            cv::Mat reusable; // Keep an output image buffer available for reuse by the decoder.
            while (!stopping && reader.next(jpeg)) { // Read until shutdown or the end of the MJPEG stream.
                const auto received = std::chrono::steady_clock::now(); // Timestamp the assembled JPEG before decoding it.
                cv::Mat frame = cv::imdecode(jpeg, cv::IMREAD_COLOR, &reusable); // Decode to a color image, reusing the old pixel buffer if possible.
                const double decodeMs = std::chrono::duration<double, std::milli>( // Measure how long JPEG decoding took in milliseconds.
                    std::chrono::steady_clock::now() - received).count(); // Compute elapsed time since the decoder received the JPEG.
                if (frame.empty()) continue; // Skip a malformed JPEG that could not produce an image.
                { // Hold the publication lock only while exchanging state.
                    std::lock_guard<std::mutex> lock(mutex); // Prevent the consumer from seeing a partially updated frame record.
                    if (pending) ++dropped; // Count the previous slot as dropped if it was never consumed.
                    ++captured; // Record one more successful decode.
                    std::swap(latest, frame); // Publish the decoded image and take the previous slot's buffer.
                    reusable = std::move(frame); // Keep the released buffer for the next decode.
                    latestReceived = received; // Publish the compressed frame receipt time with the image.
                    latestDecodeMs = decodeMs; // Publish the corresponding decode duration.
                    pending = true; // Mark the newly published image as ready to consume.
                } // Unlock the frame slot before notifying the consumer.
                changed.notify_one(); // Wake the thread waiting for the newest frame.
            } // Continue capture while the camera produces images.
        } catch (const std::exception& error) { // Catch decoder and pipe processing exceptions at the thread boundary.
            std::cerr << "Camera decode error: " << error.what() << std::endl; // Report the reason background capture ended abnormally.
        } // Finish handling the capture failure.
        { // Publish end of stream under the same lock as the ready flag.
            std::lock_guard<std::mutex> lock(mutex); // Keep the end flag synchronized with the consumer's wait predicate.
            ended = true; // Allow the consumer to stop waiting for future frames.
        } // Release the lock after updating the end flag.
        changed.notify_one(); // Wake the consumer so it can drain the slot or exit.
    } // End the capture thread routine.
public: // Expose frame consumption and lifetime control.
    explicit LatestFrames(CameraProcess& camera) : camera(camera), worker(&LatestFrames::capture, this) {} // Retain the camera reference and start the background capture thread.
    LatestFrames(const LatestFrames&) = delete; // Disallow copying a running capture thread.
    LatestFrames& operator=(const LatestFrames&) = delete; // Disallow replacing the worker through copy assignment.
    ~LatestFrames() { stop(); } // Join the capture worker when the frame buffer goes out of scope.

    bool next(cv::Mat& frame, Metrics& metrics) { // Wait for the next available image and report its capture metrics.
        std::unique_lock<std::mutex> lock(mutex); // Hold a lock that the condition variable can release during the wait.
        if (!changed.wait_for(lock, std::chrono::seconds(3), [&]{ return pending || ended; }) || !pending) // Wait at most three seconds for a frame or an end notification.
            return false; // Return failure if the wait ends with no image to consume.
        // Return the caller's previous frame for reuse by the worker.
        std::swap(frame, latest); // Give the consumer the newest image and retain its old buffer.
        pending = false; // Mark the slot as consumed so the next publication is not counted as a drop.
        metrics.decodeMs = latestDecodeMs; // Return the JPEG decode duration associated with this image.
        metrics.ageMs = std::chrono::duration<double, std::milli>( // Measure how old the captured frame is at the point of consumption.
            std::chrono::steady_clock::now() - latestReceived).count(); // Convert the time since its JPEG arrived to milliseconds.
        metrics.captured = captured; // Copy the cumulative successful frame count under the lock.
        metrics.dropped = dropped; // Copy the cumulative overwrite count under the lock.
        return true; // Signal that a frame and its metrics were returned.
    } // End frame consumption.

    void stop() { // Stop camera output before waiting for the capture thread to exit.
        if (!worker.joinable()) return; // Do nothing if the worker has already been joined.
        stopping = true; // Request that the worker stop reading further frames.
        // Unblock the pipe reader before joining, then close the pipe.
        if (camera.pid > 0) { // Send termination signals only while the camera has a valid process identifier.
            kill(-camera.pid, SIGTERM); // Signal the camera's process group to stop inherited writers too.
            kill(camera.pid, SIGTERM); // Also signal the camera process directly in case its group was not set.
        } // End the camera signaling branch.
        worker.join(); // Wait until no thread can read from the camera pipe.
        camera.stop(); // Close the pipe and reap the camera process.
    } // End capture shutdown.
}; // End the latest-frame buffer class.
#endif // Close the latest frame include guard.
