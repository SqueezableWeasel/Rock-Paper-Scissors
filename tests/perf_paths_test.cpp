#include "mjpeg_reader.h"  // Imports the chunked MJPEG frame reader under test.
#include "tpu_input.h"  // Imports the direct Coral input packing routines.

#include <cassert>  // Provides assertions for the expected bytes and frame boundaries.
#include <cstdio>  // Provides temporary-file and terminal output functions.
#include <memory>  // Provides an owning pointer for the temporary file.

static void testModelInput() {  // Test the two quantized model input conversion paths.
    std::array<std::int8_t, 256> direct{}, general{};  // Reserve one signed output value for each possible input byte in both tables.
    for (int value = 0; value < 256; ++value) {  // Populate the complete range of byte values.
        direct[value] = static_cast<std::int8_t>(value - 128);  // Build the offset-only table used by the fast path.
        general[value] = static_cast<std::int8_t>((value / 2) - 65);  // Build a contrasting table that requires general lookup.
    }  // Finish preparing both conversion tables.
    assert(isBiasOnlyInput(direct));  // Confirm the offset-only table enables the fast path.
    assert(!isBiasOnlyInput(general));  // Confirm the alternative table falls back to lookup.
    const std::size_t pixels = 513;  // All byte values and a partial SIMD block.
    std::vector<std::uint8_t> bgr(pixels * 3);  // Allocate three interleaved input channels per pixel.
    for (std::size_t i = 0; i < bgr.size(); ++i)  // Visit every byte in the input fixture.
        bgr[i] = static_cast<std::uint8_t>(i * 131 + 19);  // Fill the input with a sequence that cycles through every byte value.
    for (const auto& lut : {direct, general}) {  // Feed the same input to each conversion path.
        std::vector<std::int8_t> quantized(pixels * 3 + 1, -1);  // Reserve an extra sentinel byte beyond the expected output.
        packBgrTpuInput(bgr.data(), quantized.data(), pixels, lut, isBiasOnlyInput(lut));  // Pack BGR pixels into the signed RGB model input.
        for (std::size_t i = 0; i < pixels; ++i) {  // Compare all channels for every pixel.
            assert(quantized[i * 3] == lut[bgr[i * 3 + 2]]);  // Confirm blue-red reordering puts the looked-up red byte first.
            assert(quantized[i * 3 + 1] == lut[bgr[i * 3 + 1]]);  // Confirm the green channel remains in the middle.
            assert(quantized[i * 3 + 2] == lut[bgr[i * 3]]);  // Confirm the looked-up blue byte appears last.
        }  // Finish checking the pixels for this table.
        assert(quantized.back() == -1);  // Confirm the writer stopped before the sentinel.
    }  // Finish checking both conversion paths.
}  // End the quantized-input test.

static void testMjpeg() {  // Test MJPEG framing across buffer boundaries and truncation.
    using Bytes = std::vector<unsigned char>;  // Use a byte vector for the stream and each expected frame.
    std::unique_ptr<FILE, decltype(&std::fclose)> stream(std::tmpfile(), &std::fclose);  // Open a temporary file whose owner closes it when the test ends.
    assert(stream);  // Stop before using the file if creation failed.
    // START and END both straddle 32 KiB reads, followed by two frames in
    // the same chunk and an incomplete frame. FF 00 is an escaped byte.
    Bytes bytes(65537, 0x19);  // Fill two chunks and one additional byte with known input data.
    bytes[32767] = 0xff;  // Place the start marker prefix at the end of the first chunk.
    bytes[32768] = 0xd8;  // Place the start marker code at the beginning of the second chunk.
    bytes[65535] = 0xff;  // Place the end marker prefix at the end of the second chunk.
    bytes[65536] = 0xd9;  // Finish the first frame when the third chunk begins.
    const Bytes second{0xff, 0xd8, 0xff, 0, 0xff, 0xd9};  // Prepare a complete frame containing an escaped FF data byte.
    const Bytes third{0xff, 0xd8, 0x01, 0xff, 0xff, 0xd9};  // Prepare another adjacent frame with an extra FF before its end marker.
    bytes.insert(bytes.end(), second.begin(), second.end());  // Append the second complete frame to the input.
    bytes.insert(bytes.end(), third.begin(), third.end());  // Append the third complete frame immediately after it.
    bytes.insert(bytes.end(), {0xff, 0xd8, 0x33, 0xff});  // Add an unfinished frame with no end marker at the end of the stream.
    assert(std::fwrite(bytes.data(), 1, bytes.size(), stream.get()) == bytes.size());  // Write the complete fixture and verify no bytes were lost.
    assert(std::fflush(stream.get()) == 0);  // Flush the buffered write before reading the file descriptor directly.
    assert(::lseek(fileno(stream.get()), 0, SEEK_SET) == 0);  // Rewind to the first chunk of the fixture.

    MjpegReader reader(fileno(stream.get()));  // Read the bytes through the same descriptor API used for the camera pipe.
    Bytes frame;  // Collect one emitted frame at a time.
    assert(reader.next(frame) && frame == Bytes(bytes.begin() + 32767, bytes.begin() + 65537));  // Verify the first frame spans both markers split across chunk boundaries.
    assert(reader.next(frame) && frame == second);  // Verify the escaped byte did not split the second frame.
    assert(reader.next(frame) && frame == third);  // Verify the adjacent third frame is returned without losing any bytes.
    assert(!reader.next(frame));  // Verify the trailing partial frame is not returned as complete.
}  // Release the temporary file after the framing checks.

int main() {  // Run both performance-path regression tests.
    testModelInput();  // Verify exact channel order, quantization, and output bounds.
    testMjpeg();  // Verify complete and partial MJPEG frame handling.
    std::puts("PASS exact model input and split, adjacent, and truncated MJPEG frames");  // Report success after both sets of assertions.
}  // Finish the test process.
