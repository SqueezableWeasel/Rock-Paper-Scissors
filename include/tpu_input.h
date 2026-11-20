#pragma once // Include tensor input preparation declarations only once per source file.

#include <array> // Use a fixed array for the 256 possible byte quantization values.
#include <cstddef> // Use platform-sized pixel counts and indices.
#include <cstdint> // Use exact eight bit source and destination sample types.

#if defined(__aarch64__) && defined(__ARM_NEON) // Include SIMD support only on builds with 64-bit ARM NEON.
#include <arm_neon.h> // Import vector load, xor, and store operations.
#endif // End the conditional SIMD header inclusion.

inline bool isBiasOnlyInput(const std::array<std::int8_t, 256>& lut) { // Check whether quantization is exactly unsigned byte minus 128.
    for (int i = 0; i < 256; ++i) // Verify every possible input byte against that mapping.
        if (lut[i] != i - 128) return false; // Keep the lookup table path if any output value differs.
    return true; // Enable the simpler conversion only after all values match.
} // End the input quantization check.

// Read interleaved BGR pixels and write quantized RGB directly into the
// interpreter input. The unit-scale path is valid only for an exact LUT match.
inline void packBgrTpuInput(const std::uint8_t* source, std::int8_t* destination, // Write BGR camera pixels straight into the RGB input tensor.
                           std::size_t pixels, const std::array<std::int8_t, 256>& lut, // Accept the pixel count and the model's exact quantization table.
                           bool biasOnly) { // Select the bias-only path only when the table has been verified.
    if (biasOnly) { // Convert directly when the quantization has only a constant bias.
#if defined(__aarch64__) && defined(__ARM_NEON) // Use the NEON implementation only on supported 64-bit ARM builds.
        const uint8x16_t bias = vdupq_n_u8(0x80); // Set the sign bit mask for each of the 16 vector lanes.
        for (; pixels >= 16; source += 48, destination += 48, pixels -= 16) { // Process groups of 16 pixels and advance both byte pointers together.
            const uint8x16x3_t bgr = vld3q_u8(source); // Read packed BGR pixels into separate channel vectors.
            uint8x16x3_t rgb; // Hold the channels in the tensor's RGB order.
            rgb.val[0] = veorq_u8(bgr.val[2], bias); // Move red into the first channel and subtract 128 in the byte representation.
            rgb.val[1] = veorq_u8(bgr.val[1], bias); // Apply the same bias to green and retain its channel position.
            rgb.val[2] = veorq_u8(bgr.val[0], bias); // Move blue into the third channel and apply the bias.
            vst3q_u8(reinterpret_cast<std::uint8_t*>(destination), rgb); // Store the three quantized channel vectors as interleaved RGB.
        } // End the NEON loop and leave any partial group for scalar conversion.
#endif // End the conditional vector path.
        for (; pixels != 0; source += 3, destination += 3, --pixels) { // Process the remaining pixels or the complete input on non-NEON builds.
            destination[0] = static_cast<std::int8_t>(static_cast<int>(source[2]) - 128); // Read red from BGR index two and write it as the first signed tensor channel.
            destination[1] = static_cast<std::int8_t>(static_cast<int>(source[1]) - 128); // Apply the constant bias to green.
            destination[2] = static_cast<std::int8_t>(static_cast<int>(source[0]) - 128); // Read blue from BGR index zero and write it as the third signed channel.
        } // End the scalar conversion for bias-only input.
    } else { // Use the table for models with a different quantization mapping.
        for (; pixels != 0; source += 3, destination += 3, --pixels) { // Process one three-channel pixel at a time.
            destination[0] = lut[source[2]]; // Look up red and store it first in the RGB tensor.
            destination[1] = lut[source[1]]; // Look up green and store it second in the RGB tensor.
            destination[2] = lut[source[0]]; // Look up blue and store it third in the RGB tensor.
        } // End the table-based pixel loop.
    } // Finish the selected quantization path.
} // End tensor preparation.
