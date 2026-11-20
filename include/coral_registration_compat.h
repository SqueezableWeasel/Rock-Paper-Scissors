#ifndef RPS_CORAL_REGISTRATION_COMPAT_H_ // Prevent duplicate definitions of the Coral registration adapter.
#define RPS_CORAL_REGISTRATION_COMPAT_H_ // Mark the registration adapter header as included.

#include <cstddef> // Provide offsetof for checking callback layout.
#include <cstring> // Provide memcpy for reading only the shared callback prefix.
#include <type_traits> // Verify the callback prefix can be copied byte for byte.
#include <tensorflow/lite/c/common.h> // Import the build's TensorFlow Lite registration definition.

// Legacy libedgetpu returns a shorter TfLiteRegistration than TFLite 2.20
// expects. Do not copy/dereference that object as a complete modern struct.
// These four callbacks have the same leading layout in TFLite 2.5 and 2.20.
// This is a narrow registration adapter, not a general TensorFlow ABI bridge.
namespace rps_coral { // Keep Coral compatibility symbols separate from the game code.
struct OpCallbacks { // Describe just the callback prefix shared with the legacy runtime.
    decltype(TfLiteRegistration::init) init; // Match the type of TensorFlow Lite's operation initializer.
    decltype(TfLiteRegistration::free) free; // Match the type of TensorFlow Lite's operation cleanup callback.
    decltype(TfLiteRegistration::prepare) prepare; // Match the type of TensorFlow Lite's tensor preparation callback.
    decltype(TfLiteRegistration::invoke) invoke; // Match the type of TensorFlow Lite's operation execution callback.
}; // End the portable callback prefix.

static_assert(std::is_trivially_copyable<OpCallbacks>::value, "Callback prefix must be copyable"); // Ensure copying the prefix with memcpy is valid for its type.
static_assert(offsetof(OpCallbacks, init) == offsetof(TfLiteRegistration, init), "init layout changed"); // Verify that the initializer starts at the expected byte offset.
static_assert(offsetof(OpCallbacks, free) == offsetof(TfLiteRegistration, free), "free layout changed"); // Verify that the cleanup callback starts at the expected byte offset.
static_assert(offsetof(OpCallbacks, prepare) == offsetof(TfLiteRegistration, prepare), "prepare layout changed"); // Verify that the preparation callback starts at the expected byte offset.
static_assert(offsetof(OpCallbacks, invoke) == offsetof(TfLiteRegistration, invoke), "invoke layout changed"); // Verify that the execution callback starts at the expected byte offset.
static_assert(sizeof(OpCallbacks) == offsetof(TfLiteRegistration, profiling_string), "Callback prefix changed"); // Verify that the known prefix ends before profiling and other registration fields.

inline TfLiteRegistration adaptRegistration(const void* runtimeRegistration) { // Construct a registration for this build from only the proven legacy prefix.
    TfLiteRegistration result{}; // Start with zeroed fields so unknown extensions remain disabled.
    if (!runtimeRegistration) return result; // Return the empty registration if the runtime supplies no callback block.
    OpCallbacks callbacks{}; // Create local storage sized to the shared prefix.
    std::memcpy(&callbacks, runtimeRegistration, sizeof(callbacks)); // Copy the known prefix rather than reading a potentially larger registration.
    result.init = callbacks.init; // Install the runtime's operation initializer.
    result.free = callbacks.free; // Install the runtime's operation cleanup function.
    result.prepare = callbacks.prepare; // Install the runtime's tensor preparation function.
    result.invoke = callbacks.invoke; // Install the runtime's operation invocation function.
    // AddCustom sets the custom name, builtin code and operator version.
    // Newly added fields, notably registration_external, remain null/zero.
    return result; // Return the modern registration with the copied callbacks.
} // End the registration adaptation routine.
}  // namespace rps_coral

#endif // Close the Coral registration include guard.
