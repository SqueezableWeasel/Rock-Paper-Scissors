#include <opencv2/core.hpp>  // Provides OpenCV matrices, points, and rectangle types.
#include <opencv2/imgproc.hpp>  // Provides image transforms, color conversion, and overlay drawing.
#include <opencv2/imgcodecs.hpp>  // Provides JPEG decoding for the camera stream.
#include "stretch_display.h"  // Provides the Qt window for camera frames and keyboard input.
#include "tpu_input.h"  // Packs camera pixels into a quantized Coral model input.
#include "edgetpu.h"  // Declares the Coral Edge TPU runtime interface.
#include "coral_registration_compat.h"  // Adapts the Coral custom operator to the linked TensorFlow Lite ABI.
#include <cstdlib>  // Provides getenv for the model directory override.
#include <initializer_list>  // Provides shape lists for locating model output tensors.
#include <array>  // Provides fixed storage for the quantization table.

#include <tensorflow/lite/model.h>  // Loads TensorFlow Lite FlatBuffer model files.
#include <tensorflow/lite/interpreter.h>  // Provides the TensorFlow Lite execution and tensor API.
#include <tensorflow/lite/interpreter_builder.h>  // Builds an interpreter from a model and operator resolver.
#include <tensorflow/lite/kernels/register.h>  // Provides the built-in operator resolver.

#include <iostream>  // Provides console status and error streams.
#include <memory>  // Provides owning pointers and shared TPU context lifetime.
#include <vector>  // Provides storage for detections, landmarks, and JPEG bytes.
#include <deque>  // Provides the time-ordered history of gesture samples.
#include <map>  // Provides ordered counts for gesture votes.
#include <algorithm>  // Provides sorting, searches, and bounds helpers.
#include <chrono>  // Provides monotonic time for inference and gesture timing.
#include <cmath>  // Provides geometry, exponentials, and finite-value checks.
#include <cstring>  // Provides standard C string and memory declarations.
#include <cstdio>  // Provides the camera pipe's C stream interface.
#include <string>  // Provides gesture labels and model paths.
#include <cstdint>  // Provides fixed-width types for the quantized tensors.
#include <csignal>  // Provides process termination signals.
#include <unistd.h>  // Provides Linux pipes, process creation, and file descriptors.
#include <sys/wait.h>  // Provides waitpid for reaping the camera process.

using namespace std;  // Allows unqualified names for standard library types used below.

// ============================================================
// CONFIGURATION
// ============================================================

constexpr int PALM_SIZE = 192;  // Sets the palm model input width and height in pixels.
constexpr int HAND_SIZE = 224;  // Sets the landmark model input width and height in pixels.
constexpr int NUM_ANCHORS = 2016;  // Matches the palm model's number of detection locations.

constexpr float PI = 3.14159265358979323846f;  // Converts joint angles from radians to degrees.


float palmThreshold = 0.65f;  // Sets the adjustable minimum palm detector score.
constexpr float presenceThreshold = 0.65f;  // Requires a stronger landmark presence score before accepting a hand.

constexpr float CROP_BOX_SCALE = 2.2f;  // Expands the palm box to include the hand and fingers.
constexpr float CROP_LENGTH_SCALE = 2.6f;  // Sets a minimum crop size from wrist-to-middle-knuckle distance.

// ============================================================
// STRUCTURES
// ============================================================

struct Anchor {  // Holds an anchor's normalized position on the model grid.
    float x;  // Stores the horizontal coordinate within the model input.
    float y;  // Stores the vertical coordinate within the model input.
};  // Completes the anchor coordinate record.

struct Palm {  // Holds one detected palm in camera coordinates.
    cv::Rect2f box;  // Stores the camera-space rectangle used to size the crop.
    cv::Point2f keypoints[7];  // Stores the seven reference points used to orient the crop.
    float score = 0.0f;  // Stores the palm detector probability for ranking.
};  // Completes the palm detection record.

struct Hand {  // Holds the landmark model result for one frame.
    vector<cv::Point2f> landmarks;  // Stores joint locations in camera pixel coordinates.
    float presence = 0.0f;  // Retains the model's raw probability for diagnostics.
    bool valid = false;  // Marks results that passed presence and coordinate checks.
};  // Completes the hand result record.

// ============================================================
// TEMPORAL GESTURE STABILIZER
// ============================================================

class GestureStabilizer {  // Stabilizes per-frame labels using a short observation history.
public:  // Exposes the clock and the sample record type.
    using Clock = chrono::steady_clock;  // Uses a monotonic clock so wall-clock changes do not affect gesture timing.

    struct Sample {  // Pairs one recognized label with its observation time.
        string gesture;  // Stores the per-frame gesture label.
        Clock::time_point timestamp;  // Records when the sample entered the history.
    };  // Completes the time-stamped sample record.

private:  // Keeps history and voting policy private to the stabilizer.
    deque<Sample> history;  // Stores recent observations in arrival order.

    string stableGesture = "UNKNOWN";  // Starts without a confirmed gesture.

    static constexpr int WINDOW_MS = 800;  // Retains at most 800 milliseconds of votes.
    static constexpr int MIN_OBSERVATION_MS = 350;  // Requires the gesture history to span at least 350 milliseconds.
    static constexpr int MIN_SAMPLES = 2;  // Requires more than one observation before voting.
    static constexpr int LOST_HAND_TIMEOUT_MS = 450;  // Drops the previous gesture after a sustained 450 ms hand loss.

    static constexpr double ACCEPT_RATIO = 0.65;  // Requires at least 65 percent agreement for the first stable label.
    static constexpr double SWITCH_RATIO = 0.80;  // Requires at least 80 percent agreement to replace a stable label.

    Clock::time_point lastHandSeen = Clock::now();  // Starts the hand-loss timer at initialization.

public:  // Exposes the operations used by the frame loop.
    string update(  // Updates the stable label from a new frame observation.
        const string& rawGesture,  // Receives the current classifier label.
        bool handDetected  // Receives whether the current hand passed validation.
    ) {  // Begins applying the history and voting rules.
        auto now = Clock::now();  // Takes one timestamp for all checks in this update.

        // Remove old samples even during tracking loss.
        while (!history.empty()) {  // Drops expired observations from the front of the queue.
            auto age = chrono::duration_cast<  // Computes elapsed time since the oldest observation.
                chrono::milliseconds>(  // Measures history age in milliseconds.
                now - history.front().timestamp  // Subtracts the oldest sample time from the current time.
            ).count();  // Extracts the numeric oldest-sample age.

            if (age <= WINDOW_MS)  // Keeps the first sample still inside the voting window.
                break;  // Leaves the pruning loop since newer samples are younger.

            history.pop_front();  // Removes an expired vote.
        }  // Ends the oldest-first history pruning loop.

        // Handle temporary tracking loss.
        if (!handDetected) {  // Handles a frame without an accepted hand.
            auto missingTime = chrono::duration_cast<  // Computes the time since the last accepted hand.
                chrono::milliseconds>(  // Measures tracking loss in milliseconds.
                now - lastHandSeen  // Compares against the last valid observation time.
            ).count();  // Extracts the numeric duration of tracking loss.

            if (missingTime > LOST_HAND_TIMEOUT_MS) {  // Resets the vote history only after the loss timeout.
                stableGesture = "UNKNOWN";  // Removes the previously accepted gesture.
                history.clear();  // Discards all votes collected before the long interruption.
            }  // Ends the sustained-loss reset block.

            return stableGesture;  // Preserves the stable label during a brief tracking interruption.
        }  // Ends the path for frames without an accepted hand.

        lastHandSeen = now;  // Refreshes the most recent valid hand time.

        // Add the current classification.
        history.push_back({rawGesture, now});  // Appends the current label with its observation timestamp.

        if (history.size() < MIN_SAMPLES)  // Checks whether enough observations exist to vote.
            return stableGesture;  // Keeps the existing label while more evidence arrives.

        // Count votes. UNKNOWN remains in the denominator.
        map<string, int> votes;  // Counts votes by recognized label.

        for (const auto& sample : history) {  // Visits each observation remaining in the time window.
            if (sample.gesture != "UNKNOWN" &&  // Excludes an indeterminate classifier result from label totals.
                sample.gesture != "NO HAND") {  // Also excludes any no-hand result from label totals.
                votes[sample.gesture]++;  // Increments the count for this recognized label.
            }  // Ends the recognized-label vote branch.
        }  // Ends the scan over the gesture history.

        if (votes.empty())  // Checks for any recognized labels to compare.
            return stableGesture;  // Keeps the current label when no recognized vote exists.

        string candidate = "UNKNOWN";  // Starts the candidate search without a winning label.
        int bestVotes = 0;  // Tracks the largest number of votes found.

        for (const auto& entry : votes) {  // Compares the vote totals across candidate labels.
            if (entry.second > bestVotes) {  // Replaces the candidate only when a label has more votes.
                bestVotes = entry.second;  // Stores this label's vote count as the new maximum.
                candidate = entry.first;  // Remembers which label received the most votes.
            }  // Ends the winning-candidate update.
        }  // Ends the vote comparison loop.

        double ratio =  // Calculates the fraction of observations supporting the candidate.
            static_cast<double>(bestVotes) /  // Uses floating-point arithmetic for the winning vote count.
            static_cast<double>(history.size());  // Includes uncertain observations in the denominator.

        auto observationTime = chrono::duration_cast<  // Measures how much observation time supports the vote.
            chrono::milliseconds>(  // Measures the observation interval in milliseconds.
            now - history.front().timestamp  // Uses the timestamp of the oldest retained sample.
        ).count();  // Extracts the numeric width of the observation interval.

        if (observationTime < MIN_OBSERVATION_MS)  // Checks the minimum observation duration before confirming a label.
            return stableGesture;  // Leaves the current label in place while the history is still short.

        // Require more agreement to switch gestures
        // than to establish the initial gesture.
        if (stableGesture == "UNKNOWN") {  // Applies the initial acquisition threshold when no label is stable.
            if (ratio >= ACCEPT_RATIO)  // Checks the first-label agreement requirement.
                stableGesture = candidate;  // Publishes the candidate as the first stable gesture.
        }  // Ends the label acquisition branch.
        else if (candidate != stableGesture) {  // Applies switching logic when a different candidate wins.
            if (ratio >= SWITCH_RATIO)  // Checks the stricter agreement threshold for changing labels.
                stableGesture = candidate;  // Replaces the stable label once the new vote is strong enough.
        }  // Ends the stable-label switching branch.

        return stableGesture;  // Returns the current stabilized gesture for the frame.
    }  // Finishes processing this observation.

    void reset() {  // Clears observations when scoring or the operator requests a reset.
        history.clear();  // Removes samples collected before the reset.
        stableGesture = "UNKNOWN";  // Requires a fresh stable label after reset.
        lastHandSeen = Clock::now();  // Restarts the lost-hand timer after reset.
    }  // Finishes restoring the initial stabilizer state.
};  // Completes the gesture stabilizer class.

// ============================================================
// UTILITIES
// ============================================================

float sigmoid(float x) {  // Converts a detector logit into a probability without large exponentials.
    if (x >= 0.0f) {  // Chooses the stable formula for nonnegative logits.
        float z = exp(-x);  // Computes a nonpositive exponential argument in this branch.
        return 1.0f / (1.0f + z);  // Returns the logistic probability for a nonnegative logit.
    }  // Ends the nonnegative-logit branch.

    float z = exp(x);  // Computes a nonpositive exponential argument for a negative logit.
    return z / (1.0f + z);  // Returns the logistic probability for a negative logit.
}  // Completes the logit conversion helper.

vector<Anchor> generateAnchors() {  // Builds anchors in the same order as the palm model outputs.
    vector<Anchor> anchors;  // Collects normalized anchor locations for the two grids.
    anchors.reserve(NUM_ANCHORS);  // Reserves the expected total to avoid reallocations.

    const int grids[] = {24, 12};  // Uses 24-by-24 and 12-by-12 anchor grids.
    const int repeats[] = {2, 6};  // Specifies anchor repeats at each cell on the two grids.

    for (int layer = 0; layer < 2; ++layer) {  // Iterates over the two anchor layers in model order.
        int grid = grids[layer];  // Selects the grid size for the current layer.

        for (int y = 0; y < grid; ++y) {  // Visits grid rows from top to bottom.
            for (int x = 0; x < grid; ++x) {  // Visits columns from left to right.
                for (int a = 0; a < repeats[layer]; ++a) {  // Emits the required repeats at each grid location.
                    anchors.push_back({  // Adds a new anchor at the center of this cell.
                        (x + 0.5f) / grid,  // Normalizes the x coordinate of the cell center.
                        (y + 0.5f) / grid  // Normalizes the y coordinate of the cell center.
                    });  // Stores the completed anchor coordinate pair.
                }  // Ends repetition at this grid cell.
            }  // Ends traversal of the current row's cells.
        }  // Ends traversal of the current grid's rows.
    }  // Ends accumulation across both anchor layers.

    return anchors;  // Returns the complete anchor sequence for model decoding.
}  // Completes anchor generation.

float iou(  // Computes intersection over union to compare candidate palm rectangles.
    const cv::Rect2f& a,  // Receives the first rectangle in camera coordinates.
    const cv::Rect2f& b  // Receives the second rectangle in the same coordinate system.
) {  // Begins calculating the overlap extents.
    float x1 = max(a.x, b.x);  // Uses the later left edge as the overlap's left boundary.
    float y1 = max(a.y, b.y);  // Uses the later top edge as the overlap's top boundary.

    float x2 = min(  // Finds the overlap's right boundary.
        a.x + a.width,  // Evaluates the first rectangle's right edge.
        b.x + b.width  // Evaluates the second rectangle's right edge.
    );  // Selects the earlier of the right edges.

    float y2 = min(  // Finds the overlap's bottom boundary.
        a.y + a.height,  // Evaluates the first rectangle's bottom edge.
        b.y + b.height  // Evaluates the second rectangle's bottom edge.
    );  // Selects the earlier of the bottom edges.

    float w = max(0.0f, x2 - x1);  // Clamps negative horizontal overlap to zero.
    float h = max(0.0f, y2 - y1);  // Clamps negative vertical overlap to zero.

    float intersection = w * h;  // Calculates the area shared by the two rectangles.
    float unionArea =  // Calculates total area covered by either rectangle.
        a.area() + b.area() - intersection;  // Subtracts the shared area once so it is not counted twice.

    return unionArea > 0.0f  // Guards against division by a zero or invalid union area.
        ? intersection / unionArea  // Returns the fraction of the union covered by both rectangles.
        : 0.0f;  // Treats degenerate rectangles as having no overlap.
}  // Completes the rectangle overlap calculation.

// ============================================================
// TENSORFLOW LITE WRAPPER
// ============================================================

// Both interpreters retain the same USB context. It outlives their destruction.
shared_ptr<edgetpu::EdgeTpuContext> sharedUsbTpu() {  // Obtains a USB TPU context shared by palm and landmark inference.
    static weak_ptr<edgetpu::EdgeTpuContext> cached;  // Retains a weak reference so the last model controls device lifetime.
    auto context = cached.lock();  // Reuses the context if another interpreter still owns it.
    if (!context) {  // Opens a device only when no active context is available.
        auto* manager = edgetpu::EdgeTpuManager::GetSingleton();  // Retrieves the Coral device manager.
        if (manager) context = manager->OpenDevice(edgetpu::DeviceType::kApexUsb);  // Opens a USB Edge TPU if the manager is available.
        cached = context;  // Records the new context so later models can reuse it.
    }  // Ends device acquisition for a missing context.
    return context;  // Returns the shared context or an empty pointer on failure.
}  // Completes the shared USB device helper.

string coralModelPath(const char* filename) {  // Resolves the on-disk path for a compiled Coral model.
    const char* configured = std::getenv("CORAL_MODEL_DIR");  // Reads the optional model-directory override from the environment.
    string directory = configured && *configured ? configured : "coral_hand_models";  // Falls back to the project's default model directory for an empty override.
    return directory + "/" + filename;  // Appends the model filename to the directory.
}  // Completes model path resolution.

class TFLiteModel {  // Owns one compiled TensorFlow Lite model and its interpreter.
    // Members are destroyed in reverse declaration order.
    shared_ptr<edgetpu::EdgeTpuContext> tpuContext;  // Keeps the USB device alive through interpreter destruction.
    unique_ptr<tflite::FlatBufferModel> model;  // Owns the FlatBuffer backing the interpreter's model graph.
    unique_ptr<tflite::Interpreter> interpreter;  // Runs inference and exposes the model tensors.
    string modelPath;  // Records the path for diagnostics.
    double inferenceMs = 0.0;  // Records the most recent TPU invocation duration.
    std::array<int8_t, 256> inputLut{};  // Maps camera channel values to signed model input bytes.
    bool biasOnlyInput = false;  // Selects the faster packing path when quantization is only a signed-byte offset.

    bool fail(const string& message) const {  // Reports a model-specific failure with a consistent return value.
        cerr << "TPU error [" << modelPath << "]: " << message << endl;  // Identifies the failing model and the reason on standard error.
        return false;  // Returns false so callers can propagate the failure directly.
    }  // Completes the failure-reporting helper.

public:  // Exposes model loading, execution, timing, and tensor access.
    bool load(const string& path, int inputSize) {  // Loads and validates a compiled model for a specific square input size.
        interpreter.reset();  // Discards the old interpreter before replacing its backing model.
        model.reset();  // Releases any previously loaded FlatBuffer.
        modelPath = path;  // Keeps the new path for subsequent diagnostics.
        cerr << "[Coral] Loading " << path << endl;  // Reports the file about to be loaded.
        model = tflite::FlatBufferModel::BuildFromFile(path.c_str());  // Opens the model file and builds a FlatBuffer model object.
        if (!model) return fail("cannot open model; run from rps_fusion or set CORAL_MODEL_DIR");  // Stops with a useful path hint if the file could not be loaded.

        const auto* graph = model->GetModel();  // Obtains the serialized graph metadata for compatibility checks.
        if (!graph->subgraphs() || graph->subgraphs()->size() != 1 ||  // Requires exactly one subgraph before reading its operations.
            !graph->operator_codes()) return fail("unexpected model structure");  // Also requires an operator-code table for identifying the compiled operation.
        const auto* operations = graph->subgraphs()->Get(0)->operators();  // Retrieves the operations in the only subgraph.
        if (!operations || operations->size() != 1) return fail("expected one compiled Edge TPU operation");  // Requires the compiler's single-operation graph layout.
        const auto opcode = operations->Get(0)->opcode_index();  // Reads the operator-code index of the compiled operation.
        if (opcode >= graph->operator_codes()->size()) return fail("invalid operation index");  // Checks the index before accessing the operator-code table.
        const auto* code = graph->operator_codes()->Get(opcode);  // Looks up the operation's declared type.
        if (!code->custom_code() || code->custom_code()->str() != edgetpu::kCustomOp)  // Verifies the custom operation name expected by the Edge TPU runtime.
            return fail("model is not Edge TPU compiled");  // Rejects a compatible-looking graph with the wrong custom operator.

        cerr << "[Coral] Opening/reusing USB TPU" << endl;  // Reports when the USB device is about to be acquired.
        tpuContext = sharedUsbTpu();  // Shares the USB accelerator with the other model interpreter.
        if (!tpuContext) return fail("cannot open USB Coral; check cable, permissions, and other running TPU programs");  // Stops if no accessible USB Coral is available.
        tflite::ops::builtin::BuiltinOpResolver resolver;  // Creates the resolver that will hold the Coral custom operation.
        auto registration = rps_coral::adaptRegistration(edgetpu::RegisterCustomOp());  // Adapts the registration callbacks to the TensorFlow Lite ABI in use.
        if (!registration.init || !registration.prepare || !registration.invoke)  // Checks the runtime supplied the essential operator callbacks.
            return fail("Coral runtime returned incomplete operation callbacks");  // Rejects incomplete registration before building an interpreter.
        resolver.AddCustom(edgetpu::kCustomOp, &registration);  // Registers the adapted operator under the compiled model's name.
        cerr << "[Coral] Building interpreter (registration adapter v2)" << endl;  // Identifies the registration adapter version in the startup log.
        if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk || !interpreter)  // Creates the interpreter and checks that construction produced a usable object.
            return fail("interpreter construction failed");  // Stops before configuring tensors if construction failed.
        interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpuContext.get());  // Supplies the Coral context to the custom operator through TensorFlow Lite.
        interpreter->SetNumThreads(1);  // Keeps TensorFlow Lite's host-side execution single-threaded.
        cerr << "[Coral] Allocating tensors" << endl;  // Reports the upcoming tensor-allocation phase.
        if (interpreter->AllocateTensors() != kTfLiteOk) return fail("tensor allocation failed");  // Allocates the model input and output tensor buffers.
        if (interpreter->inputs().size() != 1) return fail("expected one input");  // Requires one input so packing uses the correct tensor.
        const auto* input = interpreter->tensor(interpreter->inputs()[0]);  // Retrieves the input tensor metadata.
        if (!input || input->type != kTfLiteInt8 || !input->dims || input->dims->size != 4 ||  // Checks that the input is signed 8-bit with four dimensions.
            input->dims->data[0] != 1 || input->dims->data[1] != inputSize ||  // Requires a batch of one and the expected height.
            input->dims->data[2] != inputSize || input->dims->data[3] != 3 ||  // Requires the expected width and three channels.
            input->bytes != static_cast<size_t>(inputSize) * inputSize * 3 ||  // Checks that the tensor has one byte per input channel value.
            !std::isfinite(input->params.scale) || input->params.scale <= 0)  // Ensures the quantization scale supports a finite positive conversion.
            return fail("unexpected input shape, type, or quantization");  // Rejects a model whose input does not match the current packing path.

        const float multiplier = 1.0f / (255.0f * input->params.scale);  // Combines normalization by 255 with the tensor's inverse quantization scale.
        for (int pixel = 0; pixel < 256; ++pixel) {  // Precomputes the quantized result of every possible camera channel value.
            const float q = std::round(pixel * multiplier) + input->params.zero_point;  // Rounds the normalized channel, then applies the tensor zero point.
            inputLut[pixel] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q)));  // Clamps to the signed 8-bit range before converting.
        }  // Ends construction of the channel lookup table.
        biasOnlyInput = isBiasOnlyInput(inputLut);  // Checks whether the lookup table is equivalent to a constant byte bias.
        // Actual invocation verifies the hardware path before starting the camera.
        cv::Mat blank(inputSize, inputSize, CV_8UC3, cv::Scalar(0, 0, 0));  // Builds a black input at the validated model dimensions.
        cerr << "[Coral] Invoking warm-up" << endl;  // Reports the first invocation before touching the accelerator.
        if (!runBgr(blank)) return false;  // Requires one successful inference before accepting the model.
        cout << "TPU ready: " << path << " | warm-up " << inferenceMs << " ms" << endl;  // Logs the loaded model path and warm-up inference time.
        return true;  // Reports that model loading and the hardware check succeeded.
    }  // Completes loading and warming up the compiled model.

    double lastInferenceMs() const { return inferenceMs; }  // Exposes the most recent TPU invocation duration in milliseconds.

    bool runBgr(const cv::Mat& bgr) {  // Executes the model using a contiguous BGR image at its input resolution.
        if (!interpreter) return fail("model not initialized");  // Rejects execution if the model has not been initialized.
        const auto* input = interpreter->tensor(interpreter->inputs()[0]);  // Gets the metadata for the interpreter's first input.
        if (bgr.empty() || bgr.type() != CV_8UC3 || !bgr.isContinuous() ||  // Requires a nonempty contiguous three-channel byte image.
            bgr.rows != input->dims->data[1] || bgr.cols != input->dims->data[2])  // Checks the image dimensions against the tensor configuration.
            return fail("input image must be contiguous BGR at the model input size");  // Explains the required input layout on mismatch.
        auto* destination = interpreter->typed_input_tensor<int8_t>(0);  // Retrieves the writable signed-byte tensor buffer.
        if (!destination) return fail("missing input buffer");  // Rejects inference if TensorFlow Lite cannot provide an input buffer.
        packBgrTpuInput(bgr.ptr<uint8_t>(), destination, bgr.total(), inputLut, biasOnlyInput);  // Reorders BGR to RGB and quantizes directly into the input tensor.
        auto start = chrono::steady_clock::now();  // Starts timing after the input has been packed.
        const auto status = interpreter->Invoke();  // Invokes the compiled Coral model.
        inferenceMs = chrono::duration<double, milli>(chrono::steady_clock::now() - start).count();  // Records accelerator execution time separately from input preparation.
        return status == kTfLiteOk ? true : fail("inference failed");  // Reports success or logs an invocation failure.
    }  // Completes one model inference.

    int findOutput(const string& name, initializer_list<int> shape) const {  // Locates a signed-byte output by its expected name and complete shape.
        if (!interpreter) return -1;  // Reports no match when the interpreter has not been built.
        for (int index : interpreter->outputs()) {  // Searches the actual output tensor indices.
            const auto* tensor = interpreter->tensor(index);  // Gets metadata for the output under consideration.
            if (!tensor || !tensor->name || name != tensor->name || !tensor->dims ||  // Rejects missing metadata and names that do not match.
                tensor->dims->size != static_cast<int>(shape.size()) || tensor->type != kTfLiteInt8)  // Also requires the expected rank and signed-byte tensor type.
                continue;  // Continues searching after an incompatible output.
            int dimension = 0;  // Starts shape comparison at the first dimension.
            bool matches = true;  // Assumes this shape matches until a differing dimension is found.
            for (int value : shape) if (tensor->dims->data[dimension++] != value) matches = false;  // Checks each expected dimension in order.
            if (matches) return index;  // Returns the actual tensor index when all dimensions match.
        }  // Ends the search through interpreter outputs.
        cerr << "Missing expected TPU output: " << name << " in " << modelPath << endl;  // Identifies the missing tensor and model in the error log.
        return -1;  // Returns the sentinel for an unresolved output index.
    }  // Completes output discovery.

    bool readOutput(int index, vector<float>& values) const {  // Reads a quantized output and converts it back to floating-point values.
        if (!interpreter || index < 0 ||  // Checks model initialization and rejects negative tensor indices.
            std::find(interpreter->outputs().begin(), interpreter->outputs().end(), index) == interpreter->outputs().end())  // Ensures the requested tensor is an output of this interpreter.
            return fail("invalid output index");  // Reports a bad tensor request before obtaining its metadata.
        const auto* tensor = interpreter->tensor(index);  // Retrieves the output tensor metadata and data pointer.
        if (!tensor || tensor->type != kTfLiteInt8 || !tensor->data.int8 ||  // Requires signed-byte values and an accessible output buffer.
            !std::isfinite(tensor->params.scale) || tensor->params.scale <= 0)  // Checks that the inverse quantization scale is finite and positive.
            return fail("invalid output quantization");  // Rejects an incompatible output before reading it.
        values.resize(tensor->bytes);  // Resizes the destination for one value per signed-byte element.
        for (size_t i = 0; i < values.size(); ++i)  // Visits each element of the quantized tensor.
            values[i] = (static_cast<int>(tensor->data.int8[i]) - tensor->params.zero_point) * tensor->params.scale;  // Applies the inverse zero-point and scale transform.
        return true;  // Reports that all output elements were decoded.
    }  // Completes tensor output decoding.
};  // Completes the compiled model wrapper.


// ============================================================
// PALM DETECTOR
// ============================================================

class PalmDetector {  // Runs a full-frame palm detector and decodes its best candidate.
    cv::Mat inputImage;  // Reuses a square image buffer for palm detection.

public:  // Exposes model initialization, output metadata, and detection.
    TFLiteModel net;  // Owns and executes the palm detector's compiled model.
    vector<Anchor> anchors;  // Stores the spatial anchors expected by the detector outputs.
    int boxesOutput = -1;  // Records the decoded box tensor index once discovered.
    int scoresOutput = -1;  // Records the detector confidence tensor index once discovered.
    vector<float> boxesBuffer, scoresBuffer;  // Reuses buffers for dequantized boxes and confidence logits.

    bool initialize() {  // Loads the palm model and resolves its required outputs.
        anchors = generateAnchors();  // Generates anchors in the sequence expected by the model.

        if (anchors.size() != NUM_ANCHORS ||  // Checks that the generated anchor count matches the model.
            !net.load(coralModelPath("hand_detector_int8_edgetpu.tflite"), PALM_SIZE)) return false;  // Loads the compiled detector with the configured square input dimensions.
        boxesOutput = net.findOutput("Identity", {1, NUM_ANCHORS, 18});  // Matches the tensor with 18 regression values per anchor.
        scoresOutput = net.findOutput("Identity_1", {1, NUM_ANCHORS, 1});  // Matches the tensor with one confidence logit per anchor.
        return boxesOutput >= 0 && scoresOutput >= 0;  // Requires both output tensor indices to be available.
    }  // Completes initialization of the palm detector.

    vector<Palm> detect(const cv::Mat& frame) {  // Locates promising palm regions in the current camera frame.
        vector<Palm> palms;  // Collects decoded candidates that pass confidence and geometry tests.

        int w = frame.cols;  // Records the source width for resizing and unletterboxing.
        int h = frame.rows;  // Records the source height for resizing and unletterboxing.

        float scale = min(  // Computes a uniform scale that fits within the square tensor input.
            PALM_SIZE / (float)w,  // Finds the scale permitted by the source width.
            PALM_SIZE / (float)h  // Finds the scale permitted by the source height.
        );  // Uses the smaller scale to avoid clipping either dimension.

        int rw = min(PALM_SIZE, cvRound(w * scale));  // Rounds and bounds the resized width to the input size.
        int rh = min(PALM_SIZE, cvRound(h * scale));  // Rounds and bounds the resized height to the input size.

        int px = (PALM_SIZE - rw) / 2;  // Computes the horizontal offset used to center the prepared image.
        int py = (PALM_SIZE - rh) / 2;  // Computes the vertical offset used to center the prepared image.

        inputImage.create(PALM_SIZE, PALM_SIZE, CV_8UC3);  // Allocates or reuses the palm model's square BGR buffer.
        if (rw != PALM_SIZE || rh != PALM_SIZE) inputImage.setTo(cv::Scalar(0, 0, 0));  // Clears the padding when the source does not occupy the entire square.
        cv::Mat letterbox = inputImage(cv::Rect(px, py, rw, rh));  // Takes a writable view of the letterboxed image region.
        cv::resize(frame, letterbox, letterbox.size());  // Resizes directly into the centered region without a second buffer.

        if (!net.runBgr(inputImage))  // Runs palm detection on the prepared input.
            return palms;  // Reports no detections if model execution failed.

        if (!net.readOutput(boxesOutput, boxesBuffer) ||  // Reads the bounding-box and keypoint regression tensor.
            !net.readOutput(scoresOutput, scoresBuffer)) return palms;  // Requires confidence logits as well before decoding detections.
        const float* boxes = boxesBuffer.data();  // References the contiguous decoded regression values.
        const float* scores = scoresBuffer.data();  // References the contiguous decoded confidence logits.

        auto toPixel = [&](float x, float y) {  // Maps normalized square model coordinates back to camera pixels.
            return cv::Point2f(  // Constructs a camera-space point with both corrected coordinates.
                (x * PALM_SIZE - px) *  // Removes horizontal letterboxing after converting x to model pixels.
                    w / (float)rw,  // Reverses the horizontal resize scale to recover camera x.

                (y * PALM_SIZE - py) *  // Removes vertical letterboxing after converting y to model pixels.
                    h / (float)rh  // Reverses the vertical resize scale to recover camera y.
            );  // Returns the point with padding and resizing reversed.
        };  // Completes the coordinate conversion used by candidate decoding.

        for (int i = 0; i < NUM_ANCHORS; ++i) {  // Decodes candidates in the same anchor order as the model.
            float score = sigmoid(scores[i]);  // Converts the current detection logit to a probability.

            if (score < palmThreshold)  // Checks whether the candidate meets the current palm threshold.
                continue;  // Skips weak candidates before decoding their geometry.

            const float* raw = boxes + i * 18;  // Locates the candidate's 18 regression values in the flat output.
            const Anchor& a = anchors[i];  // Retrieves the grid center used as the regression origin.

            float cx = raw[0] / PALM_SIZE + a.x;  // Decodes normalized center x relative to the anchor.
            float cy = raw[1] / PALM_SIZE + a.y;  // Decodes normalized center y relative to the anchor.

            float bw = raw[2] / PALM_SIZE;  // Converts regressed width from model pixels to a normalized span.
            float bh = raw[3] / PALM_SIZE;  // Converts regressed height from model pixels to a normalized span.

            if (!isfinite(cx) || !isfinite(cy) ||  // Rejects a nonfinite decoded box center.
                !isfinite(bw) || !isfinite(bh) ||  // Rejects a nonfinite decoded box size.
                bw <= 0 || bh <= 0)  // Requires positive width and height.
                continue;  // Skips any box with invalid center or dimensions.

            Palm p;  // Starts filling one palm detection record.
            p.score = score;  // Preserves the candidate confidence for ranking.

            cv::Point2f tl = toPixel(  // Converts the top-left box corner to camera coordinates.
                cx - bw / 2,  // Offsets x by half the candidate width.
                cy - bh / 2  // Offsets y by half the candidate height.
            );  // Stores the corrected top-left corner.

            cv::Point2f br = toPixel(  // Converts the bottom-right box corner to camera coordinates.
                cx + bw / 2,  // Adds half the candidate width to the center x.
                cy + bh / 2  // Adds half the candidate height to the center y.
            );  // Stores the corrected bottom-right corner.

            p.box = cv::Rect2f(  // Creates a camera-space rectangle from the two corners.
                tl.x,  // Sets the left edge using the corrected top-left corner.
                tl.y,  // Sets the top edge using the corrected top-left corner.
                br.x - tl.x,  // Computes the pixel width between the corners.
                br.y - tl.y  // Computes the pixel height between the corners.
            );  // Stores the region used for candidate overlap and cropping.

            if (p.box.width < 20 ||  // Rejects boxes narrower than 20 camera pixels.
                p.box.height < 20 ||  // Rejects boxes shorter than 20 camera pixels.
                p.box.width > w * 0.85f ||  // Rejects unusually wide boxes relative to the frame.
                p.box.height > h * 0.85f)  // Rejects unusually tall boxes relative to the frame.
                continue;  // Skips candidates outside the plausible size range.

            for (int k = 0; k < 7; ++k) {  // Decodes each of the seven palm reference points.
                p.keypoints[k] = toPixel(  // Converts the current reference point to camera pixels.
                    raw[4 + 2*k] / PALM_SIZE + a.x,  // Uses the anchor-relative x regression at this keypoint.
                    raw[5 + 2*k] / PALM_SIZE + a.y  // Uses the paired anchor-relative y regression.
                );  // Stores the current keypoint after reversing the input transform.
            }  // Ends keypoint decoding for this candidate.

            palms.push_back(p);  // Saves the decoded candidate for confidence ranking.
        }  // Ends the scan of palm model anchors.

        sort(  // Orders the candidates before applying overlap suppression.
            palms.begin(),  // Starts the sort at the first decoded candidate.
            palms.end(),  // Includes all decoded candidates in the sort range.
            [](const Palm& a, const Palm& b) {  // Defines the comparison for a pair of palm detections.
                return a.score > b.score;  // Places higher-confidence palms earlier.
            }  // Completes the descending confidence comparator.
        );  // Applies the ordering to the candidate array.

        vector<Palm> filtered;  // Collects the candidate retained for this single-hand tracker.

        for (const auto& p : palms) {  // Reviews candidates from highest confidence downward.
            bool overlap = false;  // Starts with no overlap against a retained candidate.

            for (const auto& kept : filtered) {  // Compares against the rectangles already kept.
                if (iou(p.box, kept.box) > 0.3f) {  // Identifies an overlapping box above the suppression threshold.
                    overlap = true;  // Records that this candidate duplicates a retained region.
                    break;  // Stops once an overlapping region is found.
                }  // Ends the overlap-found branch.
            }  // Ends comparison against the retained set.

            if (!overlap)  // Keeps the candidate if it does not duplicate a retained region.
                filtered.push_back(p);  // Adds the selected palm to the filtered result.

            if (filtered.size() >= 1)  // Enforces the single-hand output limit.
                break;  // Finishes selection once one palm has been retained.
        }  // Ends the filtering pass over candidates.

        return filtered;  // Returns the selected palm, or an empty result.
    }  // Completes detection for this frame.
};  // Completes the full-frame palm detector.

// ============================================================
// HAND LANDMARK TRACKER
// ============================================================

class HandTracker {  // Tracks joints within a palm-oriented crop from the camera frame.
    cv::Mat crop;  // Reuses the perspective-corrected landmark input buffer.

public:  // Exposes model outputs, initialization, and per-frame tracking.
    TFLiteModel net;  // Owns and executes the landmark model on the shared Coral device.
    int landmarksOutput = -1;  // Stores the index of the tensor with 21 joint coordinate triples.
    int presenceOutput = -1;  // Stores the index of the hand-presence tensor.
    vector<float> landmarksBuffer, presenceBuffer;  // Reuses decoded landmark and presence buffers across frames.

    bool initialize() {  // Loads the compiled landmark model and validates required outputs.
        if (!net.load(coralModelPath("hand_landmarks_detector_int8_edgetpu.tflite"), HAND_SIZE)) return false;  // Loads the model with its expected square crop dimensions.
        // Name + shape distinguish pixel landmarks from world landmarks,
        // and presence from handedness (which has the same [1,1] shape).
        landmarksOutput = net.findOutput("Identity", {1, 63});  // Locates the image-space landmark tensor with 63 elements.
        presenceOutput = net.findOutput("Identity_1", {1, 1});  // Locates the presence scalar using its exact tensor name and shape.
        return landmarksOutput >= 0 && presenceOutput >= 0;  // Requires both landmark and presence outputs.
    }  // Completes initialization of the landmark tracker.

    Hand track(  // Estimates the hand joints for one candidate palm.
        const cv::Mat& frame,  // Receives the original camera frame as the crop source.
        const Palm& palm,  // Receives the palm box and orientation reference points.
        cv::Mat* debugCrop = nullptr  // Optionally returns a copy of the actual model crop for debugging.
    ) {  // Begins constructing the crop and running landmark inference.
        Hand hand;  // Starts with an invalid hand result until all checks succeed.

        cv::Point2f wrist = palm.keypoints[0];  // Uses the detected wrist as the crop's orientation origin.
        cv::Point2f middle = palm.keypoints[2];  // Uses the middle knuckle as the direction toward the fingers.

        cv::Point2f up = middle - wrist;  // Forms the wrist-to-middle-knuckle direction vector.

        float length = sqrt(up.dot(up));  // Measures that direction in camera pixels.

        if (length < 1.0f)  // Detects an orientation vector too short to normalize reliably.
            return hand;  // Returns the invalid result for a degenerate palm orientation.

        up *= 1.0f / length;  // Normalizes the vertical basis vector of the rotated crop.

        cv::Point2f right(up.y, -up.x);  // Rotates the up direction by 90 degrees to obtain the right basis.
        cv::Point2f down = -up;  // Reverses the up direction for the crop's downward axis.

        float size = max(  // Uses the larger palm box dimension to size a square crop.
            palm.box.width,  // Considers the bounding box's horizontal extent.
            palm.box.height  // Considers the bounding box's vertical extent.
        ) * CROP_BOX_SCALE;  // Expands the larger dimension to include the fingers and wrist.

        size = max(  // Also guarantees space based on the two orientation keypoints.
            size,  // Keeps the crop size calculated from the box as one candidate.
            length * CROP_LENGTH_SCALE  // Computes the alternative minimum from wrist-to-knuckle distance.
        );  // Takes the larger of the two size estimates.

        cv::Point2f center =  // Moves the crop center away from the wrist toward the fingers.
            wrist + up * (size * 0.35f);  // Offsets the center by 35 percent of the crop width along the up axis.

        cv::Point2f tl =  // Locates the top-left corner of the rotated square.
            center -  // Uses the crop center as the reference for this corner.
            right * (size / 2.0f) -  // Moves half the crop size to the left.
            down * (size / 2.0f);  // Moves half the crop size up.

        cv::Point2f tr =  // Locates the top-right corner of the rotated square.
            center +  // Uses the crop center as the reference for this corner.
            right * (size / 2.0f) -  // Moves half the crop size to the right.
            down * (size / 2.0f);  // Moves half the crop size up.

        cv::Point2f bl =  // Locates the bottom-left corner of the rotated square.
            center -  // Uses the crop center as the reference for this corner.
            right * (size / 2.0f) +  // Moves half the crop size to the left.
            down * (size / 2.0f);  // Moves half the crop size down.

        cv::Point2f src[3] = {tl, tr, bl};  // Defines the three source points that determine the affine transform.

        cv::Point2f dst[3] = {  // Sets the corresponding corner locations in the model tensor.
            {0, 0},  // Maps the top-left corner to the crop origin.
            {(float)HAND_SIZE, 0},  // Maps the top-right corner to the crop's top edge at full width.
            {0, (float)HAND_SIZE}  // Maps the bottom-left corner to the crop's left edge at full height.
        };  // Completes the output corner coordinates.

        cv::Mat affine =  // Builds the camera-to-model affine transform.
            cv::getAffineTransform(src, dst);  // Solves the affine mapping from the three corner pairs.

        cv::warpAffine(  // Samples the rotated crop into the square model buffer.
            frame,  // Uses the camera frame as the input image.
            crop,  // Writes into the reusable landmark crop.
            affine,  // Applies the camera-to-crop affine transform.
            cv::Size(HAND_SIZE, HAND_SIZE),  // Produces the landmark model's expected width and height.
            cv::INTER_LINEAR,  // Interpolates between source pixels when sampling.
            cv::BORDER_CONSTANT,  // Uses a constant color outside the source frame.
            cv::Scalar(0, 0, 0)  // Supplies black for areas outside the camera image.
        );  // Completes preparing the aligned crop.

        if (debugCrop)  // Checks whether the caller wants a copy for the diagnostic window.
            *debugCrop = crop.clone();  // Clones the crop so the reusable buffer cannot overwrite the preview.

        if (!net.runBgr(crop))  // Executes landmark inference on the prepared crop.
            return hand;  // Returns an invalid hand if the accelerator call failed.

        if (!net.readOutput(landmarksOutput, landmarksBuffer) ||  // Reads the per-joint coordinates from the first model output.
            !net.readOutput(presenceOutput, presenceBuffer)) return hand;  // Also reads the separate hand-presence score before using the landmarks.
        const float* landmarks = landmarksBuffer.data();  // References the decoded array of joint coordinate triples.
        hand.presence = presenceBuffer[0];  // Stores the raw presence estimate even when it is too low to accept.

        if (!isfinite(hand.presence) ||  // Rejects a nonfinite presence estimate.
            hand.presence < presenceThreshold)  // Also requires the minimum hand-presence confidence.
            return hand;  // Leaves the result invalid if the crop does not contain a credible hand.

        cv::Mat inverse;  // Reserves a matrix for the crop-to-camera transform.

        cv::invertAffineTransform(  // Inverts the transform to express landmarks in the original frame.
            affine,  // Supplies the original camera-to-crop matrix.
            inverse  // Receives the inverse transform used during landmark decoding.
        );  // Completes the inverse transformation setup.

        hand.landmarks.reserve(21);  // Reserves space for all 21 camera-space joints.
        for (int i = 0; i < 21; ++i) {  // Decodes each joint in model order.
            float x = landmarks[i * 3];  // Reads x from the current joint's coordinate triple.
            float y = landmarks[i * 3 + 1];  // Reads y from the current joint's coordinate triple.

            if (!isfinite(x) || !isfinite(y))  // Rejects coordinates that cannot be transformed safely.
                return Hand{};  // Returns an empty result if any joint coordinate is invalid.

            float originalX =  // Calculates camera x using the first row of the inverse transform.
                inverse.at<double>(0, 0) * x +  // Applies the x-to-x linear term.
                inverse.at<double>(0, 1) * y +  // Applies the y-to-x linear term.
                inverse.at<double>(0, 2);  // Adds the x translation term.

            float originalY =  // Calculates camera y using the second row of the inverse transform.
                inverse.at<double>(1, 0) * x +  // Applies the x-to-y linear term.
                inverse.at<double>(1, 1) * y +  // Applies the y-to-y linear term.
                inverse.at<double>(1, 2);  // Adds the y translation term.

            hand.landmarks.emplace_back(  // Appends the transformed joint to the result.
                originalX,  // Supplies the joint's x coordinate in the camera frame.
                originalY  // Supplies the joint's y coordinate in the camera frame.
            );  // Completes the current camera-space point.
        }  // Ends the transformation of all 21 joints.

        hand.valid = true;  // Marks the result usable after presence and coordinate checks.

        return hand;  // Returns the accepted hand and its camera-space landmarks.
    }  // Completes one palm crop's landmark inference.
};  // Completes the hand landmark tracker class.

// Runs before camera startup; exercises both models sharing one USB device.
bool runTpuSelfTest(PalmDetector& palm, HandTracker& hand) {  // Exercises both compiled models before starting the camera.
    cv::Mat palmImage(PALM_SIZE, PALM_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));  // Creates a black frame at the palm model's input size.
    cv::Mat handImage(HAND_SIZE, HAND_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));  // Creates a black crop at the landmark model's input size.
    for (int i = 0; i < 5; ++i) {  // Repeats both model invocations on the shared accelerator.
        if (!palm.net.runBgr(palmImage) || !hand.net.runBgr(handImage)) return false;  // Stops the self-test if either invocation fails.
    }  // Ends the repeated accelerator invocations.
    vector<float> values;  // Reuses a floating-point buffer for checking output values.
    for (const auto& output : vector<pair<TFLiteModel*, int>>{  // Enumerates the model and tensor index for each expected output.
            {&palm.net, palm.boxesOutput}, {&palm.net, palm.scoresOutput},  // Includes the palm regression and detector confidence outputs.
            {&hand.net, hand.landmarksOutput}, {&hand.net, hand.presenceOutput}}) {  // Includes the hand coordinates and presence outputs.
        if (!output.first->readOutput(output.second, values)) return false;  // Reads each output through the regular model decoding path.
        for (float value : values) if (!std::isfinite(value)) return false;  // Rejects a tensor if any decoded value is not finite.
    }  // Ends the output validation pass.
    cout << "PASS: both models invoked on the shared USB TPU.\n"  // Reports that both models executed successfully.
         << "Last palm inference: " << palm.net.lastInferenceMs() << " ms\n"  // Prints the palm model's most recent inference duration.
         << "Last landmark inference: " << hand.net.lastInferenceMs() << " ms\n";  // Prints the landmark model's most recent inference duration.
    return true;  // Reports that execution and output validation both passed.
}  // Completes the shared USB TPU self-test.

// ============================================================
// HAND SKELETON
// ============================================================

const int CONNECTIONS[][2] = {  // Lists the landmark index pairs used to draw the hand skeleton.
    {0,1}, {1,2}, {2,3}, {3,4},  // Connects the wrist to the thumb and along its three segments.
    {0,5}, {5,6}, {6,7}, {7,8},  // Connects the wrist to the index finger and follows its joints to the tip.
    {5,9}, {9,10}, {10,11}, {11,12},  // Connects the index and middle knuckles and draws the middle finger.
    {9,13}, {13,14}, {14,15}, {15,16},  // Connects the middle and ring knuckles and draws the ring finger.
    {0,17}, {17,18}, {18,19}, {19,20},  // Connects the wrist to the little finger and follows that finger to its tip.
    {5,9}, {9,13}, {13,17}  // Adds the cross-palm links between the four finger bases.
};  // Completes the skeleton's edge list.

void drawHand(  // Draws the accepted hand landmarks over the camera frame.
    cv::Mat& frame,  // Receives the frame to modify in place.
    const Hand& hand  // Receives camera-space joints from the landmark model.
) {  // Begins drawing the skeleton for this frame.
    if (!hand.valid)  // Checks that the current model result represents an accepted hand.
        return;  // Leaves the frame untouched if tracking failed.

    for (const auto& edge : CONNECTIONS) {  // Draws each edge of the skeleton.
        cv::line(  // Creates one line segment between a pair of connected joints.
            frame,  // Uses the current camera frame as the drawing surface.
            hand.landmarks[edge[0]],  // Uses the first endpoint from the skeleton edge.
            hand.landmarks[edge[1]],  // Uses the second endpoint from the skeleton edge.
            cv::Scalar(0, 255, 0),  // Paints the connection in green.
            2,  // Uses a thickness of two camera pixels.
            cv::LINE_AA  // Antialiases the link before fullscreen scaling.
        );  // Finishes rendering the link between the two joints.
    }  // Ends rendering the skeleton connections.

    for (int i = 0; i < 21; ++i) {  // Adds one visible marker for each camera-space landmark.
        cv::circle(  // Draws a filled circle at this joint.
            frame,  // Uses the current camera frame as the drawing surface.
            hand.landmarks[i],  // Places the marker at the transformed joint location.
            4,  // Sets the marker radius to four camera pixels.
            cv::Scalar(0, 180, 255),  // Paints the marker in orange.
            -1,  // Fills the entire marker rather than tracing its outline.
            cv::LINE_AA  // Antialiases the marker's outer edge.
        );  // Finishes drawing the current joint marker.
    }  // Ends the pass over landmark indices.
}  // Completes rendering the hand overlay.

// ============================================================
// FINGER GEOMETRY
// ============================================================

float jointAngle(  // Measures a joint angle in degrees from three camera-space points.
    cv::Point2f a,  // Receives the first point adjacent to the joint.
    cv::Point2f b,  // Receives the joint at the vertex of the angle.
    cv::Point2f c  // Receives the second point adjacent to the joint.
) {  // Begins constructing the two vectors from the vertex.
    cv::Point2f u = a - b;  // Forms the vector from the joint to the first neighboring point.
    cv::Point2f v = c - b;  // Forms the vector from the joint to the second neighboring point.

    float denominator =  // Computes the product of the vector magnitudes.
        sqrt(u.dot(u) * v.dot(v));  // Uses dot products to obtain both squared lengths.

    if (denominator < 1e-5f)  // Detects vectors too short for a reliable cosine calculation.
        return 0.0f;  // Treats degenerate joints as unextended.

    float cosine = u.dot(v) / denominator;  // Computes the cosine of the angle using the normalized dot product.

    cosine = max(  // Clamps rounding error into the valid inverse-cosine range.
        -1.0f,  // Applies the lower bound of the valid cosine range.
        min(1.0f, cosine)  // Applies the upper bound of the valid cosine range.
    );  // Stores the bounded cosine for inverse trigonometry.

    return acos(cosine) * 180.0f / PI;  // Converts the measured angle from radians to degrees.
}  // Completes the joint angle helper.

// ============================================================
// RAW GESTURE CLASSIFIER
// ============================================================

string classifyGesture(const Hand& hand) {  // Classifies the tracked finger arrangement for one frame.
    if (!hand.valid)  // Checks whether the frame contains a validated hand.
        return "NO HAND";  // Returns the explicit no-hand label when tracking failed.

    const auto& p = hand.landmarks;  // Aliases the joint coordinates for concise indexing.

    const int fingers[4][4] = {  // Maps each non-thumb finger to four consecutive model landmarks.
        {5,6,7,8},  // Lists the index finger joints from base to tip.
        {9,10,11,12},  // Lists the middle finger joints from base to tip.
        {13,14,15,16},  // Lists the ring finger joints from base to tip.
        {17,18,19,20}  // Lists the little finger joints from base to tip.
    };  // Completes the finger-to-landmark map.

    bool extended[4];  // Records whether each of the four fingers is straight.

    for (int i = 0; i < 4; ++i) {  // Tests both inner joint angles for each finger.
        const int* f = fingers[i];  // Selects the landmark indices for the current finger.

        float pip = jointAngle(  // Measures the proximal interphalangeal joint angle.
            p[f[0]], p[f[1]], p[f[2]]  // Uses the finger base and the next two joints.
        );  // Completes the proximal joint angle calculation.

        float dip = jointAngle(  // Measures the distal interphalangeal joint angle.
            p[f[1]], p[f[2]], p[f[3]]  // Uses the last three finger joints, including the fingertip.
        );  // Completes the distal joint angle calculation.

        extended[i] =  // Assigns the straightness result for the current finger.
            pip > 155.0f &&  // Requires the proximal joint to be nearly straight.
            dip > 150.0f;  // Also requires the distal joint to be nearly straight.
    }  // Ends the per-finger straightness loop.

    bool index = extended[0];  // Names the index finger state for the gesture rules.
    bool middle = extended[1];  // Names the middle finger state for the gesture rules.
    bool ring = extended[2];  // Names the ring finger state for the gesture rules.
    bool pinky = extended[3];  // Names the little finger state for the gesture rules.

    if (!index && !middle && !ring && !pinky)  // Identifies a fist when all four tested fingers are bent.
        return "ROCK";  // Returns rock for the closed-fist shape.

    if (index && middle && ring && pinky)  // Identifies an open hand when all four tested fingers are straight.
        return "PAPER";  // Returns paper for the open-hand shape.

    if (index && middle && !ring && !pinky)  // Identifies two extended fingers with the others bent.
        return "SCISSORS";  // Returns scissors for the index-and-middle-finger shape.

    return "UNKNOWN";  // Leaves other finger arrangements unclassified.
}  // Completes the per-frame gesture classifier.

// ============================================================
// MJPEG READER
// ============================================================

bool readJPEG(  // Reads a single standalone-camera frame from the MJPEG byte stream.
    FILE* pipe,  // Receives the C stream holding camera stdout.
    vector<unsigned char>& jpeg  // Receives the reusable JPEG byte buffer.
) {  // Begins scanning for one complete JPEG image.
    jpeg.clear();  // Clears bytes left by the previous image.

    bool started = false;  // Waits for a JPEG start marker before writing bytes.
    int previous = -1;  // Starts without a previous byte for marker detection.

    while (true) {  // Keeps scanning until a complete frame or a read failure.
        // The camera FILE has exactly one reader.
        int c = getc_unlocked(pipe);  // Reads one byte without locking because the pipe has only one reader.

        if (c == EOF)  // Handles an ended stream or read error.
            return false;  // Reports that no complete JPEG was read.

        if (!started) {  // Searches for an image start when outside a JPEG.
            if (previous == 0xFF && c == 0xD8) {  // Recognizes the two-byte start-of-image marker.
                jpeg.push_back(0xFF);  // Adds the marker's first byte to the new image.
                jpeg.push_back(0xD8);  // Adds the marker's second byte to the new image.
                started = true;  // Switches to copying the image payload.
            }  // Ends handling of the image start marker.
        }  // Ends the path that scans bytes before the image start.
        else {  // Copies image bytes after a start marker has been recognized.
            jpeg.push_back(  // Appends the current byte to the active JPEG.
                static_cast<unsigned char>(c)  // Removes the int type used by getc to carry its EOF sentinel.
            );  // Completes appending the image byte.

            if (previous == 0xFF && c == 0xD9)  // Recognizes the two-byte end-of-image marker.
                return true;  // Reports that one full image is ready to decode.

            if (jpeg.size() > 5 * 1024 * 1024)  // Bounds memory use when input never reaches a JPEG end marker.
                return false;  // Rejects an oversized or malformed frame.
        }  // Ends payload handling for the current byte.

        previous = c;  // Keeps this byte for the next marker check.
    }  // Ends the byte scanning loop.
}  // Completes standalone MJPEG frame extraction.

// ============================================================
// MAIN
// ============================================================
#ifndef HAND_TRACKER_LIBRARY  // Enables the standalone entry point only when building this file as a program.

int main(int argc, char** argv) {  // Runs the standalone camera and landmark preview.
    PalmDetector palmDetector;  // Creates the full-frame palm detector.
    HandTracker handTracker;  // Creates the landmark tracker that follows the detected palm.
    GestureStabilizer stabilizer;  // Creates the gesture vote history used by the preview.

    if (!palmDetector.initialize() ||  // Initializes the palm model and checks for a failure.
        !handTracker.initialize()) {  // Initializes the landmark model only if the palm model loaded.
        cerr << "Model initialization failed.\n";  // Reports that camera startup is blocked by a model error.
        return 1;  // Exits with a failure status.
    }  // Ends model initialization error handling.

    if (argc == 2 && string(argv[1]) == "--self-test")  // Recognizes the standalone command-line self-test.
        return runTpuSelfTest(palmDetector, handTracker) ? 0 : 1;  // Runs both models without starting the camera or Qt.

    cout << "Both TPU models initialized.\n";  // Confirms both TPU models initialized before camera startup.

    // --------------------------------------------------------
    // CAMERA PROCESS
    // --------------------------------------------------------

    int fds[2];  // Stores the read and write descriptors for the camera pipe.

    if (pipe(fds) != 0) {  // Creates a pipe for delivering camera JPEG bytes.
        perror("pipe");  // Prints the operating system's error if pipe creation failed.
        return 1;  // Exits without starting a camera process.
    }  // Ends pipe-creation error handling.

    pid_t cameraPid = fork();  // Forks a child process that will execute the camera utility.

    if (cameraPid < 0) {  // Handles a failed process creation.
        perror("fork");  // Reports the system error from fork.
        close(fds[0]);  // Closes the unused pipe read descriptor.
        close(fds[1]);  // Closes the unused pipe write descriptor.
        return 1;  // Exits with failure after releasing both descriptors.
    }  // Ends process-creation error handling.

    if (cameraPid == 0) {  // Uses this branch only in the new camera child process.
        setpgid(0, 0);  // Makes a separate process group for shutdown signaling.

        dup2(fds[1], STDOUT_FILENO);  // Redirects camera stdout into the pipe.

        close(fds[0]);  // Closes the read end in the child since it only sends frames.
        close(fds[1]);  // Closes the duplicated write descriptor after redirecting stdout.

        execlp(  // Replaces the child process with the camera utility found on PATH.
            "rpicam-vid",  // Selects the Raspberry Pi camera video program.
            "rpicam-vid",  // Sets the camera program's conventional argument-zero name.
            "-t", "0",  // Requests capture without an automatic timeout.
            "--width", "640",  // Requests 640 pixels of frame width.
            "--height", "480",  // Requests 480 pixels of frame height.
            "--framerate", "30",  // Sets the camera's requested frame rate to 30.
            "--codec", "mjpeg",  // Outputs individually decodable JPEG frames.
            "--nopreview",  // Disables the utility's own preview because Qt displays the frames.
            "-o", "-",  // Writes the encoded bytes to standard output.
            (char*)nullptr  // Terminates the variadic argument list with its required null pointer.
        );  // Attempts to start the camera with these settings.

        _exit(127);  // Returns a conventional exec failure status if the utility could not start.
    }  // Ends the camera-child setup branch.

    close(fds[1]);  // Closes the parent's write end so camera exit can signal EOF.

    FILE* cameraPipe =  // Wraps the parent pipe reader as a C input stream.
        fdopen(fds[0], "rb");  // Opens the pipe in binary-read mode for JPEG bytes.

    if (!cameraPipe) {  // Shuts down the child if its pipe cannot be opened for reading.
        kill(cameraPid, SIGTERM);  // Asks the camera process to terminate.
        waitpid(cameraPid, nullptr, 0);  // Reaps the child before exiting.
        close(fds[0]);  // Releases the read descriptor not owned by a stream.
        return 1;  // Exits with a startup error.
    }  // Ends camera pipe initialization error handling.

    // --------------------------------------------------------
    // MAIN LOOP
    // --------------------------------------------------------

    vector<unsigned char> jpeg;  // Reuses storage for the next JPEG pulled from the camera.

    double fps = 0.0;  // Starts the frame-rate display without an estimate.
    int64 lastTick = cv::getTickCount();  // Records the starting clock tick for frame timing.

    QApplication displayApp(argc, argv);  // Initializes the GUI and creates the Qt application object.
    displayApp.setQuitOnLastWindowClosed(false);  // Leaves shutdown under the preview loop's control.
    StretchDisplay trackerWindow("Hand Landmark Tracker");  // Creates the standalone gesture preview window.
    StretchDisplay cropWindow("Hand Crop", &trackerWindow);  // Creates a crop inspector that routes keys to the tracker window.
    trackerWindow.showFullScreen();  // Starts the main preview fullscreen.
    bool showCrop = false;  // Keeps the crop inspector disabled at startup.

    cout << "ESC/Q: Exit\n";  // Tells the operator how to exit the preview.
    cout << "C: Toggle hand crop\n";  // Documents the crop inspector shortcut.
    cout << "F: Toggle fullscreen\n";  // Documents the fullscreen shortcut.
    cout << "R: Reset stabilization\n";  // Documents the shortcut to clear gesture history.
    cout << "+/-: Adjust palm threshold\n";  // Documents how to adjust the palm confidence threshold.

    while (true) {  // Processes the camera stream until capture ends or the operator quits.
        if (!readJPEG(cameraPipe, jpeg))  // Reads the bytes for the next complete JPEG image.
            break;  // Leaves the preview loop if the read produced no complete frame.

        cv::Mat encoded(  // Wraps the encoded JPEG bytes without copying them.
            1,  // Uses one matrix row for the byte sequence.
            static_cast<int>(jpeg.size()),  // Uses the byte count as the width of the encoded matrix.
            CV_8UC1,  // Treats the encoded data as a single-channel byte array.
            jpeg.data()  // Points the matrix header at the existing JPEG buffer.
        );  // Completes the encoded JPEG view.

        cv::Mat frame = cv::imdecode(  // Decodes the camera JPEG into a new image.
            encoded,  // Supplies the encoded byte view as input.
            cv::IMREAD_COLOR  // Requests a three-channel color frame for palm and landmark inference.
        );  // Returns the decoded camera frame.

        if (frame.empty())  // Checks for an invalid or empty JPEG result.
            continue;  // Skips model inference on a frame that could not be decoded.

        // ----------------------------------------------------
        // PALM DETECTION AND LANDMARK TRACKING
        // ----------------------------------------------------

        auto palms = palmDetector.detect(frame);  // Searches the current frame for a palm.

        Hand hand;  // Defaults to no accepted hand for this frame.
        cv::Mat crop;  // Reserves a matrix to receive the optional model crop.

        if (!palms.empty()) {  // Runs landmark inference only if the palm detector found a candidate.
            hand = handTracker.track(  // Estimates a hand from the selected palm and corresponding image crop.
                frame,  // Supplies the original pixels from the camera.
                palms[0],  // Uses the highest-confidence retained palm candidate.
                &crop  // Receives a copy of the model crop for the inspector.
            );  // Stores the accepted or rejected hand result for this frame.

            if (hand.valid)  // Draws a rectangle only around a validated landmark result.
                cv::rectangle(frame, palms[0].box, cv::Scalar(255, 180, 0), 2);  // Shows the selected palm box as a two-pixel blue overlay.
        }  // Ends inference and box drawing for a detected palm.

        drawHand(frame, hand);  // Draws skeleton edges and joints for the accepted hand.

        // ----------------------------------------------------
        // RAW AND STABILIZED GESTURES
        // ----------------------------------------------------

        string rawGesture = classifyGesture(hand);  // Assigns a raw gesture label from the current frame.

        string stableGesture = stabilizer.update(  // Feeds the current classification into temporal voting.
            rawGesture,  // Supplies the classifier's per-frame label.
            hand.valid  // Supplies the hand-validity flag so missed tracking does not add a vote.
        );  // Gets the updated stable label for this frame.

        // ----------------------------------------------------
        // FPS
        // ----------------------------------------------------

        int64 nowTick = cv::getTickCount();  // Reads the current OpenCV monotonic tick counter.

        double dt =  // Calculates the time elapsed since the previous processed frame.
            (nowTick - lastTick) /  // Subtracts the previous tick count from the current count.
            cv::getTickFrequency();  // Converts elapsed ticks to seconds.

        lastTick = nowTick;  // Saves this frame's timestamp for the next rate calculation.

        if (dt > 0.0) {  // Checks that elapsed time is positive before inverting it.
            double instant = 1.0 / dt;  // Computes the instantaneous processed frame rate.

            fps = fps == 0.0  // Initializes or smooths the displayed frame rate.
                ? instant  // Uses the current measurement directly on the first valid sample.
                : 0.9 * fps + 0.1 * instant;  // Blends later measurements into a slowly changing estimate.
        }  // Ends the frame-rate update.

        // ----------------------------------------------------
        // INFORMATION PANEL
        // ----------------------------------------------------

        cv::rectangle(  // Draws a solid status background behind the text.
            frame,  // Uses the current camera image as a drawing surface.
            cv::Rect(0, 0, 350, 125),  // Reserves the upper-left area of the frame for status.
            cv::Scalar(15, 15, 15),  // Uses a nearly black fill to keep text readable.
            -1  // Fills the rectangle instead of drawing only its border.
        );  // Completes drawing the status background.

        cv::putText(  // Writes the smoothed frame rate on the first status line.
            frame,  // Uses the camera frame as the drawing surface.
            cv::format("FPS: %.1f", fps),  // Formats measured frames per second to one decimal place.
            cv::Point(12, 25),  // Sets the baseline of the first status line.
            cv::FONT_HERSHEY_SIMPLEX,  // Uses the same simple font as the other status fields.
            0.65,  // Gives the frame rate slightly larger type.
            cv::Scalar(0, 255, 255),  // Uses yellow for the frame rate.
            2  // Draws the frame-rate characters with two-pixel strokes.
        );  // Completes printing the frame rate.

        cv::putText(  // Writes the current classifier result on the second status line.
            frame,  // Uses the camera frame as the drawing surface.
            "Raw: " + rawGesture,  // Joins the raw label to its field name.
            cv::Point(12, 53),  // Positions the raw result under the frame rate.
            cv::FONT_HERSHEY_SIMPLEX,  // Uses the shared status font.
            0.60,  // Sets the size of the raw gesture text.
            cv::Scalar(0, 200, 255),  // Uses orange to distinguish the per-frame label.
            2  // Draws thicker characters for the raw label.
        );  // Completes printing the raw label.

        cv::putText(  // Writes the voting result on the third status line.
            frame,  // Uses the camera frame as the drawing surface.
            "Stable: " + (hand.valid ? stableGesture : "NO HAND"),  // Shows no hand instead of a retained label during tracking loss.
            cv::Point(12, 80),  // Positions the stable result under the raw label.
            cv::FONT_HERSHEY_SIMPLEX,  // Uses the shared status font.
            0.60,  // Matches the raw label's text size.
            cv::Scalar(0, 255, 0),  // Uses green to distinguish the stabilized result.
            2  // Draws the stable label with two-pixel strokes.
        );  // Completes printing the stable label.

        cv::putText(  // Writes the model's presence reading on the last status line.
            frame,  // Uses the camera frame as the drawing surface.
            cv::format(  // Builds the presence text from the model's raw score.
                "Hand presence: %.2f",  // Formats the score to two decimal places.
                hand.presence  // Includes the score even when the hand failed acceptance.
            ),  // Completes formatting the presence label.
            cv::Point(12, 108),  // Positions the presence diagnostic at the bottom of the panel.
            cv::FONT_HERSHEY_SIMPLEX,  // Uses the shared status font.
            0.50,  // Gives the diagnostic a smaller text size.
            cv::Scalar(255, 255, 255),  // Uses white for the presence score.
            1  // Draws the diagnostic with a thinner stroke.
        );  // Completes printing the presence diagnostic.

        // ----------------------------------------------------
        // DISPLAY
        // ----------------------------------------------------

        trackerWindow.present(frame);  // Submits the annotated camera frame to the main window.

        if (showCrop && !crop.empty()) {  // Updates the crop inspector when enabled and a crop is available.
            cropWindow.present(crop);  // Submits the aligned model crop as its own display image.
            if (!cropWindow.isVisible()) cropWindow.reopen();  // Makes the crop window visible again if it was hidden.
        }  // Ends the crop display update.
        else if (!showCrop && cropWindow.isVisible()) {  // Checks whether an open crop inspector was toggled off.
            cropWindow.hide();  // Hides the inspector while leaving the main window open.
        }  // Ends inspector hiding after the toggle.

        // ----------------------------------------------------
        // KEYBOARD CONTROLS
        // ----------------------------------------------------

        int key = trackerWindow.pollKey();  // Processes Qt events and retrieves the next available key.
        if (cropWindow.wasClosed()) showCrop = false;  // Turns off the inspector setting if its window was closed.

        if (key == 'f' || key == 'F') {  // Recognizes either spelling of the fullscreen shortcut.
            trackerWindow.toggleFullscreen();  // Switches the tracker window between fullscreen and normal.
        }  // Ends fullscreen shortcut handling.

        if (key == 27 || key == 'q' || key == 'Q')  // Accepts Escape or either case of Q as a quit request.
            break;  // Leaves the frame loop so the camera can be stopped.

        if (key == 'c')  // Recognizes the crop inspector shortcut.
            showCrop = !showCrop;  // Flips the current visibility setting for the inspector.

        if (key == 'r')  // Recognizes the request to clear gesture history.
            stabilizer.reset();  // Resets temporal votes and the stable gesture.

        if (key == '+' || key == '=') {  // Handles either key commonly used to increase the threshold.
            palmThreshold = min(  // Raises palm confidence while respecting the upper bound.
                0.95f,  // Limits the detector threshold to 0.95.
                palmThreshold + 0.05f  // Increases the previous threshold by 0.05.
            );  // Stores the capped higher threshold for the next frame.
        }  // Ends handling of the threshold-increase shortcut.

        if (key == '-' || key == '_') {  // Handles either key commonly used to decrease the threshold.
            palmThreshold = max(  // Lowers palm confidence while respecting the lower bound.
                0.10f,  // Limits the detector threshold to at least 0.10.
                palmThreshold - 0.05f  // Decreases the previous threshold by 0.05.
            );  // Stores the bounded lower threshold for the next frame.
        }  // Ends handling of the threshold-decrease shortcut.
    }  // Ends preview processing before releasing camera resources.

    // --------------------------------------------------------
    // CLEAN SHUTDOWN
    // --------------------------------------------------------

    cropWindow.close();  // Closes the crop inspector during shutdown.
    trackerWindow.close();  // Closes the main tracker window.

    kill(-cameraPid, SIGTERM);  // Signals the entire camera process group to exit.
    kill(cameraPid, SIGTERM);  // Also signals the camera process itself in case its group changed.

    fclose(cameraPipe);  // Closes the parent's camera stream and its pipe descriptor.
    waitpid(cameraPid, nullptr, 0);  // Waits for the camera process to exit so it does not remain a zombie.

    return 0;  // Returns a successful exit status for a normal shutdown.
}  // Completes the standalone tracker entry point.
#endif // HAND_TRACKER_LIBRARY
