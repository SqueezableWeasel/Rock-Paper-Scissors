#ifndef RPS_TRACKING_PIPELINE_H_ // Prevent duplicate definitions of the tracking helpers and pipeline.
#define RPS_TRACKING_PIPELINE_H_ // Mark the tracking pipeline header as included.

// Included after hand_tracker.cpp; all inference remains on the main thread.
// Derive the next crop from wrist/knuckles, not fingertips: closing a fist
// must not shrink the crop so far that opening the fingers loses the hand.
inline bool palmFromLandmarks(const Hand& hand, cv::Size frame, Palm& palm) { // Build the next frame's palm crop only from accepted hand landmarks.
    if (!hand.valid || hand.landmarks.size() != 21 || !std::isfinite(hand.presence) || // Reject invalid hands, incomplete landmark sets, and nonfinite confidence.
        hand.presence < presenceThreshold) return false; // Reject confidence below the minimum hand presence threshold.
    for (const auto& point : hand.landmarks) { // Check every landmark before using the geometry or rendering it.
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || // Reject points with nonfinite coordinates.
            point.x < -0.2f * frame.width || point.x > 1.2f * frame.width || // Permit small horizontal excursions beyond the image boundary, up to 20 percent.
            point.y < -0.2f * frame.height || point.y > 1.2f * frame.height) return false; // Apply the same margin vertically and reject implausibly distant points.
    } // Finish validating the individual landmark coordinates.
    const auto& points = hand.landmarks; // Alias the already validated landmark list.
    if (cv::norm(points[9] - points[0]) < 5 || cv::norm(points[5] - points[17]) < 5) // Require both a usable wrist-to-knuckle distance and a usable palm width.
        return false; // Reject degenerate palms that would make a poor crop.
    float left = points[0].x, top = points[0].y, right = left, bottom = top; // Initialize the crop boundaries to the wrist position.
    for (int index : {1, 2, 5, 9, 13, 17}) { // Expand the crop around thumb bases and knuckles while ignoring fingertips.
        left = std::min(left, points[index].x); // Find the leftmost palm landmark.
        right = std::max(right, points[index].x); // Find the rightmost palm landmark.
        top = std::min(top, points[index].y); // Find the topmost palm landmark.
        bottom = std::max(bottom, points[index].y); // Find the lowest palm landmark.
    } // Finish finding the palm's bounding rectangle.
    const float width = right - left, height = bottom - top; // Calculate the unpadded dimensions of the proposed crop.
    if (std::max(width, height) < 8 || std::max(width, height) > std::max(frame.width, frame.height)) // Reject boxes that are too small or larger than the image's largest dimension.
        return false; // Leave the result invalid when the palm extent is implausible.
    palm.box = cv::Rect2f(left - width * 0.1f, top - height * 0.1f, width * 1.2f, height * 1.2f); // Pad both sides of each axis by 10 percent of its dimension.
    const int indices[] = {0, 5, 9, 13, 17, 1, 2}; // Map the hand landmarks to the seven keypoints expected by the palm crop.
    for (int i = 0; i < 7; ++i) palm.keypoints[i] = points[indices[i]]; // Copy the selected palm points in the expected order.
    palm.score = hand.presence; // Carry the confidence of the landmark result into the new palm estimate.
    return true; // Confirm that the next crop can be reused.
} // End the palm crop calculation.

class TrackingPipeline { // Coordinate periodic palm detection and per-frame landmark inference.
    PalmDetector& detector; // Borrow the initialized Coral palm detector.
    HandTracker& tracker; // Borrow the initialized Coral hand landmark model.
    Palm nextPalm{}; // Save the crop derived from the last accepted landmarks.
    bool havePalm = false; // Require a full palm detection for the first frame.
    int framesSinceDetection = 0; // Count reused frames since the most recent palm detection.
    const int interval; // Keep the configured palm refresh interval in frames.
    cv::Size lastSize; // Remember the frame dimensions to detect a stream size change.
    std::chrono::steady_clock::time_point lastFrame{}; // Remember when the previous frame was processed to detect a gap.
public: // Expose per-frame profiling and overlay details to the game.
    bool ranPalm = false, ranLandmarks = false; // Record which model invocations occurred in the current frame.
    double palmMs = 0, landmarkMs = 0; // Store inference durations in milliseconds for optional profiling.
    Palm displayPalm{}; // Save the accepted palm rectangle used for rendering.
    bool hasDisplayPalm = false; // Tell the caller whether the current frame has a drawable palm.

    TrackingPipeline(PalmDetector& detector, HandTracker& tracker, int interval) // Set up the pipeline with two loaded models and a refresh interval.
        : detector(detector), tracker(tracker), interval(interval) {} // Store the external model references and keep the fixed refresh interval.

    Hand process(const cv::Mat& frame) { // Process a single frame without retaining its pixel buffer.
        using Clock = std::chrono::steady_clock; // Use monotonic timing for model profiling and frame gaps.
        const auto now = Clock::now(); // Timestamp the start of current frame processing.
        if (frame.size() != lastSize || now - lastFrame > std::chrono::milliseconds(250)) havePalm = false; // Invalidate the previous crop after a resolution change or a gap over 250 ms.
        lastSize = frame.size(); // Remember the current image dimensions for the next frame.
        lastFrame = now; // Record this frame's processing time.
        ranPalm = ranLandmarks = hasDisplayPalm = false; // Clear model activity and overlay flags for the new frame.
        palmMs = landmarkMs = 0; // Clear previous timings even if a model is skipped this frame.
        Palm palm; // Hold the palm crop chosen for landmark inference.
        if (!havePalm || framesSinceDetection >= interval - 1) { // Refresh detection when no crop is reusable or the configured interval is reached.
            const auto start = Clock::now(); // Start timing the full-frame palm detector.
            auto palms = detector.detect(frame); // Run the palm detector and get candidates in descending score order.
            palmMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count(); // Measure palm detection time in milliseconds for profiling.
            ranPalm = true; // Record that a full-frame palm pass ran.
            havePalm = false; // Clear reuse status until the selected crop passes landmark validation.
            framesSinceDetection = 0; // Restart the refresh counter after every detector call.
            if (palms.empty()) return {}; // Return immediately if the detector found no plausible palm.
            palm = palms.front(); // Select the highest-ranked palm candidate.
        } else { // Reuse the landmark-based crop between detector calls.
            palm = nextPalm; // Crop the hand using the previous accepted landmark geometry.
            ++framesSinceDetection; // Advance the counter toward the next full-frame detection.
        } // Finish choosing the crop for this frame.
        const auto start = Clock::now(); // Start timing landmark inference.
        Hand hand = tracker.track(frame, palm); // Infer hand landmarks inside the selected palm crop.
        landmarkMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count(); // Measure landmark stage time in milliseconds for profiling.
        ranLandmarks = true; // Record that the landmark stage ran in this frame.
        havePalm = palmFromLandmarks(hand, frame.size(), nextPalm); // Validate the hand and prepare the crop for the next frame.
        if (havePalm) { // Publish overlay details only for a verified hand.
            displayPalm = palm; // Show the crop that produced the accepted landmarks.
            hasDisplayPalm = true; // Mark the box as safe to draw.
        } else { // Reject the visible hand when its geometry or confidence does not qualify.
            // Keep presence available for diagnostics. The rejected landmarks
            // must not be drawn or used for activation or scoring.
            hand.valid = false; // Keep failed results from entering gesture classification.
            hand.landmarks.clear(); // Remove rejected coordinates while retaining the confidence diagnostic.
        } // End the accept-or-reject branch.
        return hand; // Return the validated hand or the rejected result with its confidence.
    } // End processing of this frame.
}; // End the tracking pipeline class.
#endif // Close the tracking pipeline include guard.
