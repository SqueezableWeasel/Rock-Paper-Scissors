#include <opencv2/core.hpp>  // Provides the point, rectangle, and image types used by the fixture.
#include <vector>  // Stores the synthetic detections and landmarks.
#include <chrono>  // Provides the clock used by the frame tracking pipeline.
#include <algorithm>  // Provides minimum and maximum operations for crop bounds.
#include <cmath>  // Provides finite-value and geometry math helpers.
#include <cassert>  // Provides assertions for each acceptance and geometry condition.
#include <limits>  // Provides a known invalid floating-point coordinate.
#include <cstdio>  // Provides the output function for the test summary.
struct Palm { cv::Rect2f box; cv::Point2f keypoints[7]; float score = 0; };  // Model the palm candidate fields consumed by the production pipeline.
struct Hand { std::vector<cv::Point2f> landmarks; float presence = 0; bool valid = false; };  // Model the landmark result fields consumed by the production pipeline.
constexpr float presenceThreshold = 0.65f;  // Match the game's minimum accepted hand presence.
// Stub inference so the production tracking acceptance can be exercised.
struct PalmDetector {  // Replace inference with a controllable palm candidate.
    Palm result;  // Hold the candidate the detector will return.
    int calls = 0;  // Track palm invocations to check that invalid crops trigger reacquisition.
    std::vector<Palm> detect(const cv::Mat&) { ++calls; return {result}; }  // Count each detection call and return the configured candidate.
};  // End the fake palm detector.
struct HandTracker {  // Replace the landmark model with a configurable hand result.
    Hand result;  // Hold the observation to return for the current frame.
    Hand track(const cv::Mat&, const Palm&) { return result; }  // Return the configured observation for any frame or crop.
};  // End the fake landmark tracker.
#include "tracking_pipeline.h"  // Import the actual tracking and validation logic under test.

int main() {  // Check crop geometry and the hand acceptance path.
    Hand hand;  // Create one synthetic confident hand.
    hand.valid = true;  // Mark the fixture as a valid model result.
    hand.presence = 0.99f;  // Put presence well above the threshold for geometry assertions.
    hand.landmarks.assign(21, cv::Point2f(300, 200));  // Allocate all 21 landmarks with finite default coordinates.
    hand.landmarks[0] = {300, 300};  // Position the wrist below the knuckles.
    hand.landmarks[1] = {270, 270};  // Position the base of the thumb left of the wrist.
    hand.landmarks[2] = {260, 240};  // Place the next thumb joint farther from the wrist.
    hand.landmarks[5] = {270, 200};  // Position the index knuckle along the left edge of the palm.
    hand.landmarks[9] = {300, 190};  // Position the middle knuckle above the wrist.
    hand.landmarks[13] = {320, 200};  // Position the ring knuckle slightly to the right.
    hand.landmarks[17] = {340, 220};  // Position the little-finger knuckle to establish palm width.
    Palm first, moved, closed;  // Reserve crops for the initial, shifted, and closed-hand observations.
    assert(palmFromLandmarks(hand, {640, 480}, first));  // Build the baseline crop and verify that it succeeds.
    // Closing/opening fingertips cannot shrink the palm-derived crop.
    for (int index : {4, 8, 12, 16, 20}) hand.landmarks[index] = {300, 280};  // Move only fingertips to simulate fingers curling into a fist.
    assert(palmFromLandmarks(hand, {640, 480}, closed));  // Rebuild the crop from the same palm joints with curled fingertips.
    assert(first.box == closed.box);  // Confirm that fingertip motion leaves the palm crop unchanged.
    // Camera-space translation must propagate to the next crop without drift.
    for (auto& point : hand.landmarks) point += cv::Point2f(12, -8);  // Shift all landmarks right and up by known pixel offsets.
    assert(palmFromLandmarks(hand, {640, 480}, moved));  // Derive the tracking crop from the translated pose.
    assert(std::abs(moved.box.x - first.box.x - 12) < 0.001f);  // Verify the crop follows the horizontal offset precisely.
    assert(std::abs(moved.box.y - first.box.y + 8) < 0.001f);  // Verify the crop follows the vertical offset precisely.
    assert(moved.keypoints[0] == hand.landmarks[0]);  // Verify the tracked wrist uses the translated landmark.
    assert(moved.keypoints[2] == hand.landmarks[9]);  // Verify the tracked middle knuckle uses the translated landmark.
    const Hand goodHand = hand;  // Keep a valid hand result for reacquisition checks.
    hand.presence = 0.64f;  // Set presence just below the game's acceptance cutoff.
    assert(!palmFromLandmarks(hand, {640, 480}, moved));  // Confirm low confidence prevents the crop from being trusted.
    hand.presence = 0.99f;  // Restore high presence to isolate the next geometry failure.
    hand.landmarks[8].x = std::numeric_limits<float>::quiet_NaN();  // Inject a non-finite fingertip coordinate.
    assert(!palmFromLandmarks(hand, {640, 480}, moved));  // Require the geometry check to reject a non-finite landmark.
    hand.landmarks.resize(20);  // Simulate a result with one landmark missing.
    assert(!palmFromLandmarks(hand, {640, 480}, moved));  // Require a complete landmark set before computing a crop.

    const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));  // Supply a blank camera-sized frame to the pipeline.
    PalmDetector detector{first};  // Configure palm detection to find the baseline crop.
    HandTracker tracker{goodHand};  // Begin with the saved full set of valid landmarks.
    TrackingPipeline pipeline(detector, tracker, 10);  // Attach the stubs to the real frame-to-frame tracker.
    tracker.result.presence = 0.50f;  // Match the weak false hand confidence observed on a face.
    const auto face = pipeline.process(frame);  // Pass that observation through the complete acceptance path.
    assert(!face.valid && face.landmarks.empty() && face.presence == 0.50f);  // Verify the weak hand and its landmarks are suppressed while presence stays available.
    assert(!pipeline.hasDisplayPalm && detector.calls == 1);  // Verify the weak result adds no overlay and came from a fresh detection.

    tracker.result = goodHand;  // Restore the confident hand for the next frame.
    assert(pipeline.process(frame).valid);  // Confirm the valid hand is accepted.
    assert(pipeline.hasDisplayPalm && detector.calls == 2);  // Confirm the previous rejection forced redetection and the accepted hand has an overlay.

    tracker.result = hand;  // Feed the malformed landmark set again.
    assert(!pipeline.process(frame).valid);  // Verify the malformed result is rejected during continued tracking.
    assert(!pipeline.hasDisplayPalm);  // Verify the rejected frame has no palm overlay.
    std::puts("PASS tracking geometry, weak face rejection, and valid hand reacquisition");  // Report the geometry and weak-detection recovery checks passed.
}  // Finish the test after all assertions succeed.
