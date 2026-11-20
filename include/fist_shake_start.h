#pragma once // Include the fist shake detector declarations only once per source file.

#include <algorithm> // Use clamping and range comparisons for stroke thresholds.
#include <cmath> // Use distance, exponential smoothing, and finite coordinate checks.
#include <optional> // Represent timestamps that may not have been observed yet.

// Camera coordinates: positive Y is down. A lowering counts once, and the
// fist has to return to the raised band before another lowering can count.
class FistShakeStart { // Recognize deliberate closed-fist lowering strokes across camera frames.
public: // Expose observations, progress, and reset controls.
    enum class Pose { Missing, Uncertain, Closed, Open }; // Distinguish missing data, uncertain shape, closed fist, and open hand.

    struct Sample { // Pass the observed shape and palm position for one camera frame.
        Pose pose = Pose::Missing; // Treat an observation with no classification as missing.
        double x = 0; // Store the horizontal center of the palm in pixels.
        double y = 0; // Store the vertical center of the palm in pixels, increasing downward.
        double palmSpan = 0; // Measure palm size in pixels to scale the gesture tolerances.
    }; // End the per-frame observation record.

    static constexpr int requiredStrokes = 3; // Require three confirmed downward strokes before starting a round.

    void reset() { *this = FistShakeStart{}; } // Clear all tracking history, timers, and progress.
    bool armed() const { return phase != Phase::Hold; } // Report whether the initial raised hold has been established.
    bool needsRaise() const { return phase == Phase::Raise; } // Report whether another stroke requires returning to the raised pose.
    int count() const { return strokes; } // Expose the number of completed lowerings.
    const char* recoveryPrompt() const { return recovery; } // Expose a suggested recovery action while waiting for valid observations.
    const char* resetReason() const { return reason; } // Expose the explanation for the last detector restart.

    // Time is monotonic seconds. Returns true only on the third lowering.
    bool update(double now, const Sample& sample) { // Consume one observation and report completion of a three-stroke sequence.
        if (strokes == requiredStrokes) return false; // Emit completion only once before the caller resets the detector.
        if (!std::isfinite(now)) { // Reject a clock reading that could corrupt elapsed-time checks.
            reset(); // Clear any state tied to the invalid time.
            return false; // Refuse to count a stroke from this observation.
        } // End the invalid time branch.
        if (lastTracking && now < *lastTracking) { // Detect a monotonic time regression if tracking has already started.
            reset(); // Restart before attempting to use older position data.
        } // End the time-order check.
        if (armed()) { // Apply the active sequence timeouts only after the raised hold is confirmed.
            if (now - *lastTracking > trackingGrace) { // Check how long the detector has gone without a usable hand position.
                restart(recovery == repositionPrompt // Select a reset reason based on whether the last hand was outside the play area.
                        ? "Fist moved out of the play area. Raise it again." // Prompt repositioning after prolonged motion outside the accepted area.
                        : "Fist tracking lost. Raise it again."); // Otherwise explain that the detector lost the hand.
            } else if (now - *lastClosed > poseGrace) { // Check whether a trackable hand has lacked a reliable closed shape for too long.
                restart("Fist shape was unclear. Raise a closed fist again."); // Restart the sequence if the hand shape cannot be confirmed.
            } else if (strokes > 0 && now - lastStroke > strokeTimeout) { // Limit the time available to complete the next lowering after the previous one.
                restart("Too long between strokes. Raise your fist again."); // Restart an incomplete sequence when the player pauses too long.
            } // End the timeout checks for the active sequence.
        } // Finish processing the active timeout conditions.

        const bool visible = sample.pose != Pose::Missing && // Treat a missing hand as unusable even if its numeric fields are filled.
            std::isfinite(sample.x) && std::isfinite(sample.y) && // Reject nonfinite horizontal or vertical coordinates.
            std::isfinite(sample.palmSpan) && sample.palmSpan >= 12.0; // Reject invalid or undersized palms before calculating stroke thresholds.
        if (armed()) { // Validate active observations against the initial raised position and size.
            const bool near = visible && positionMatches(sample); // Require the palm to be both visible and within the allowed play area.
            if (!near) { // Pause the active sequence if this frame fails the position check.
                recovery = visible ? repositionPrompt : "KEEP YOUR FIST VISIBLE"; // Explain how to recover depending on whether the palm is still visible.
                openSince.reset(); // Stop timing continuous open-hand exposure across a lost observation.
                clearBand(); // Discard a partially confirmed lowering or raise from before the tracking gap.
                return false; // Preserve the accumulated strokes until a timeout rather than count this frame.
            } // End the invalid-position branch.
            lastTracking = now; // Mark the visible hand as recently tracked.
            if (sample.pose == Pose::Open) { // Handle a visible open palm separately from an uncertain classification.
                if (!openSince) openSince = now; // Remember when the hand first appeared open.
                if (now - *openSince >= openGrace) restart("Hand opened. Raise a closed fist again."); // Restart if the open hand persists beyond the allowed interruption.
                else recovery = "KEEP YOUR FIST CLOSED"; // Otherwise prompt the player to close the hand without erasing all progress.
                clearBand(); // Clear confirmation so an open hand cannot complete a stroke.
                return false; // Wait for a closed fist before considering movement again.
            } // End open-hand handling.
            openSince.reset(); // Clear the open-hand timer when the palm is no longer classified as open.
            if (sample.pose != Pose::Closed) { // Wait out uncertain classifications without adding a stroke.
                // The fist remains in view. Wait for confirmed closed-fist
                // samples before counting or rearming the next stroke.
                recovery = "KEEP YOUR FIST CLOSED"; // Prompt the player to keep the fist closed.
                return false; // Keep the active sequence without counting this frame.
            } // End the uncertain shape branch.
            recovery = nullptr; // Clear the recovery prompt once a usable closed fist is back.
        } else { // Otherwise establish the initial raised hold.
            if (!visible || sample.pose != Pose::Closed) { // Require a visible closed fist before starting or extending the initial hold.
                lastTracking.reset(); // Clear the last tracked timestamp on failed initial acquisition.
                lastClosed.reset(); // Clear the last closed timestamp so the next valid observation starts a fresh hold.
                return false; // Return without accumulating motion while the hand is unconfirmed.
            } // End the invalid initial observation branch.
            if (!lastClosed || now - *lastClosed > 0.20 || !positionMatches(sample)) { // Begin a new hold if the preceding observation is stale or the fist moved too far.
                seed(now, sample); // Use the current observation as the starting palm position.
                return false; // Wait for additional steady observations before arming.
            } // End initial position seeding.
        } // Finish selecting the acquisition or active tracking path.

        const double dt = now - *lastClosed; // Measure elapsed time since the previous usable closed fist observation.
        if (dt <= 0.0) return false; // Ignore a repeated or backwards timestamp.
        lastTracking = lastClosed = now; // Record this frame as both tracked and reliably closed.
        const double alpha = 1.0 - std::exp(-dt / 0.060); // Calculate a time-based filter weight with a 60 ms time constant.
        filteredX += alpha * (sample.x - filteredX); // Smooth the horizontal center before comparing positions.
        filteredY += alpha * (sample.y - filteredY); // Smooth the vertical center before detecting a raise or lowering.

        if (phase == Phase::Hold) { // Confirm a steady raised hold before counting any movement.
            if (std::hypot(filteredX - raisedX, filteredY - raisedY) > // Measure how far the filtered palm has moved from the seeded position.
                std::max(7.0, raisedSpan * 0.20)) { // Restart the hold beyond seven pixels or 20 percent of the palm size, whichever is larger.
                seed(now, sample); // Use the moving palm's current location as the new seed.
            } else if (now - holdSince >= 0.20) { // Arm after the fist stays within the raised position tolerance for 200 ms.
                raisedX = filteredX; // Capture the filtered horizontal center of the raised fist.
                raisedY = filteredY; // Capture the filtered vertical center as the lowering baseline.
                raisedSpan = sample.palmSpan; // Capture palm size to normalize movement for camera distance.
                phase = Phase::Lower; // Begin watching for a lowering after the raised hold is confirmed.
                reason = nullptr; // Clear any previous reset explanation when the detector arms again.
            } // End the initial hold stability checks.
            return false; // The frame that establishes the raised position does not count as a stroke.
        } // End the raised hold phase.

        const double drop = std::max(17.0, raisedSpan * 0.42); // Require at least 17 pixels of downward travel or 42 percent of palm size, whichever is larger.
        const double offset = filteredY - raisedY; // Measure the vertical displacement from the raised baseline.
        if (phase == Phase::Lower) { // Look for the next downward stroke only while in the lowering phase.
            if (!confirmBand(now, offset >= drop)) return false; // Require the palm to remain below the drop threshold long enough to confirm.
            ++strokes; // Count one confirmed downward stroke.
            lastStroke = now; // Timestamp the stroke for the inter-stroke timeout.
            phase = Phase::Raise; // Require the palm to come back up before another lowering can count.
            return strokes == requiredStrokes; // Report success when the third lowering has been counted.
        } // End downward stroke detection.

        if (confirmBand(now, offset <= drop * 0.45)) { // Confirm that the palm has returned close enough to its raised baseline.
            // Let the raised pose follow a small amount of natural drift.
            raisedX = filteredX; // Follow small horizontal shifts on a confirmed return.
            raisedY += std::clamp(offset, -drop * 0.5, drop * 0.45) * 0.5; // Nudge the baseline vertically while limiting gradual drift.
            phase = Phase::Lower; // Look for another lowering after the fist returns to the raised band.
        } // End the confirmed return branch.
        return false; // A raise can rearm the detector but never starts a round by itself.
    } // End processing of the observation.

private: // Keep all gesture thresholds and transition state internal.
    static constexpr double trackingGrace = 0.70; // Allow a brief loss of tracking for 0.70 seconds.
    static constexpr double poseGrace = 1.10; // Allow temporary shape uncertainty for up to 1.10 seconds.
    static constexpr double openGrace = 0.28; // Allow up to 0.28 seconds of continuous open-hand classification.
    static constexpr double strokeTimeout = 5.0; // Allow up to five seconds between counted strokes.
    static constexpr char repositionPrompt[] = "MOVE FIST BACK INTO VIEW"; // Store the fixed recovery text for a fist outside the allowed area.
    enum class Phase { Hold, Lower, Raise }; // Separate the initial hold, downward stroke, and return movement.
    Phase phase = Phase::Hold; // Begin by waiting for a steady raised fist.
    std::optional<double> lastTracking; // Remember the last usable hand position timestamp.
    std::optional<double> lastClosed; // Remember the last usable closed-fist timestamp.
    std::optional<double> bandSince; // Remember when current confirmation in a movement band began.
    std::optional<double> lastBandSample; // Remember the latest sample that was inside the confirmation band.
    std::optional<double> openSince; // Remember when a visible hand first appeared open.
    const char* recovery = nullptr; // Hold the current guidance text if recovery is needed.
    const char* reason = nullptr; // Hold the reason from the last timeout or interruption.
    double holdSince = 0; // Time the steady raised hold from its initial observation.
    double lastStroke = 0; // Time the maximum pause after the most recent counted lowering.
    double raisedX = 0; // Store the raised palm's horizontal baseline.
    double raisedY = 0; // Store the raised palm's vertical baseline.
    double raisedSpan = 0; // Store the palm size used to scale movement tolerances.
    double filteredX = 0; // Store the smoothed horizontal palm coordinate.
    double filteredY = 0; // Store the smoothed vertical palm coordinate.
    int strokes = 0; // Count confirmed downward strokes in the active sequence.

    void seed(double now, const Sample& sample) { // Seed the raised position and timestamps from a usable closed fist.
        holdSince = now; // Start timing the new initial hold.
        lastTracking = lastClosed = now; // Record the seed frame as tracked and closed.
        raisedX = filteredX = sample.x; // Initialize the horizontal baseline and filter to the measured coordinate.
        raisedY = filteredY = sample.y; // Initialize the vertical baseline and filter to the measured coordinate.
        raisedSpan = sample.palmSpan; // Use the measured hand size for the initial hold tolerance.
    } // End raised pose seeding.

    bool positionMatches(const Sample& sample) const { // Test whether the current hand is still in the expected play area.
        return sample.palmSpan >= raisedSpan * 0.48 && // Reject a palm smaller than 48 percent of the raised palm reference.
               sample.palmSpan <= raisedSpan * 2.15 && // Reject a palm larger than 215 percent of the reference.
               std::abs(sample.x - raisedX) <= std::max(40.0, raisedSpan * 1.35); // Limit lateral displacement to 40 pixels or 135 percent of palm size, whichever is larger.
    } // End the play area check.

    void restart(const char* message) { // Clear sequence state while preserving an explanation for the caller.
        reset(); // Discard the current hold, timestamps, and counted strokes.
        reason = message; // Save the reason after reset clears the detector state.
    } // End the restart helper.

    void clearBand() { // Discard any in-progress confirmation of movement.
        bandSince.reset(); // Clear the start of the confirmation interval.
        lastBandSample.reset(); // Clear the timestamp of the most recent sample in the band.
    } // End confirmation reset.

    bool confirmBand(double now, bool inBand) { // Confirm movement after a brief run of samples in the required position band.
        if (!inBand) { // A position outside the band invalidates the current run.
            clearBand(); // Forget the previous band dwell.
            return false; // Return without confirming this raise or lowering.
        } // End the out-of-band branch.
        // Carry confirmation across brief classification flicker. A gap
        // beyond this window needs two new observations in the band.
        if (!lastBandSample || now - *lastBandSample > 0.15) bandSince = now; // Start a new confirmation run after a gap over 150 ms.
        lastBandSample = now; // Record the current in-band sample for the next gap check.
        if (now - *bandSince < 0.055) return false; // Require the in-band observations to span at least 55 ms.
        clearBand(); // Consume the completed run so it cannot confirm another motion.
        return true; // Report the completed raise or lowering to the state machine.
    } // End motion confirmation.
}; // End the fist shake detector class.
