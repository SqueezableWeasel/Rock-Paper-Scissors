#include "fist_shake_start.h"  // Imports the gesture state machine under test.

#include <cassert>  // Provides assertions for each expected transition.
#include <iostream>  // Provides the terminal stream for the test result.

using Pose = FistShakeStart::Pose;  // Shorten the pose names used in the simulated observations.

struct Motion {  // Bundle the detector with a clock driven by synthetic frames.
    const int fps;  // Choose the sampling rate for this replay.
    int frame = 0;  // Track the next frame index so simulated time never jumps backward.
    int starts = 0;  // Count how often the detector requests a round.
    FistShakeStart detector;  // Preserve gesture state across successive observations.

    void feed(double seconds, double x, double y, double span, Pose pose = Pose::Closed) {  // Repeat a pose and palm measurement for the given duration.
        for (int i = 0; i < std::lround(seconds * fps); ++i) {  // Convert the duration to an integer number of frames.
            if (detector.update(static_cast<double>(frame++) / fps, {pose, x, y, span}))  // Advance simulated time and submit one observation.
                ++starts;  // Record each activation reported by the state machine.
        }  // Finish submitting this run of identical observations.
    }  // Return with the updated clock and detector state.
};  // End the synthetic motion fixture.

int main() {  // Exercise complete and interrupted activation sequences.
    for (const int fps : {15, 24, 30, 60}) {  // Repeat the valid sequence at four representative capture rates.
        Motion motion{fps, 0, 0, {}};  // Start this replay with no prior samples or activations.
        motion.feed(0.75, 310, 175, 86);  // Hold a stationary fist to establish the raised position.
        assert(motion.detector.armed() && motion.starts == 0);  // Verify the hold arms the detector without requesting a round.
        // A sideways move, a tiny dip and apparent scale change alone.
        motion.feed(0.30, 250, 176, 94);  // Move sideways at nearly the same height.
        motion.feed(0.30, 298, 186, 120);  // Combine a small drop with an apparent increase in hand size.
        assert(motion.detector.count() == 0);  // Verify neither movement counts as a full downward stroke.

        motion.feed(0.30, 304, 241, 117);  // Lower the fist well below the raised position.
        assert(motion.detector.count() == 1 && motion.detector.needsRaise());  // Expect one stroke and require the hand to return before counting again.
        motion.feed(0.80, 304, 261, 130);  // Keep the fist low to check that a stationary pose is not counted twice.
        motion.feed(0.30, 294, 216, 112);  // Partial return.
        motion.feed(0.30, 294, 256, 122);  // Lower again after an incomplete return.
        assert(motion.detector.count() == 1);  // Confirm the partial return did not rearm the next stroke.
        motion.feed(0.45, 288, 177, 82);  // Bring the fist back near its starting height.
        assert(!motion.detector.needsRaise());  // Verify the full return allows another stroke.

        // Foreshortening while turning knuckles toward the camera can yield
        // UNKNOWN, a false open label, or briefly missing landmarks.
        motion.feed(0.46, 292, 235, 124, Pose::Uncertain);  // Simulate the classifier briefly losing confidence in the pose.
        assert(motion.detector.count() == 1);  // Confirm the uncertain label preserves the first stroke.
        motion.feed(0.15, 292, 244, 124, Pose::Open);  // Simulate a momentary open-hand label on the way down.
        assert(motion.detector.count() == 1 && motion.detector.recoveryPrompt());  // Expect a recovery prompt while keeping the count.
        motion.feed(0.33, 0, 0, 0, Pose::Missing);  // Simulate a brief gap in landmark detection.
        assert(motion.detector.count() == 1 && motion.starts == 0);  // Verify a missing sample neither clears progress nor starts a round.
        motion.feed(0.30, 292, 260, 134);  // Restore a visible closed fist in the lowered position.
        assert(motion.detector.count() == 2 && motion.starts == 0);  // Count the second stroke and continue waiting for the third.
        motion.feed(0.13, 490, 310, 200);  // Brief crop jump.
        assert(motion.detector.count() == 2 && motion.detector.needsRaise());  // Keep the second count through the implausible crop displacement.
        motion.feed(0.45, 278, 174, 86);  // Return to the raised position while varying the horizontal position.
        motion.feed(0.30, 290, 261, 126);  // Perform the final downward stroke.
        assert(motion.detector.count() == 3 && motion.starts == 1);  // Expect a single activation once the third stroke lands.
        assert(motion.detector.needsRaise());  // No final return needed.
        motion.feed(0.40, 290, 261, 126);  // Continue holding the final lowered pose.
        assert(motion.starts == 1);  // Make sure the activation is emitted only once.
    }  // Finish testing the complete sequence at this frame rate.

    Motion interruptions{30, 0, 0, {}};  // Use a fresh 30 FPS fixture for interruption tests.
    interruptions.feed(0.4, 310, 175, 86);  // Arm with a closed fist at the raised position.
    interruptions.feed(0.3, 310, 265, 120);  // Count the first downward stroke.
    assert(interruptions.detector.count() == 1);  // Verify there is progress for the interruption to erase.
    interruptions.feed(0.38, 310, 240, 120, Pose::Open);  // Hold an open hand beyond the allowed interruption window.
    assert(interruptions.detector.count() == 0 && interruptions.detector.resetReason());  // Confirm the open hand erases progress and records the reason.
    interruptions.feed(0.4, 310, 175, 86);  // Restore the raised fist to start over.
    interruptions.feed(0.3, 310, 255, 120);  // Count a downward stroke for the missing-hand scenario.
    interruptions.feed(0.82, 0, 0, 0, Pose::Missing);  // Withhold landmarks long enough for tracking to expire.
    assert(interruptions.detector.count() == 0);  // Confirm sustained loss discards the partial sequence.
    interruptions.feed(0.4, 310, 175, 86);  // Reestablish the raised pose after the tracking reset.
    interruptions.feed(0.3, 310, 255, 120);  // Count a stroke before testing prolonged uncertainty.
    interruptions.feed(1.25, 310, 255, 120, Pose::Uncertain);  // Hold an uncertain pose beyond the grace period.
    assert(interruptions.detector.count() == 0);  // Verify an extended uncertain label also clears the count.
    interruptions.feed(0.4, 310, 175, 86);  // Arm again for the sequence timeout case.
    interruptions.feed(0.3, 310, 255, 120);  // Count a stroke before testing a prolonged pause.
    interruptions.feed(5.15, 310, 175, 86);  // Wait at the raised position past the time allowed between strokes.
    assert(interruptions.detector.count() == 0 && interruptions.starts == 0);  // Verify the pause clears progress without ever starting a round.

    std::cout << "PASS fist shakes at 15/24/30/60 FPS, incomplete strokes, pause, "  // Start the summary of the complete and interrupted motion cases.
                 "brief dropout, crop jumps, sustained loss/open hand, and third-stroke activation\n";  // Print the remaining checked recovery cases and finish the report.
}  // End the test with implicit success after all assertions.
