#define HAND_TRACKER_LIBRARY  // Suppress the standalone tracker's main function when including it below.
#include "hand_tracker.cpp"  // Bring the Coral model wrappers and gesture classifier into this translation unit.
#include "tracking_pipeline.h"  // Reuse landmark geometry to decide when to refresh palm detection.
#include "fist_shake_start.h"  // Include the three-stroke closed-fist detector.

#include <random>  // Use a uniformly distributed random draw for the computer's move.
#include <chrono>  // Use monotonic timestamps for round deadlines and performance measurements.
#include <iomanip>  // Format performance counters with a consistent decimal precision.
#include <cerrno>  // Inspect EINTR when reaping the camera process.

using GameClock = std::chrono::steady_clock;  // A steady clock is unaffected by wall-clock adjustments.

// ============================================================
// GAME CONFIGURATION
// ============================================================

constexpr double COUNTDOWN_SECONDS = 3.0;  // Give the player three seconds before gesture capture begins.
constexpr double CAPTURE_TIMEOUT_SECONDS = 5.0;  // End capture after five seconds if there is no reliable gesture.
constexpr double RESULT_SECONDS = 3.0;  // Keep the round result on screen for three seconds.

enum class GameState {  // Define the four phases the round advances through.
    READY,  // Wait for a fist shake or a Space press.
    COUNTDOWN,  // Show the countdown and defer scoring.
    CAPTURE,  // Wait for a confirmed rock, paper, or scissors gesture.
    RESULT  // Show the result before accepting another start.
};  // End the list of round states.

struct Game {  // Keep all round and scoreboard state together.
    GameState state = GameState::READY;  // Open the game in the ready phase.
    GameClock::time_point stateStart = GameClock::now();  // Record when the active phase began.

    std::string playerMove = "UNKNOWN";  // Leave the player's move unset until capture succeeds.
    std::string computerMove = "UNKNOWN";  // Leave the computer's move unset until a round begins.
    std::string result = "";  // Keep the result text empty before a round is scored.

    int playerScore = 0;  // Count rounds won by the player.
    int computerScore = 0;  // Count rounds won by the computer.
    int ties = 0;  // Count rounds in which both moves match.
    int rounds = 0;  // Count completed rounds separately from individual outcomes.
};  // End the state carried from frame to frame.

// ============================================================
// RANDOM COMPUTER MOVES
// ============================================================

std::mt19937 rng(std::random_device{}());  // Seed the game's pseudorandom generator from the system entropy source.

std::string randomMove() {  // Select one move for the computer to commit to.
    static const std::string moves[] = {  // Store the three labels that scoring and the HUD expect.
        "ROCK",  // Use ROCK for the closed-fist move.
        "PAPER",  // Use PAPER for the open-hand move.
        "SCISSORS"  // Use SCISSORS for the two-finger move.
    };  // End the available move table.

    std::uniform_int_distribution<int> dist(0, 2);  // Draw an inclusive index from zero through two.

    return moves[dist(rng)];  // Look up the move selected by the random index.
}  // End the computer move selection.

// ============================================================
// DETERMINE WINNER
// ============================================================

std::string determineWinner(  // Resolve a round from the two game labels.
    const std::string& player,  // Read the captured player label.
    const std::string& computer  // Read the computer's preselected label.
) {  // Begin comparing the moves.
    if (player == computer)  // Check equality first because it makes the round a tie.
        return "TIE";  // Return the label used to update the tie counter.

    if ((player == "ROCK" && computer == "SCISSORS") ||  // Rock beats scissors.
        (player == "PAPER" && computer == "ROCK") ||  // Paper beats rock.
        (player == "SCISSORS" && computer == "PAPER")) {  // Scissors beats paper.
        return "YOU WIN!";  // Return the player-win label for any of those combinations.
    }  // End the player-win cases.

    return "COMPUTER WINS!";  // The remaining valid combinations are computer wins.
}  // End scoring for the move pair.

// ============================================================
// GAME STATE HELPERS
// ============================================================

double elapsedSeconds(  // Convert a difference between steady-clock timestamps to seconds.
    GameClock::time_point start,  // Take the moment the interval began.
    GameClock::time_point now  // Take the current timestamp supplied by the caller.
) {  // Begin measuring the elapsed interval.
    return std::chrono::duration<double>(  // Convert the clock's native duration to fractional seconds.
        now - start  // Subtract the start time from the current time.
    ).count();  // Extract the numeric seconds value.
}  // End the time conversion helper.

void startRound(Game& game, GestureStabilizer& stabilizer) {  // Begin a round and clear recognition history from the previous phase.
    game.state = GameState::COUNTDOWN;  // Switch the game to its countdown display.
    game.stateStart = GameClock::now();  // Start the countdown timer from the current steady clock.

    game.playerMove = "UNKNOWN";  // Wait for the player's actual throw before assigning a move.

    // Commit to the computer's move before seeing
    // the player's gesture.
    game.computerMove = randomMove();  // Commit to a random computer move at the start of the round.

    game.result.clear();  // Remove the previous round's outcome.
    stabilizer.reset();  // Discard gestures collected before this round.
}  // End the transition into countdown.

void finishRound(  // Record a confirmed move and update the round totals.
    Game& game,  // Modify the active game's state and scoreboard.
    const std::string& playerMove,  // Receive the label that passed gesture confirmation.
    GameClock::time_point now  // Use the frame timestamp to start the result timer.
) {  // Begin scoring the confirmed throw.
    game.playerMove = playerMove;  // Save the player's move for display and logging.

    game.result = determineWinner(  // Compute the outcome from the two committed moves.
        game.playerMove,  // Compare against the verified player move.
        game.computerMove  // Compare against the computer move selected at countdown start.
    );  // Finish assigning the result label.

    if (game.result == "YOU WIN!")  // Check whether this round awards the player a point.
        ++game.playerScore;  // Add the player's win to the running total.
    else if (game.result == "COMPUTER WINS!")  // Check for a computer win when the player did not win.
        ++game.computerScore;  // Add the computer's win to its total.
    else  // Treat the remaining valid outcome as a tie.
        ++game.ties;  // Add one to the tie count.

    ++game.rounds;  // Advance the total number of scored rounds.

    game.state = GameState::RESULT;  // Switch the HUD to the result screen.
    game.stateStart = now;  // Start the result timer using the same frame time.

    std::cout  // Write a single round summary to the terminal.
        << "Round " << game.rounds  // Identify this round by its running number.
        << " | Player: " << game.playerMove  // Include the player's confirmed throw.
        << " | Computer: " << game.computerMove  // Include the computer's committed throw.
        << " | " << game.result << std::endl;  // Include the outcome and flush the completed line.
}  // End the result transition and score update.

void updateGame(  // Advance the round state using the current frame's recognition result.
    Game& game,  // Update the game state in place.
    GestureStabilizer& stabilizer,  // Use the gesture history when the countdown ends.
    const std::string& rawGesture,  // Read the label classified from this frame.
    const std::string& stableGesture,  // Read the label agreed on across recent frames.
    bool handValid,  // Check whether the tracking pipeline accepted the present hand.
    GameClock::time_point now  // Use one timestamp for all deadline checks in this update.
) {  // Begin handling the active phase.
    double elapsed = elapsedSeconds(game.stateStart, now);  // Measure the age of the active phase once per frame.

    switch (game.state) {  // Choose the transition rules for the present round state.
        case GameState::READY:  // Handle the idle phase, where start input is processed separately.
            break;  // Leave the game idle until it receives a start.

        case GameState::COUNTDOWN:  // Handle the countdown that precedes capture.
            if (elapsed >= COUNTDOWN_SECONDS) {  // Check whether the full countdown interval elapsed.
                game.state = GameState::CAPTURE;  // Begin accepting the player's throw.
                game.stateStart = now;  // Start the capture timeout from this frame.

                // Discard countdown predictions.
                // Only a post-countdown gesture can score.
                stabilizer.reset();  // Remove any labels gathered during the countdown.
            }  // Finish the countdown-to-capture transition.
            break;  // End processing for the countdown phase.

        case GameState::CAPTURE:  // Handle the phase that waits for the player's throw.
            if (handValid && rawGesture == stableGesture &&  // Require a valid current hand and agreement between current and stable labels.
                (stableGesture == "ROCK" ||  // Allow a recognized ROCK move.
                 stableGesture == "PAPER" ||  // Allow a recognized PAPER move.
                 stableGesture == "SCISSORS")) {  // Allow a recognized SCISSORS move.
                finishRound(game, stableGesture, now);  // Score the confirmed move using the current frame's timestamp.
            }  // End the confirmed-throw branch.
            else if (elapsed >= CAPTURE_TIMEOUT_SECONDS) {  // Give up if the player has not shown a reliable gesture in time.
                game.result = "NO GESTURE DETECTED";  // Display a timeout message without awarding a point.
                game.state = GameState::RESULT;  // Show the timeout using the ordinary result phase.
                game.stateStart = now;  // Start the timer for the timeout message.
            }  // End the capture-expiry branch.
            break;  // End processing for the capture phase.

        case GameState::RESULT:  // Handle the phase that shows the outcome.
            if (elapsed >= RESULT_SECONDS) {  // Check whether the result has been visible long enough.
                game.state = GameState::READY;  // Clear the footer by returning to the ready state.
                game.stateStart = now;  // Record the start time of the new ready phase.
            }  // Finish the result-to-ready transition.
            break;  // End processing for the result phase.
    }  // End dispatch for all four round phases.
}  // End the per-frame game update.

FistShakeStart::Sample fistShakeSample(const Hand& hand, const std::string& gesture) {  // Extract the pose, position, and scale needed by the shake detector.
    using Pose = FistShakeStart::Pose;  // Refer to the shake detector's pose categories without a long qualifier.
    if (!hand.valid || hand.landmarks.size() != 21)  // Require an accepted hand with all 21 landmarks.
        return {};  // Report missing hand data when tracking has rejected the candidate.
    Pose pose = Pose::Uncertain;  // Start with an uncertain pose when the classifier has no recognized label.
    if (gesture == "ROCK") pose = Pose::Closed;  // A ROCK label provides evidence of a closed fist.
    else if (gesture == "PAPER" || gesture == "SCISSORS") pose = Pose::Open;  // PAPER and SCISSORS mean the fingers are extended.

    const auto& points = hand.landmarks;  // Reference the validated points without copying them.
    cv::Point2f center;  // Start the sum of the five palm landmarks at the origin.
    for (const int joint : {0, 5, 9, 13, 17}) center += points[joint];  // Add the wrist and four knuckle landmarks to estimate the palm center.
    center *= 0.2f;  // Divide the summed coordinates by the five points.
    const double span = std::max(cv::norm(points[9] - points[0]),  // Measure wrist-to-middle-knuckle length as one estimate of palm size.
                                 cv::norm(points[17] - points[5]));  // Also measure knuckle width so orientation affects scale less.
    return {pose, center.x, center.y, span};  // Pass the classification, center, and size to the motion detector.
}  // End conversion from landmarks to a shake sample.

// ============================================================
// DRAW HELPERS
// ============================================================

void drawText(  // Draw one antialiased text label onto an OpenCV image.
    cv::Mat& image,  // Receive the image that will be modified in place.
    const std::string& text,  // Receive the label to render.
    int x,  // Position the text origin horizontally in camera pixels.
    int y,  // Position the baseline vertically in camera pixels.
    double scale,  // Control text size relative to the Hershey font.
    cv::Scalar color,  // Receive the text color in OpenCV's BGR order.
    int thickness = 2  // Use a two-pixel stroke unless the caller requests a finer line.
) {  // Begin the common text drawing wrapper.
    cv::putText(  // Render the label using OpenCV's Hershey implementation.
        image,  // Draw directly into the camera frame or supplied image.
        text,  // Pass through the prepared text without changing it.
        cv::Point(x, y),  // Use the caller's coordinates as the baseline origin.
        cv::FONT_HERSHEY_SIMPLEX,  // Use the same simple font for all game labels.
        scale,  // Apply the requested font scale.
        color,  // Apply the requested BGR color.
        thickness,  // Apply the requested stroke width.
        cv::LINE_AA  // Smooth the text edges for readability.
    );  // Finish drawing the label in place.
}  // End the text helper.

void drawGameUI(  // Add the score strip and any active-round messages to a frame.
    cv::Mat& frame,  // Receive the composed camera frame and hand overlay.
    const Game& game,  // Read the round state and score without modifying them.
    const std::string& rawGesture,  // Receive the current frame's label for diagnostics.
    const std::string& stableGesture,  // Receive the temporally stabilized label for diagnostics.
    float handPresence,  // Display the landmark model's presence score.
    double fps,  // Display the recent game-loop frame rate.
    GameClock::time_point now  // Use the current frame's time for the countdown text.
) {  // Begin composing the HUD.
    const cv::Scalar WHITE(255, 255, 255);  // Use white for neutral text.
    const cv::Scalar GREEN(80, 255, 80);  // Use bright green for scores and player wins.
    const cv::Scalar YELLOW(0, 255, 255);  // Use yellow for countdown, capture, and ties.
    const cv::Scalar ORANGE(0, 180, 255);  // Use orange for the current raw gesture label.
    const cv::Scalar RED(70, 70, 255);  // Use red for computer wins.

    // The score strip is always visible. The footer appears during a round.
    constexpr int headerHeight = 46;  // Reserve the top 46 rows of the camera image for the score strip.
    const cv::Scalar panelColor(20, 20, 20);  // Use a dark gray background for readability over the video.
    frame.rowRange(0, headerHeight).setTo(panelColor);  // Fill the top strip across the entire frame width.

    drawText(frame, "ROCK PAPER SCISSORS", 10, 18, 0.55, WHITE);  // Draw the title on the first line of the strip.

    drawText(  // Draw all three running scores on one line.
        frame,  // Place the score text into the current frame.
        cv::format(  // Format scoreboard numbers into a single string.
            "YOU: %d   CPU: %d   TIES: %d",  // Label the player wins, computer wins, and ties in that order.
            game.playerScore,  // Insert the player's win count.
            game.computerScore,  // Insert the computer's win count.
            game.ties  // Insert the tie count.
        ),  // Finish formatting the one-line scoreboard.
        288, 18, 0.52, GREEN  // Align the scoreboard to the right of the title.
    );  // Finish drawing the scoreboard.

    drawText(  // Draw the current frame's pose classification under the title.
        frame,  // Render the diagnostic onto the same frame.
        "Raw: " + rawGesture,  // Prefix the immediate classifier output with its field name.
        10, 37, 0.43, ORANGE, 1  // Use fine orange text on the second line.
    );  // Finish drawing the raw label.

    drawText(  // Draw the label selected by the temporal stabilizer.
        frame,  // Render the stabilized diagnostic into this frame.
        "Stable: " + stableGesture,  // Prefix the stabilized label for quick comparison with the raw label.
        184, 37, 0.43, GREEN, 1  // Place the green text between the raw label and FPS counters.
    );  // Finish drawing the stabilized label.

    drawText(  // Draw the throughput and presence indicators.
        frame,  // Put the indicators into the top strip.
        cv::format(  // Combine both decimal readings into a string.
            "FPS %.1f  Presence %.2f",  // Show one decimal for FPS and two for model confidence.
            fps, handPresence  // Supply the current FPS estimate and presence score.
        ),  // Finish formatting the performance indicators.
        380, 37, 0.43, WHITE, 1  // Position the indicators at the right of the second line.
    );  // Finish drawing the diagnostics row.

    if (game.state == GameState::READY) return;  // Leave the camera visible to the bottom edge between rounds.

    constexpr int footerHeight = 44;  // Reserve 44 rows for the footer during an active round.
    const int panelY = frame.rows - footerHeight;  // Anchor the footer to the bottom of the current frame.
    frame.rowRange(panelY, frame.rows).setTo(panelColor);  // Fill only the active round's footer area.

    double elapsed = elapsedSeconds(  // Compute how long the current gameplay phase has been active.
        game.stateStart, now  // Measure from the phase start to the current frame.
    );  // Finish computing phase age for message selection.

    switch (game.state) {  // Choose the message for the active round phase.
        case GameState::READY:  // Include the ready state for completeness after the early return.
            break;  // The ready screen has no footer text.

        case GameState::COUNTDOWN: {  // Handle the themed countdown.
            int remaining = std::max(  // Clamp the displayed countdown to at least one second.
                1,  // Keep showing the final word until the state machine begins capture.
                static_cast<int>(  // Convert the rounded-up time to a whole number.
                    std::ceil(  // Round any fractional second up to its countdown step.
                        COUNTDOWN_SECONDS - elapsed  // Subtract elapsed time from the countdown duration.
                    )  // Finish rounding the time remaining.
                )  // Finish converting the seconds to an integer.
            );  // Store the clamped countdown step.

            std::string message;  // Hold the word to display for the current step.

            if (remaining >= 3)  // The first countdown step has three seconds left.
                message = "ROCK...";  // Display the first word of the game title.
            else if (remaining == 2)  // The second countdown step has two seconds left.
                message = "PAPER...";  // Display the middle word of the game title.
            else  // Use the final word for the last countdown step.
                message = "SCISSORS...";  // Display SCISSORS during the final second.

            drawText(  // Draw the current countdown word in the footer.
                frame,  // Draw the countdown directly into the camera frame.
                message,  // Use the word selected for this second.
                10, panelY + 19,  // Put the countdown word on the first footer line.
                0.63, YELLOW  // Make the countdown a prominent yellow label.
            );  // Finish drawing the main countdown text.

            drawText(  // Add a short instruction beneath the countdown.
                frame,  // Draw the instruction in the same frame.
                "Get ready to throw!",  // Tell the player to prepare for their throw.
                10, panelY + 36,  // Align the instruction to the second footer line.
                0.43, WHITE, 1  // Use small white lettering to separate it from the countdown.
            );  // Finish drawing the countdown instruction.
            break;  // Finish rendering the countdown phase.
        }  // End the countdown branch and its local variables.

        case GameState::CAPTURE:  // Handle the phase where the player can throw a move.
            drawText(  // Draw the main cue that scoring input is now enabled.
                frame,  // Draw it into the current frame.
                "SHOOT! HOLD YOUR GESTURE",  // Ask the player to hold their thrown gesture.
                10, panelY + 19,  // Put the cue on the first footer line.
                0.60, YELLOW  // Use a yellow cue consistent with the countdown.
            );  // Finish drawing the throw cue.

            drawText(  // Explain why capture may take more than one frame.
                frame,  // Draw that status on the same frame.
                "Waiting for stable recognition...",  // Display a short waiting message until stabilization succeeds.
                10, panelY + 36,  // Use the second footer line for the waiting message.
                0.43, WHITE, 1  // Use small white type with a fine stroke.
            );  // Finish drawing the capture status.
            break;  // Finish rendering the capture phase.

        case GameState::RESULT: {  // Handle the round's outcome or capture timeout.
            cv::Scalar color = WHITE;  // Use white for an outcome without a special color.

            if (game.result == "YOU WIN!")  // Test for a player win.
                color = GREEN;  // Highlight a player win in green.
            else if (game.result == "COMPUTER WINS!")  // Test for a computer win.
                color = RED;  // Distinguish a computer win in red.
            else if (game.result == "TIE")  // Test for a tied round.
                color = YELLOW;  // Highlight a tie in yellow.

            drawText(  // Draw the outcome as the primary footer label.
                frame,  // Draw in the current camera frame.
                game.result,  // Pass the win, tie, or timeout message.
                10, panelY + 19,  // Put the outcome on the first footer baseline.
                0.60, color  // Use bold lettering and the outcome's selected color.
            );  // Finish drawing the primary result.

            if (game.playerMove != "UNKNOWN") {  // Show the two throws only when a player move was captured.
                drawText(  // Draw the move comparison under the outcome.
                    frame,  // Draw it in the current frame.
                    "YOU: " + game.playerMove +  // Label the player's captured move.
                    "   CPU: " + game.computerMove,  // Append the computer's preselected move.
                    10, panelY + 36,  // Align the comparison to the second footer line.
                    0.43, WHITE, 1  // Keep the comparison smaller than the outcome.
                );  // Finish drawing the move comparison.
            }  // Skip this comparison when capture timed out.
            break;  // Finish rendering the result phase.
        }  // End the result branch and its local color.
    }  // End message dispatch for the game HUD.
}  // Finish composing the frame's overlays.

// ============================================================
// CAMERA PROCESS
// ============================================================

struct CameraProcess {  // Own the child process and read end of the camera pipe.
    pid_t pid = -1;  // Use a negative process ID to mark an inactive camera.
    int streamFd = -1;  // Use a negative descriptor to mark a closed pipe.

    bool start() {  // Start rpicam-vid and connect its output to the frame worker.
        int fds[2];  // Reserve space for the read and write file descriptors.

        if (pipe(fds) != 0) {  // Create a pipe to carry MJPEG bytes from the camera.
            perror("pipe");  // Report why the operating system could not create the pipe.
            return false;  // Let the caller handle camera startup failure.
        }  // End the pipe creation error path.

        pid = fork();  // Fork a child that can replace itself with the camera program.

        if (pid < 0) {  // Check whether the operating system could create the child.
            perror("fork");  // Report the fork failure.
            close(fds[0]);  // Close the unused read descriptor on failure.
            close(fds[1]);  // Close the unused write descriptor on failure.
            return false;  // Let the caller handle the failed startup.
        }  // End the child creation error path.

        if (pid == 0) {  // Configure the camera command in the child process.
            setpgid(0, 0);  // Give the camera its own process group for shutdown.

            if (dup2(fds[1], STDOUT_FILENO) < 0)  // Redirect the child's standard output to the write end of the pipe.
                _exit(127);  // Exit the child if output cannot be redirected.

            close(fds[0]);  // The camera process does not read from this pipe.
            close(fds[1]);  // Drop the extra write descriptor after duplicating it to stdout.

            execlp(  // Replace the child with the Raspberry Pi camera application.
                "rpicam-vid",  // Find the camera executable on PATH.
                "rpicam-vid",  // Pass the executable name as argv zero.
                "-t", "0",  // Record continuously until the app stops the child.
                "--width", "640",  // Request 640 horizontal pixels.
                "--height", "480",  // Request 480 vertical pixels.
                "--framerate", "30",  // Ask the camera to deliver 30 frames per second.
                "--codec", "mjpeg",  // Produce a stream of JPEG images for the capture worker.
                "--nopreview",  // Let the game handle preview drawing.
                "-o", "-",  // Write the MJPEG stream to stdout.
                (char*)nullptr  // Terminate the argument list required by execlp.
            );  // Finish replacing the camera child.

            _exit(127);  // Exit the child if the camera executable could not run.
        }  // End setup in the child process.

        close(fds[1]);  // The parent should keep only the read end so EOF can be observed.

        streamFd = fds[0];  // Save the descriptor used by the frame worker.
        return true;  // Indicate that the camera process and pipe were started.
    }  // End camera startup.

    void stop() {  // Stop camera capture and release operating system resources.
        if (pid > 0) {  // Signal the camera only when there is a child process.
            // Signal the camera and its process group.
            kill(-pid, SIGTERM);  // Stop the whole camera process group if it exists.
            kill(pid, SIGTERM);  // Also signal the child directly in case group creation failed.
        }  // Finish signaling the child.

        if (streamFd >= 0) {  // Check that there is an open read descriptor.
            close(streamFd);  // Release the read end of the camera pipe.
            streamFd = -1;  // Mark the pipe closed so repeated cleanup is safe.
        }  // Finish closing the descriptor.

        if (pid > 0) {  // Reap the child if startup recorded its process ID.
            while (waitpid(pid, nullptr, 0) < 0 &&  // Wait for the camera process to exit.
                   errno == EINTR) {}  // Retry waitpid only when a signal interrupted it.

            pid = -1;  // Mark the child already reaped.
        }  // Finish waiting for the child.
    }  // End camera shutdown.

    ~CameraProcess() {  // Clean up the camera when the owning object leaves scope.
        stop();  // Reuse the shutdown routine on normal and early exits.
    }  // End the destructor.
};  // End the camera resource wrapper.

// ============================================================
// MAIN
// ============================================================

#include "latest_frames.h"  // Include the capture worker after defining the camera type it uses.

int main(int argc, char** argv) {  // Parse options, initialize the camera and models, and run the game.
    int palmInterval = 10;  // Refresh palm detection every ten processed frames by default.
    bool profile = false, selfTest = false;  // Start with performance logging and self-test disabled.
    bool legacyLidarOption = false;  // Remember whether the caller used the old LiDAR flag.
    try {  // Catch invalid values and incomplete options before initialization.
        for (int i = 1; i < argc; ++i) {  // Inspect each argument after the executable name.
            const std::string argument = argv[i];  // Copy the current option name for comparison.
            if (argument == "--profile") profile = true;  // Enable periodic timing and tracking diagnostics.
            else if (argument == "--camera-only") continue; // Older launch commands.
            else if (argument == "--lidar" && i + 1 < argc && argv[i + 1][0] != '-') {  // Accept an old LiDAR invocation only when its port argument is present.
                legacyLidarOption = true;  // Mark the legacy option so the app can explain its current behavior.
                ++i;  // Consume the unused port argument.
            }  // End parsing of the legacy port option.
            else if (argument == "--self-test") selfTest = true;  // Select model verification without opening the camera or display.
            else if (argument == "--palm-interval" && i + 1 < argc) {  // Read the optional palm refresh interval from the next argument.
                std::string value = argv[++i];  // Consume the interval as a string so trailing characters can be checked.
                size_t used = 0;  // Track how many characters the integer conversion consumes.
                palmInterval = std::stoi(value, &used);  // Parse the requested refresh interval in frames.
                if (used != value.size() || palmInterval < 1 || palmInterval > 60)  // Reject partial numbers and values outside the supported range.
                    throw std::invalid_argument("palm interval must be 1..60");  // Explain the permitted palm interval in the argument error.
            } else throw std::invalid_argument("unknown or incomplete argument: " + argument);  // Reject unrecognized options or options missing their value.
        }  // Finish reading the command-line arguments.
    } catch (const std::exception& error) {  // Handle parser errors before opening hardware.
        std::cerr << error.what() << "\nUsage: rock_paper_scissors [--self-test] [--profile] [--palm-interval 1..60]\n";  // Show the reason and the accepted command-line options.
        return 2;  // Use exit status two for invalid command-line arguments.
    }  // End argument error handling.
    if (legacyLidarOption)  // Explain the old LiDAR flag when someone uses it.
        std::cerr << "The game uses the camera for fist shakes. Use c1_probe to read LiDAR scans.\n";  // Direct the caller to the C1 probe for separate scan diagnostics.
    // Let camera decoding and the main loop use separate cores without
    // creating an OpenCV thread pool for every small resize/warp operation.
    cv::setNumThreads(1);  // Limit OpenCV's worker threads so camera decode and gameplay can share the processor.
    PalmDetector palmDetector;  // Create the model that finds the hand region.
    HandTracker handTracker;  // Create the model that estimates the 21 landmarks inside that region.
    GestureStabilizer stabilizer;  // Create the gesture stabilizer for the recent label history.
    FistShakeStart shake;  // Create the detector that counts the closed-fist motion sequence.
    Game game;  // Initialize game state and scores.

    if (!palmDetector.initialize() ||  // Load the palm model and check its Coral runtime.
        !handTracker.initialize()) {  // Load the landmark model if the palm model initialized.
        std::cerr << "Failed to initialize models.\n";  // Report unsuccessful model initialization.
        return 1;  // Exit before opening the camera when either model is unavailable.
    }  // End model startup checks.

    if (selfTest)  // Branch to hardware verification when requested.
        return runTpuSelfTest(palmDetector, handTracker) ? 0 : 1;  // Return the outcome of model verification to the shell.

    CameraProcess camera;  // Own the continuous camera process for this session.

    if (!camera.start()) {  // Start the stream before creating the game window.
        std::cerr << "Failed to start camera.\n";  // Explain that camera startup failed.
        return 1;  // Exit when camera capture is unavailable.
    }  // End the camera startup error path.

    std::cout  // Print startup instructions to the terminal.
        << "Rock Paper Scissors started.\n"  // Announce that the game is running.
        << "Hold up a closed fist and lower it three times, raising it between strokes.\n"  // Explain the closed-fist sequence for starting a round.
        << "SPACE: Start round manually\n"  // Describe the keyboard shortcut for starting a round.
        << "R: Reset scoreboard\n"  // Describe the score reset key.
        << "+/-: Palm confidence\n"  // Describe the palm detection confidence controls.
        << "F: Toggle fullscreen\n"  // Describe the fullscreen toggle.
        << "Q/ESC: Quit\n";  // Describe both ways to exit.

    QApplication displayApp(argc, argv);  // Initialize Qt on the main thread with this program's arguments.
    displayApp.setQuitOnLastWindowClosed(false);  // Let the game handle closing the window and shutting down its worker.
    StretchDisplay gameWindow("Rock Paper Scissors");  // Create the camera preview window with the game's title.
    gameWindow.showFullScreen();  // Start with the preview filling the display.

    LatestFrames frames(camera);  // Start the worker that decodes and publishes the newest camera frame.
    TrackingPipeline pipeline(palmDetector, handTracker, palmInterval);  // Configure the palm and landmark pipeline with the requested refresh cadence.
    std::cout << "Palm refresh: every " << palmInterval << " processed frames (or on tracking loss).\n";  // Report the periodic refresh interval and the tracking-loss fallback.
    double fps = 0.0;  // Initialize the FPS display before processing frames.
    auto fpsStart = GameClock::now(), profileStart = fpsStart;  // Start the HUD and diagnostic timing windows.
    int fpsFrames = 0, profileFrames = 0, palmCalls = 0, landmarkCalls = 0;  // Initialize frame and model-call counts for those windows.
    double waitTotal = 0, decodeTotal = 0, palmTotal = 0, landmarkTotal = 0, ageTotal = 0;  // Initialize the accumulated durations that feed the diagnostic report.
    double overlayTotal = 0, displayTotal = 0;  // Track time spent composing overlays separately from display work.
    std::uint64_t lastCaptured = 0, lastDropped = 0, lastPainted = 0;  // Record the previous snapshot's cumulative capture and paint counters.
    LatestFrames::Metrics capture;  // Hold capture timing and drop statistics for the frame being consumed.
    cv::Mat frame;  // Reuse the frame object so the capture worker can recycle its storage.

    while (true) {  // Process frames until the camera ends or the user quits.
        const auto loopStart = GameClock::now();  // Mark the beginning of the iteration before waiting for a frame.
        if (!frames.next(frame, capture)) {  // Request the latest decoded camera frame and associated metrics.
            std::cerr << "Camera ended or delivered no frame for 3 seconds.\n";  // Report end of stream or the camera wait timeout.
            break;  // Exit the frame loop and proceed to cleanup.
        }  // Finish handling an unavailable frame.
        const auto inferenceStart = GameClock::now();  // Mark when frame acquisition ends and inference starts.
        Hand hand = pipeline.process(frame);  // Detect or track a hand and validate its landmark geometry.
        const auto uiStart = GameClock::now();  // Mark when inference ends and overlay composition starts.
        if (pipeline.hasDisplayPalm) {  // Check whether tracking accepted a box for this frame.
            cv::rectangle(frame, pipeline.displayPalm.box, cv::Scalar(255, 180, 0), 2);  // Draw the accepted palm box in cyan.
        }  // End palm box drawing.

        drawHand(frame, hand);  // Draw the accepted hand skeleton over the camera feed.

        // ----------------------------------------------------
        // GESTURE STABILIZATION
        // ----------------------------------------------------

        std::string rawGesture =  // Prepare the immediate label from this frame.
            classifyGesture(hand);  // Classify the validated landmarks as a game gesture.

        std::string stableGesture =  // Prepare the stable label from the recent history.
            stabilizer.update(  // Update the gesture votes with this frame.
                rawGesture,  // Pass the immediate classification to the stabilizer.
                hand.valid  // Tell the stabilizer whether an accepted hand was detected.
            );  // Store the stabilized result for gameplay and diagnostics.

        auto now = GameClock::now();  // Sample a single steady-clock time for the game and preview update.

        // ----------------------------------------------------
        // GAME STATE
        // ----------------------------------------------------

        updateGame(  // Advance the game state using the accepted recognition result.
            game,  // Pass the round state and scores by reference.
            stabilizer,  // Pass the stabilizer so countdown labels can be cleared.
            rawGesture,  // Pass the classification produced by the latest frame.
            stableGesture,  // Pass the label stabilized over recent frames.
            hand.valid,  // Tell the game whether the present frame has a valid hand.
            now  // Use the same camera-iteration timestamp for its deadlines.
        );  // Finish the per-frame game state update.
        if (game.state == GameState::READY) {  // Look for the starting shake only between rounds.
            const bool wasArmed = shake.armed();  // Remember whether a sequence was armed before the update.
            if (shake.update(elapsedSeconds(GameClock::time_point{}, now), fistShakeSample(hand, rawGesture))) {  // Feed current fist position and scale to the detector on the steady clock.
                startRound(game, stabilizer);  // Start a round when the third downward stroke is accepted.
                shake.reset();  // Clear the motion state for the next round.
            } else if (profile && wasArmed && !shake.armed() && shake.resetReason()) {  // Report a reset only when an armed sequence fails during profiling.
                std::cout << "[shake] " << shake.resetReason() << '\n';  // Print the detector's specific reset explanation to the terminal.
            }  // Finish handling the shake detector's result.
        }  // End activation handling for the ready phase.

        // ----------------------------------------------------
        // FPS
        // ----------------------------------------------------

        ++fpsFrames;  // Count this processed frame in the HUD's sampling window.
        const double fpsElapsed = std::chrono::duration<double>(GameClock::now() - fpsStart).count();  // Measure the current FPS window in fractional seconds.
        if (fpsElapsed >= 1.0) {  // Recalculate the displayed rate once at least a second elapsed.
            fps = fpsFrames / fpsElapsed;  // Divide processed frames by wall-clock time for the measured rate.
            fpsFrames = 0;  // Reset the count for the next FPS window.
            fpsStart = GameClock::now();  // Start counting the next window from the current time.
        }  // Finish the periodic FPS calculation.

        // ----------------------------------------------------
        // GAME INTERFACE
        // ----------------------------------------------------

        drawGameUI(  // Draw the score strip and any active-round footer.
            frame,  // Use the frame that already holds the hand overlay.
            game,  // Read score and state from the ongoing game.
            rawGesture,  // Show the current classifier label.
            hand.valid ? stableGesture : "NO HAND",  // Hide a stale stable label when no current hand passed tracking.
            hand.presence,  // Show the model's hand-presence reading.
            fps,  // Show the recent frame rate.
            now  // Use the current game time to choose the countdown label.
        );  // Finish the HUD composition.

        const auto displayStart = GameClock::now();  // Mark the end of overlays and the beginning of display work.
        gameWindow.present(frame);  // Schedule the composed frame for drawing into the Qt window.

        // ----------------------------------------------------
        // KEYBOARD CONTROLS
        // ----------------------------------------------------

        int key = gameWindow.pollKey();  // Process pending Qt events and read one keyboard input.

        if (profile) {  // Collect detailed timings only when profiling is enabled.
            const auto profileNow = GameClock::now();  // Capture the end time for frame processing and display.
            ++profileFrames;  // Add this iteration to the current profile window.
            waitTotal += std::chrono::duration<double, std::milli>(inferenceStart - loopStart).count();  // Measure time waiting for and receiving the frame.
            decodeTotal += capture.decodeMs;  // Add the camera worker's JPEG decode time, which overlaps the main loop.
            ageTotal += capture.ageMs;  // Add the age of the delivered image at the point it was consumed.
            palmTotal += pipeline.palmMs;  // Add palm stage time when it ran in this iteration.
            landmarkTotal += pipeline.landmarkMs;  // Add landmark stage time when it ran in this iteration.
            palmCalls += pipeline.ranPalm;  // Count actual palm model calls separately from processed frames.
            landmarkCalls += pipeline.ranLandmarks;  // Count actual landmark model calls separately from processed frames.
            overlayTotal += std::chrono::duration<double, std::milli>(displayStart - uiStart).count();  // Accumulate gesture, game state, and overlay work after inference.
            displayTotal += std::chrono::duration<double, std::milli>(profileNow - displayStart).count();  // Accumulate frame conversion, Qt painting, and queued event handling.
            const double seconds = std::chrono::duration<double>(profileNow - profileStart).count();  // Measure the elapsed wall time since the prior report.
            if (seconds >= 2.0) {  // Write a report about every two seconds.
                const auto painted = gameWindow.paintedFrames();  // Read the cumulative number of fresh frames painted by Qt.
                std::cout << std::fixed << std::setprecision(1)  // Format numerical metrics with one decimal place.
                          << "[profile] FPS game " << profileFrames / seconds  // Print the number of game iterations per second.
                          << " | camera " << (capture.captured - lastCaptured) / seconds  // Print decoded camera frames delivered per second.
                          << " | painted " << (painted - lastPainted) / seconds  // Print the rate of fresh frames actually painted by Qt.
                          << " | dropped " << (capture.dropped - lastDropped)  // Print decoded frames overwritten before the game consumed them.
                          << " | ms wait " << waitTotal / profileFrames  // Print average frame acquisition wait in milliseconds.
                          << " | JPEG " << decodeTotal / profileFrames  // Print average JPEG decode duration from the capture worker.
                          << " | palm " << (palmCalls ? palmTotal / palmCalls : 0) << " (" << palmCalls << " calls)"  // Print average palm stage time per call and its call count.
                          << " | landmarks " << (landmarkCalls ? landmarkTotal / landmarkCalls : 0) << " (" << landmarkCalls << " calls)"  // Print average landmark stage time per call and its call count.
                          << " | overlay " << overlayTotal / profileFrames  // Print the mean time after inference spent on game state and overlays.
                          << " | display " << displayTotal / profileFrames  // Print average time for display conversion, painting, and event handling.
                          << " | frame age " << ageTotal / profileFrames << std::endl;  // Print mean image age in milliseconds and flush the report.
                profileStart = profileNow;  // Start the next reporting window at this snapshot.
                lastCaptured = capture.captured;  // Save how many frames the capture worker had decoded.
                lastDropped = capture.dropped;  // Save how many decoded frames had been dropped.
                lastPainted = painted;  // Save how many fresh frames Qt had painted.
                profileFrames = palmCalls = landmarkCalls = 0;  // Clear iteration and model-call totals for the new window.
                waitTotal = decodeTotal = palmTotal = landmarkTotal = ageTotal = overlayTotal = displayTotal = 0;  // Clear accumulated stage timings and frame age for the new window.
            }  // End the periodic reporting branch.
        }  // Finish collecting performance diagnostics.

        if (key == 'f' || key == 'F') {  // Check both uppercase and lowercase fullscreen keys.
            gameWindow.toggleFullscreen();  // Switch between fullscreen and a resizable preview.
        }  // Finish fullscreen handling.

        if (key == 27 || key == 'q' || key == 'Q')  // Treat Escape and either case of Q as quit.
            break;  // Exit the main loop in response to the quit key.

        if (key == ' ' && game.state == GameState::READY) {  // Allow Space to start a round from the ready phase.
            startRound(game, stabilizer);  // Begin the same countdown as the motion trigger.
            shake.reset();  // Clear any partially completed automatic start sequence.
        }  // Finish handling a keyboard start.

        if (key == 'r') {  // Recognize the score reset key.
            game = Game{};  // Replace the current game state with a fresh scoreboard.
            stabilizer.reset();  // Forget stabilized labels from the abandoned round.
            shake.reset();  // Forget any partly completed start motion.
        }  // End scoreboard reset handling.

        if (key == '+' || key == '=') {  // Use plus or equals to increase the palm detection threshold.
            palmThreshold = std::min(  // Limit the increased threshold to its maximum.
                0.95f,  // Cap the threshold at 0.95 to keep a usable setting.
                palmThreshold + 0.05f  // Increase the threshold by five hundredths.
            );  // Store the bounded stricter threshold.
        }  // Finish the threshold increase branch.

        if (key == '-' || key == '_') {  // Use minus or underscore to decrease the palm detection threshold.
            palmThreshold = std::max(  // Limit the decreased threshold to its minimum.
                0.10f,  // Keep the threshold at or above 0.10.
                palmThreshold - 0.05f  // Decrease the threshold by five hundredths.
            );  // Store the bounded lower threshold.
        }  // Finish the threshold decrease branch.
    }  // End the frame loop.

    gameWindow.close();  // Close the game window before disposing of the capture worker.
    frames.stop();  // Stop the camera and join the worker before resources go out of scope.

    std::cout << "Game closed.\n";  // Confirm normal shutdown in the terminal.

    return 0;  // Report successful completion to the calling shell.
}  // End the game executable's entry point.
