#include "c1_reader.h"  // Exposes the asynchronous C1 scanner and monotonic clock.
#include <csignal>  // Provides the signal handler API and signal-safe integer type.
static volatile std::sig_atomic_t quit = 0;  // Records a stop request from the interrupt handler.
static void interrupted(int) { quit = 1; }  // Mark the request and let the main thread handle shutdown.
int main(int argc, char** argv) {  // Run a short scan diagnostic from the command line.
    if (argc != 2) { std::cerr << "Usage: c1_probe /dev/ttyUSB0\n"; return 2; }  // Require one serial device path and return a usage error otherwise.
    C1Reader lidar;  // Own the scanner and its worker for the lifetime of the probe.
    if (!lidar.start(argv[1])) { std::cerr << "C1 connection failed\n"; return 1; }  // Open the requested port and stop if scanning cannot start.
    std::signal(SIGINT, interrupted);  // Let Ctrl+C stop the polling loop before normal cleanup.
    unsigned long long previous = 0;  // Track the last published scan sequence to avoid duplicate reports.
    const double started = sensorNow();  // Record the start time for the ten-second sampling window.
    while (!quit && sensorNow() - started < 10) {  // Poll until interrupted or the sampling window ends.
        const auto scan = lidar.snapshot();  // Copy the latest scan under the reader's synchronization.
        if (scan.sequence != previous) {  // Process only a scan published since the previous poll.
            previous = scan.sequence;  // Remember which scan has just been reported.
            int valid = 0; double nearest = 100;  // Count usable bins and seed the minimum with a distance beyond the expected range.
            for (double range : scan.meters) if (range > 0) { ++valid; nearest = std::min(nearest, range); }  // Ignore empty bins when counting and finding the closest return.
            std::cout << "scan " << scan.sequence << " | valid angle bins " << valid;  // Print the sequence and count of angles with a measurement.
            if (valid) std::cout << " | nearest anywhere " << nearest << " m (not necessarily your hand)";  // Report the nearest return only when a usable bin exists.
            std::cout << std::endl;  // Flush the report so it appears immediately in the terminal.
        }  // Finish reporting the newly published scan.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // Let the reader collect more data between polls.
    }  // Finish the timed polling loop.
    return previous ? 0 : 1;  // Return success only if the reader published at least one scan.
}  // Destroy the scanner so its worker and port are closed.
