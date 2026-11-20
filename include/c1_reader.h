#ifndef RPS_C1_READER_H // Prevent duplicate declarations when the scan reader header is included more than once.
#define RPS_C1_READER_H // Mark the scan reader header as included.
#include "sl_lidar.h" // Import the Slamtec serial channel factory.
#include "sl_lidar_driver.h" // Import the Slamtec LiDAR driver API and scan structures.
#include <algorithm> // Use partial ordering to calculate each angular bin's middle distance.
#include <array> // Keep a fixed number of angle bins and measurement slots.
#include <vector> // Hold a variable number of returns in each angle bin.
#include <thread> // Run scan acquisition away from the game loop.
#include <mutex> // Protect the snapshot shared between threads.
#include <atomic> // Share the shutdown flag safely with the reader thread.
#include <memory> // Manage cleanup of the driver and serial channel.
#include <chrono> // Use a monotonic clock and timed waits.
#include <iostream> // Provide terminal output for connection details.

struct ScanSnapshot { // Store a complete scan reduced to one range per degree.
    std::array<double, 360> meters{}; // Zero means no valid return.
    unsigned long long sequence = 0; // Identify each published scan so readers can distinguish fresh data.
}; // End the scan snapshot record.

inline double sensorNow() { // Provide a monotonic time in seconds for the sensor probe.
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); // Convert the steady clock's time since its own epoch to fractional seconds.
} // End the monotonic clock helper.
class C1Reader { // Own the connection and background acquisition for one Slamtec C1.
    std::unique_ptr<sl::IChannel> channel; // Keep the serial channel alive while the driver uses it.
    std::unique_ptr<sl::ILidarDriver> driver; // Own the driver that controls scanning and returns samples.
    bool connected = false, scanning = false; // Record which device operations need to be undone at shutdown.
    std::atomic<bool> quitting{false}; // Start with acquisition enabled until another thread requests shutdown.
    std::mutex mutex; // Serialize publication and retrieval of scan snapshots.
    ScanSnapshot latest; // Keep the most recently published scan.
    std::thread worker; // Last: all shared state exists before thread starts.
    void read() { // Read, reduce, and publish scans on the worker thread.
        std::array<std::vector<double>, 360> bins; // Accumulate all distances returned at each whole degree.
        std::array<sl_lidar_response_measurement_node_hq_t, 8192> nodes; // Reserve the SDK's buffer for one batch of up to 8192 high quality nodes.
        while (!quitting) { // Repeat until shutdown is requested.
            size_t count = nodes.size(); // Tell the driver how many nodes fit before it writes the actual count.
            if (SL_IS_FAIL(driver->grabScanDataHq(nodes.data(), count, 200))) { // Wait up to 200 milliseconds for a batch and handle a failed read.
                std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Back off briefly on read failure to avoid a busy loop.
                continue; // Retry acquisition without publishing an incomplete scan.
            } // End the read failure branch.
            ScanSnapshot scan; // Begin with an empty scan so unobserved angles remain at zero.
            for (auto& bin : bins) bin.clear(); // Reuse allocation capacity while discarding returns from the previous batch.
            for (size_t n = 0; n < count; ++n) { // Visit each node the driver reported.
                if (!nodes[n].dist_mm_q2 || !nodes[n].quality) continue; // Skip zero distance and zero quality returns.
                const int bin = static_cast<int>(nodes[n].angle_z_q14 * 90.0 / 16384.0) % 360; // Convert the fixed point angle to a whole degree from 0 through 359.
                bins[bin].push_back(nodes[n].dist_mm_q2 / 4000.0); // Convert quarter millimeter units to meters and append the return.
            } // Finish assigning the batch to angle bins.
            for (size_t i = 0; i < bins.size(); ++i) if (!bins[i].empty()) { // Compute a representative distance only for bins with valid returns.
                auto& values = bins[i]; // Work on the vector for this degree in place.
                std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end()); // Select the middle element without fully sorting the vector.
                scan.meters[i] = values[values.size() / 2]; // Publish the middle distance for this degree.
            } // Finish reducing the angle bins.
            { std::lock_guard<std::mutex> lock(mutex); scan.sequence = latest.sequence + 1; latest = scan; } // Assign the next sequence number and publish the snapshot under the lock.
        } // Finish the acquisition loop once shutdown is requested.
    } // End the background reader.
public: // Expose connection control and snapshot retrieval.
    C1Reader() = default; // Wait for an explicit start before opening the serial port.
    C1Reader(const C1Reader&) = delete; // Prevent copies of the active device connection and worker.
    C1Reader& operator=(const C1Reader&) = delete; // Prevent overwriting the connection with a copied reader.
    ~C1Reader() { stop(); } // Stop acquisition and release the device when the reader goes out of scope.
    bool start(const std::string& port) { // Connect to a specified serial port and start acquiring scans.
        auto madeChannel = sl::createSerialPortChannel(port, 460800); // Request a serial channel at the C1's configured baud rate.
        if (!madeChannel) return false; // Report failure if the SDK cannot create the channel.
        channel.reset(*madeChannel); // Adopt the serial channel returned by the SDK.
        auto madeDriver = sl::createLidarDriver(); // Ask the SDK to create its LiDAR driver.
        if (!madeDriver) return false; // Report failure if the SDK cannot create the driver.
        driver.reset(*madeDriver); // Adopt the driver before attempting a connection.
        if (!channel || !driver || SL_IS_FAIL(driver->connect(channel.get()))) return false; // Reject null handles or an unsuccessful connection.
        connected = true; // Remember to disconnect this driver during cleanup.
        sl_lidar_response_device_info_t info; // Allocate storage for the sensor identity response.
        sl_lidar_response_device_health_t health; // Allocate storage for the device health response.
        if (SL_IS_FAIL(driver->getDeviceInfo(info)) || SL_IS_FAIL(driver->getHealth(health)) || // Require both identity and health queries to succeed.
            health.status == SL_LIDAR_STATUS_ERROR) return false; // Refuse to scan if the sensor reports an error state.
        std::cerr << "[C1] Connected on " << port << " at 460800 baud; model " << int(info.model) << std::endl; // Log the selected port, baud rate, and reported model identifier.
        // C1 motor operation follows the scan command; do not call motor APIs
        // on an unconnected channel or assume A1's DTR motor-control behavior.
        if (SL_IS_FAIL(driver->startScan(false, true))) return false; // Request scanning and return failure if it does not start.
        scanning = true; // Record that shutdown must send a stop scan command.
        worker = std::thread(&C1Reader::read, this); // Start the read loop only after all connection state is ready.
        return true; // Confirm that the device is scanning and the worker is running.
    } // End the connection startup routine.
    ScanSnapshot snapshot() { std::lock_guard<std::mutex> lock(mutex); return latest; } // Copy the most recent snapshot under the same lock used by the worker.
    void stop() { // Stop the worker before releasing the driver or channel it may use.
        quitting = true; // Signal the worker to stop waiting for new scans.
        if (worker.joinable()) worker.join(); // Grab timeout bounds the wait.
        if (driver && scanning) { driver->stop(500); scanning = false; } // Stop the sensor scan with a bounded command wait.
        if (driver && connected) { driver->disconnect(); connected = false; } // Disconnect an open device once scanning has stopped.
        driver.reset(); channel.reset(); // Destroy the driver before destroying its serial channel.
    } // End cleanup with both device handles empty.
}; // End the C1 reader class.
#endif // Close the scan reader include guard.
