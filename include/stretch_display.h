#pragma once // Include the display widget declarations only once in each source file.

#include <QApplication> // Use the Qt application to dispatch GUI events in the inference loop.
#include <QCloseEvent> // Handle the window manager's close request.
#include <QEventLoop> // Select which event types to process while polling the display.
#include <QImage> // Keep a paintable copy of the latest camera image.
#include <QKeyEvent> // Read keyboard events from the focused window.
#include <QPainter> // Draw the frame scaled to the widget's available rectangle.
#include <QWidget> // Derive the display window from Qt's QWidget.
#include <opencv2/core.hpp> // Use the OpenCV frame and wrap Qt's pixel buffer.
#include <opencv2/imgproc.hpp> // Convert OpenCV BGR pixels to the Qt paint buffer format.
#include <cstdint> // Store the paint counter in a fixed-width integer.
#include <deque> // Queue the most recent keyboard events in arrival order.

// Paint into the actual top-level client rectangle. OpenCV's HighGUI image
// child can stay 640x480 even when its parent window is fullscreen.
// Use only on the main/UI thread, after creating QApplication.
class StretchDisplay : public QWidget { // Provide a resizeable window that stretches the video to its entire client rectangle.
public: // Expose setup, frame presentation, and keyboard polling to the app.
    explicit StretchDisplay(const char* title, StretchDisplay* keyReceiver = nullptr) // Construct a window with an optional recipient for its key presses.
        : keyReceiver_(keyReceiver) { // Remember the destination for key presses from a secondary window.
        setWindowTitle(QString::fromUtf8(title)); // Set the visible title from the caller's UTF-8 label.
        setAttribute(Qt::WA_OpaquePaintEvent); // Tell Qt that paintEvent fills the widget without needing a background clear.
        setFocusPolicy(Qt::StrongFocus); // Allow the display to take focus and accept keyboard controls.
        resize(640, 480); // Initial WINDOWED size only; fullscreen uses the display.
        setMinimumSize(1, 1); // Let the window shrink while excluding an empty render area.
    } // End display initialization.

    void present(const cv::Mat& frame) { // Copy a camera frame into the widget and schedule a repaint.
        CV_Assert(frame.type() == CV_8UC3 && !frame.empty()); // Require a nonempty eight-bit three-channel BGR image.
        // Convert once at camera resolution and retain ownership for later
        // paints. RGB32 uses Qt's fast opaque scaling into the backing store.
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN // Use direct BGR-to-BGRA conversion when RGB32 stores blue first in memory.
        if (image_.width() != frame.cols || image_.height() != frame.rows) // Reallocate the display buffer only when the incoming frame size changes.
            image_ = QImage(frame.cols, frame.rows, QImage::Format_RGB32); // Allocate a reusable 32-bit opaque Qt image at camera resolution.
        CV_Assert(!image_.isNull()); // Fail early if the destination image could not be allocated.
        cv::Mat pixels(frame.rows, frame.cols, CV_8UC4, image_.bits(), image_.bytesPerLine()); // Wrap Qt's writable pixel buffer in an OpenCV view using its actual row stride.
        cv::cvtColor(frame, pixels, cv::COLOR_BGR2BGRA); // Convert the frame directly into the Qt buffer without a second pixel copy.
#else // Fall back to Qt color conversion on other byte orders.
        const QImage bgr(static_cast<const uchar*>(frame.data), frame.cols, frame.rows, // Wrap the source frame without taking ownership of its pixels.
                         static_cast<int>(frame.step), QImage::Format_BGR888); // Include the OpenCV row stride so sliced images are also handled correctly.
        image_ = bgr.convertToFormat(QImage::Format_RGB32); // Convert to an owned opaque image in the platform's channel order.
#endif // End the byte order dependent image conversion.
        freshFrame_ = true; // Mark the next paint as the first one for this frame.
        update(); // Request a paint event after updating the image.
    } // End frame submission.

    std::uint64_t paintedFrames() const { return paintedFrames_; } // Expose the count of distinct submitted frames that Qt has painted.

    int pollKey() { // Dispatch pending GUI events and return one queued key if available.
        // No fixed wait/sleep: acquisition already paces the 30 FPS loop.
        QApplication::processEvents(QEventLoop::AllEvents, 1); // Briefly process Qt events for video paints, keys, and window updates.
        if (closed_) return 27; // Treat a window close as Escape so existing exit handling runs.
        if (keys_.empty()) return -1; // Return the no-key sentinel when there is no pending input.
        const int key = keys_.front(); // Read the least recent queued key first.
        keys_.pop_front(); // Remove that key from the queue so it is delivered once.
        return key; // Return the queued key to the game.
    } // End GUI event polling.

    void toggleFullscreen() { // Toggle between fullscreen and the previous normal window.
        if (isFullScreen()) showNormal(); // Exit fullscreen if the widget is currently fullscreen.
        else showFullScreen(); // Fill the display if the widget is currently windowed.
    } // End the display mode toggle.

    bool wasClosed() const { return closed_; } // Report whether a close event was received.
    void reopen() { closed_ = false; show(); } // Clear the closed flag and show the window on request.

protected: // Override the QWidget events used to paint and collect input.
    void paintEvent(QPaintEvent*) override { // Redraw the video whenever Qt requests a paint.
        QPainter painter(this); // Begin painting on this widget.
        if (image_.isNull()) painter.fillRect(rect(), Qt::black); // Fill the window with black until the first frame is available.
        else { // Otherwise paint the submitted video image and track its delivery.
            // Stretch both axes, including all overlays. Re-evaluating rect()
            // here follows fullscreen, window resizing and display-mode changes.
            painter.setRenderHint(QPainter::SmoothPixmapTransform, false); // Use fast nearest-neighbor scaling for the live preview.
            painter.drawImage(rect(), image_); // Fit the image independently along both axes to fill the window.
            if (freshFrame_) { // Update the paint count only if this frame has not been drawn before.
                ++paintedFrames_; // Record one more distinct frame presented to the widget.
                freshFrame_ = false; // Mark the frame as drawn so repeated Qt paints are excluded.
            } // End fresh frame accounting.
        } // Finish drawing the current image.
    } // End the paint event handler.

    void keyPressEvent(QKeyEvent* event) override { // Translate Qt keyboard events to the game's integer key codes.
        if (event->isAutoRepeat()) { event->accept(); return; } // Consume auto-repeat events without enqueueing duplicates.
        int key = -1; // Start with an unrecognized-key sentinel.
        if (event->key() == Qt::Key_Escape) key = 27; // Map Escape to the ASCII code the game uses to close.
        else if (!event->text().isEmpty()) { // Read other keys only if Qt supplied printable or control text.
            const ushort value = event->text().front().unicode(); // Get the Unicode value of the first character.
            if (value < 128) key = static_cast<int>(value); // Forward only values representable by the game's ASCII key interface.
        } // End conversion of Qt key text.
        auto& queue = keyReceiver_ ? keyReceiver_->keys_ : keys_; // Use the main window's queue for keys from a secondary preview.
        if (key >= 0 && queue.size() < 16) queue.push_back(key); // Queue recognized input while keeping the buffer bounded.
        event->accept(); // Tell Qt the key press has been handled.
    } // End the key press event handler.

    void closeEvent(QCloseEvent* event) override { // Record a close request from the window manager.
        closed_ = true; // Let pollKey deliver the existing Escape action next time.
        event->accept(); // Allow the window close to proceed.
    } // End the close event handler.

private: // Keep owned image storage and pending input internal to the widget.
    QImage image_; // Store the last submitted frame independently of the camera buffer.
    std::deque<int> keys_; // Keep key presses pending until the game polls them.
    StretchDisplay* keyReceiver_ = nullptr; // Optionally forward key input to a different display.
    bool closed_ = false; // Remember whether this widget has received a close event.
    bool freshFrame_ = false; // Flag an image awaiting its first paint.
    std::uint64_t paintedFrames_ = 0; // Count submitted frames that have been painted.
}; // End the stretch display widget class.
