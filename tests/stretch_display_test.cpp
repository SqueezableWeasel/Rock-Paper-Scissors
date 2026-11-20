#include "stretch_display.h"  // Imports the fullscreen video widget under test.
#include <cassert>  // Provides assertions for display states and pixel comparisons.
#include <iostream>  // Provides a stream for the test summaries.

int main(int argc, char** argv) {  // Run the widget and input regression as a small Qt application.
    QApplication app(argc, argv);  // Initialize Qt's event handling and widget support.
    app.setQuitOnLastWindowClosed(false);  // Leave the event system running while the test closes the widget.
    StretchDisplay window("Display regression test");  // Give the test its own preview window.
    window.showFullScreen();  // Request fullscreen mode for the initial check.
    window.pollKey();  // Process the fullscreen request through the Qt event queue.
    assert(window.isFullScreen());  // Confirm the window entered fullscreen.
    window.toggleFullscreen();  // Request a return to windowed mode.
    window.pollKey();  // Process the pending mode change.
    assert(!window.isFullScreen());  // Confirm the window can leave fullscreen.

    // Deliberately non-contiguous BGR data: exercise stride and channel order.
    cv::Mat storage(482, 650, CV_8UC3, cv::Scalar(0, 0, 0));  // Allocate a BGR image wider than the submitted view.
    cv::Mat frame = storage(cv::Rect(3, 1, 640, 480));  // Select an offset 640-by-480 view with the parent's row stride.
    frame(cv::Rect(0, 0, 320, 240)).setTo(cv::Scalar(0, 0, 255));  // Fill the upper-left region with red pixels.
    frame(cv::Rect(320, 0, 320, 240)).setTo(cv::Scalar(0, 255, 0));  // Fill the upper-right region with green pixels.
    frame(cv::Rect(0, 240, 320, 240)).setTo(cv::Scalar(255, 0, 0));  // Fill the lower-left region with blue pixels.
    frame(cv::Rect(320, 240, 320, 240)).setTo(cv::Scalar(0, 255, 255));  // Fill the lower-right region with yellow pixels.
    window.present(frame);  // Submit the non-contiguous frame to the widget.
    frame.setTo(cv::Scalar(0, 0, 0)); // Renderer must own its submitted frame.

    for (const QSize size : {QSize(1632, 1024), QSize(800, 480),  // Check the usual desktop sizes first.
                             QSize(1920, 1080), QSize(320, 240)}) {  // Also check a large fullscreen target and a small window.
        window.resize(size);  // Resize the widget to the current paint target.
        window.pollKey();  // Let Qt finish applying the resize.
        QImage rendered(size, QImage::Format_RGB32);  // Allocate a 32-bit capture for the rendered widget.
        rendered.fill(Qt::gray);  // Prefill with gray so gaps in coverage are visible.
        window.render(&rendered);  // Paint the widget into the capture for verification.
        // Check every pixel, including all edges: no fixed-size centered image,
        // surrounding background, letterboxing or clipping of the quadrants.
        for (int y = 0; y < size.height(); ++y) {  // Read all output rows, including border pixels.
            for (int x = 0; x < size.width(); ++x) {  // Read all columns at each output row.
                const QRgb expected = y < size.height()/2  // Choose the source color based on the vertical half.
                    ? (x < size.width()/2 ? qRgb(255, 0, 0) : qRgb(0, 255, 0))  // Expect red or green in the upper half.
                    : (x < size.width()/2 ? qRgb(0, 0, 255) : qRgb(255, 255, 0));  // Expect blue or yellow in the lower half.
                assert(rendered.pixel(x, y) == expected);  // Verify the current output pixel has the matching color.
            }  // Finish checking this row from edge to edge.
        }  // Finish checking the entire capture.
        std::cout << "PASS full coverage " << size.width() << "x" << size.height() << '\n';  // Report full frame coverage at the current dimensions.
    }  // Finish testing the supported render targets.

    for (const char character : {'c', 'V', ' ', 'f', 'r', '+', '-', 'q'}) {  // Simulate the character keys used by the application.
        QKeyEvent event(QEvent::KeyPress, static_cast<int>(character),  // Create a key press carrying the raw key code.
                        Qt::NoModifier, QString(QChar(character)));  // Include its printable text and no modifier keys.
        QApplication::sendEvent(&window, &event);  // Deliver the event directly to the widget.
        assert(window.pollKey() == character);  // Verify the widget exposes the same character in its key queue.
    }  // Finish the printable key checks.
    QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);  // Create an Escape key event without printable text.
    QApplication::sendEvent(&window, &escape);  // Deliver Escape to exercise the special key mapping.
    assert(window.pollKey() == 27);  // Confirm Escape is translated to the game's quit code.
    assert(window.pollKey() == -1);  // Verify the queue reports no key after Escape is consumed.
    window.toggleFullscreen();  // Request fullscreen again after the keyboard checks.
    window.pollKey();  // Process the pending fullscreen change.
    assert(window.isFullScreen());  // Verify the second toggle returns to fullscreen.
    window.close();  // Close the preview window through Qt.
    assert(window.pollKey() == 27);  // Verify a window close is also reported as a quit key.
    std::cout << "PASS frame ownership, BGR stride, controls, fullscreen toggle and close\n";  // Report the widget's render, input, and mode checks passed.
}  // Finish the test application.
