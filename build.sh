#!/usr/bin/env bash
set -euo pipefail  # Stop on failed commands, unset variables, and failed pipeline stages.
script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"  # Resolve source paths from the script location.
coral_library="${EDGETPU_LIBRARY:-/usr/local/lib/libedgetpu.so.1}"  # Allow the installed Coral library path to be overridden.
if [ ! -r "$coral_library" ]; then  # Check that the linker can read the Coral runtime.
    echo "Cannot find $coral_library. Set EDGETPU_LIBRARY to your installed libedgetpu.so.1." >&2  # Explain how to supply the library.
    exit 1  # Stop before building programs that require it.
fi  # End the Coral library check.
command -v pkg-config >/dev/null  # Require pkg-config for compiler and linker flags.
pkg-config --exists opencv4  # Confirm that the OpenCV development package is installed.
if ! pkg-config --atleast-version=5.14 Qt5Widgets; then  # Check for the Qt version used by the fullscreen renderer.
    echo "Install the display headers: sudo apt install qtbase5-dev" >&2  # Identify the package that supplies Qt headers.
    exit 1  # Stop before compiling without Qt.
fi  # End the Qt development package check.

make -C "$script_dir/vendor/rplidar_sdk/sdk" -j2  # Compile the bundled Slamtec SDK in its own directory.
lidar_library="$script_dir/vendor/rplidar_sdk/output/Linux/Release/libsl_lidar_sdk.a"  # Locate the static library just built.
set -- -std=c++17 -O2 -g  # Start the probe compiler arguments with C++17, optimization, and debugger symbols.
set -- "$@" -I"$script_dir/include"  # Find the project's probe and reader headers.
set -- "$@" -I"$script_dir/vendor/rplidar_sdk/sdk/include"  # Find the upstream Slamtec API.
set -- "$@" "$script_dir/c1_probe.cpp"  # Compile the independent LiDAR diagnostic program.
set -- "$@" "$lidar_library" -lpthread  # Link the scanner implementation and threading library.
g++ "$@" -o "$script_dir/c1_probe"  # Build the probe and place it at the repository root.

display_flags="$(pkg-config --cflags --libs opencv4 Qt5Widgets)"  # Get the installed packages' include and linker flags.
for program in rock_paper_scissors hand_tracker; do  # Build both camera applications with their shared dependencies.
    echo "Building $program..."  # Identify the executable being compiled.
    set -- -std=c++17 -O3 -g -fPIC -DQT_NO_KEYWORDS  # Optimize the app and comply with Qt's header requirements.
    set -- "$@" -fno-omit-frame-pointer -Wall -Wextra  # Improve backtraces and enable useful compiler diagnostics.
    set -- "$@" "$script_dir/$program.cpp"  # Compile the entry point selected by this iteration.
    set -- "$@" -I"$script_dir/include"  # Find local tracking and Coral wrappers.
    set -- "$@" $display_flags  # Split the pkg-config output into separate OpenCV and Qt flags.
    set -- "$@" -ltensorflow-lite "$coral_library" -lpthread  # Link inference, Edge TPU support, and threading.
    g++ "$@" -o "$script_dir/$program"  # Compile and link the selected executable at the repository root.
done  # Finish building the game and standalone tracker.
echo "Build complete. From the repository directory, run:"  # Print the next commands after all builds succeed.
echo "  ./rock_paper_scissors --self-test"  # Show how to verify the two models on the TPU.
echo "  ./rock_paper_scissors --profile"  # Show how to start live gameplay with frame timing.
