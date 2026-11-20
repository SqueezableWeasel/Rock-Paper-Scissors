#!/usr/bin/env bash
set -euo pipefail  # Stop if setup fails or an unset variable is used.
script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"  # Resolve the repository directory from this script.
cd "$script_dir"  # Use the repository root so the default model directory resolves correctly.
command -v gdb >/dev/null  # Check that the debugger is available.
if [ "$#" -eq 0 ]; then  # Require an explicit mode before opening the debugger.
    echo "Pass --profile or --self-test" >&2  # Show the usual diagnosis modes.
    exit 2  # Report missing arguments to the calling shell.
fi  # Finish validating the invocation.
set -- --args "$script_dir/rock_paper_scissors" "$@"  # Forward the selected mode and its arguments to the game.
set -- -ex "info sharedlibrary" "$@"  # Report which OpenCV, Qt, and Coral libraries were loaded.
set -- -ex "thread apply all bt" "$@"  # Print a backtrace for each available thread on failure.
set -- -ex run "$@"  # Start the game before requesting backtraces and library information.
set -- -ex "set debuginfod enabled off" "$@"  # Keep symbol lookup local during diagnostics.
set -- -ex "set pagination off" "$@"  # Print the complete report without pausing.
exec gdb -q -batch "$@"  # Run the debugger without a banner or interactive prompts.
