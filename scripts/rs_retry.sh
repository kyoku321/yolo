#!/bin/bash
# Retry a RealSense test until the camera cooperates.
#
# The D435i on macOS (Apple Silicon) drops into a stuck state after failed
# access attempts - every attempt must start from a fresh USB power cycle:
#   unplug 15 s -> replug into the DIRECT USB-C port (not the dock/hub)
#
# Usage (always with sudo):
#   sudo scripts/rs_retry.sh probe     # rs_probe.py health check (default)
#   sudo scripts/rs_retry.sh capture   # raw rs-capture 5 s stream
#   sudo scripts/rs_retry.sh live      # full rs_live.py 3D output loop
set -u
Y=/Users/kyoku/Documents/IC/yolo
RS=/Users/kyoku/realsense-src/build/Release
PY="$Y/.venv/bin/python"
MODE=${1:-probe}
N=${2:-5}

for i in $(seq 1 "$N"); do
    echo "================ attempt $i / $N ================"
    case "$MODE" in
        capture) "$RS/rs-capture" --duration 5; rc=$? ;;
        live)    "$PY" "$Y/scripts/rs_live.py" --target "SUI GIN"; rc=$? ;;
        *)       "$PY" "$Y/scripts/rs_probe.py"; rc=$? ;;
    esac
    if [ $rc -eq 0 ]; then
        echo "SUCCESS on attempt $i"
        exit 0
    fi
    if [ $rc -eq 130 ]; then   # Ctrl-C by the user in live mode
        echo "stopped by user"
        exit 0
    fi
    if [ "$i" -lt "$N" ]; then
        echo
        echo "FAILED (exit $rc). Power-cycle the camera:"
        echo "  1) UNPLUG the D435i and wait 15 s"
        echo "  2) REPLUG into the DIRECT USB-C port (not the dock/hub)"
        echo "  3) Press Enter to retry ..."
        read -r
    fi
done
echo "gave up after $N attempts"
exit 1
