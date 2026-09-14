#!/bin/bash
# Keep the macOS UVC camera assistant from re-claiming the D435i's USB
# interface, run the given command, and (optionally) restore the service.
#
# Usage (run from the yolo directory):
#   sudo scripts/rs_go.sh .venv/bin/python scripts/rs_probe.py
#   sudo scripts/rs_go.sh .venv/bin/python scripts/rs_live.py --target "SUI GIN"
#
# Session control:
#   sudo scripts/rs_go.sh --keep  .venv/bin/python scripts/rs_live.py ...
#       leave the UVC assistant disabled (for multi-command sessions)
#   sudo scripts/rs_go.sh --restore
#       re-enable the UVC assistant (restore normal USB-camera behavior)
set -u
UVC_LABEL=system/com.apple.cmio.uvcassistantextension

if [ "${1:-}" = "--restore" ]; then
    launchctl enable "$UVC_LABEL" 2>/dev/null
    launchctl kickstart -k "$UVC_LABEL" 2>/dev/null
    echo "UVC assistant restored"
    exit 0
fi
KEEP=0
if [ "${1:-}" = "--keep" ]; then KEEP=1; shift; fi

launchctl disable "$UVC_LABEL" 2>/dev/null
killall UVCAssistant 2>/dev/null
sleep 2
"$@"
rc=$?
if [ $KEEP -eq 0 ]; then
    launchctl enable "$UVC_LABEL" 2>/dev/null
    launchctl kickstart -k "$UVC_LABEL" 2>/dev/null
else
    echo "(UVC assistant left disabled - restore later with: sudo scripts/rs_go.sh --restore)"
fi
exit $rc
