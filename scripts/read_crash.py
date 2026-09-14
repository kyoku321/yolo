"""Print the faulting-thread backtrace from the newest RealSense crash report.

macOS writes .ips crash reports for root processes to
/Library/Logs/DiagnosticReports/ (mode 600, so run with sudo):

    sudo .venv/bin/python scripts/read_crash.py [pattern]

pattern defaults to "rs-" (rs-capture / rs-enumerate-devices / ...).
Use "python" to read the pyrealsense2 segfault reports.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    pat = sys.argv[1] if len(sys.argv) > 1 else "rs"
    files = sorted(
        glob.glob(f"/Library/Logs/DiagnosticReports/{pat}*.ips")
        + glob.glob(f"/Users/*/Library/Logs/DiagnosticReports/{pat}*.ips"),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not files:
        print(f"no crash reports matching {pat!r} found")
        return 1
    for f in files[:2]:
        print(f"===== {f} =====")
        raw = Path(f).read_text()
        try:
            data = json.loads(raw.split("\n", 1)[1])
        except (ValueError, IndexError):
            print(raw[:2000])
            continue
        print("exception:", data.get("exception"))
        print("termination:", data.get("termination"))
        imgs = data.get("usedImages", [])
        ct = data.get("faultingThread", 0)
        print(f"faulting thread {ct}:")
        for fr in data["threads"][ct].get("frames", [])[:18]:
            idx = fr.get("imageIndex")
            img = imgs[idx] if idx is not None and idx < len(imgs) else {}
            print(f"  {img.get('name', '?'):30s} {fr.get('symbol', '?')} "
                  f"+{fr.get('symbolLocation', '?')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
