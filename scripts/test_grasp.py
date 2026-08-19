"""Unit test for the Live grasp-readiness logic — no models, no camera.

Feeds a LiveRecognizer a FAKE pipeline (fixed boxes / fixed names) and
checks:
  1. track ids are assigned and stable across frames
  2. CLIP (match_boxes) only runs for NEW boxes
  3. a static target becomes grasp-ready after GRASP_WINDOW frames
  4. a moving target (frame-over-frame IoU < GRASP_IOU) is NOT ready,
     and becomes ready again after re-stabilising
  5. a frame with a new object (n_new > 0) resets ALL windows
  6. unknown names report "no matched target"
  7. a stale session (no recent frames) is not graspable
  8. reset() clears the grasp state

Run:  python scripts/test_grasp.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from shelf_demo import live as L
from shelf_demo.detector import Box
from shelf_demo.pipeline import Recognition

A = (100, 100, 200, 200)      # "Cola"   - 100x100 box
B = (400, 100, 500, 200)      # "GABA"   - 100x100 box
A_Moved = (110, 100, 210, 200)  # Cola shifted 10px: IoU with A = 0.818 < 0.9
C = (600, 300, 700, 400)      # "Sprite" - new object for the busy-frame test


class _FakeDetector:
    def __init__(self):
        self.boxes: list[Box] = []

    def detect(self, image, conf=None, imgsz=None):
        return list(self.boxes)


class _FakePipeline:
    def __init__(self):
        self.detector = _FakeDetector()
        self.clip_calls = 0
        self._names: dict[tuple, str] = {}

    def set_scene(self, scene: list[tuple[tuple, str]]) -> None:
        """scene: [(xyxy, name)] — name '' means Unknown."""
        self.detector.boxes = [Box(*xy, 0.9) for xy, _ in scene]
        self._names = {xy: name for xy, name in scene}

    def match_boxes(self, image, boxes, threshold=None):
        recs = []
        for b in boxes:
            self.clip_calls += 1
            name = self._names.get(b.xyxy, "")
            recs.append(Recognition(b, 1 if name else None, name,
                                    0.9 if name else 0.2, bool(name)))
        return recs


def main() -> int:
    pipe = _FakePipeline()
    rec = L.LiveRecognizer(pipe)
    img = Image.new("RGB", (640, 360))
    ok = True

    def check(label: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("" if cond else f"  [{extra}]"))

    K = L.GRASP_WINDOW

    # ---- frames 1..K+1: static scene -------------------------------------
    pipe.set_scene([(A, "Cola"), (B, "GABA")])
    r1 = rec.process(img)
    check("frame 1: both boxes detected", len(r1.recognitions) == 2)
    check("frame 1: both are new (CLIP ran on both)", r1.n_new == 2)
    check("frame 1: CLIP called exactly 2x", pipe.clip_calls == 2)
    ids1 = {r.name: r.track_id for r in r1.recognitions}
    check("frame 1: track ids assigned (unique, >= 0)",
          len(set(ids1.values())) == 2 and all(i >= 0 for i in ids1.values()),
          str(ids1))

    r2 = rec.process(img)
    check("frame 2: no new boxes -> no extra CLIP", pipe.clip_calls == 2)
    ids2 = {r.name: r.track_id for r in r2.recognitions}
    check("frame 2: track ids stable", ids1 == ids2, f"{ids1} vs {ids2}")

    # windows seed on frame 2 (first quiescent frame) and grow by 1/frame
    for f in range(3, K + 2):
        rec.process(img)
    st = rec.grasp_status("Cola")
    check(f"after frame {K+1}: Cola ready (window full)",
          st["ready"] is True, str(st))
    check("Cola box == mean of stable window", st["box"] == list(A),
          str(st.get("box")))
    check("Cola point == box centre", st["point"] == [150.0, 150.0],
          str(st.get("point")))
    check("Cola frames_stable == K", st["frames_stable"] == K)
    check("GABA ready too", rec.grasp_status("GABA")["ready"] is True)

    # ---- unknown / bad names ---------------------------------------------
    st = rec.grasp_status("Juice")
    check("unknown name: not ready", st["ready"] is False)
    check("unknown name: reason mentions no target",
          "no matched target" in st.get("reason", ""), str(st))
    check("available_names lists live matches",
          st["available_names"] == ["Cola", "GABA"], str(st.get("available_names")))

    # ---- moving target: window resets, then re-stabilises ------------------
    pipe.set_scene([(A_Moved, "Cola"), (B, "GABA")])
    rec.process(img)
    st = rec.grasp_status("Cola")
    check("moved 10px (IoU 0.818 < 0.9): Cola not ready, window=1",
          st["ready"] is False and st["frames_stable"] == 1, str(st))
    pipe.set_scene([(A, "Cola"), (B, "GABA")])   # snap back, hold still
    for _ in range(K):
        rec.process(img)
    check("after re-stabilising K frames: Cola ready again",
          rec.grasp_status("Cola")["ready"] is True)

    # ---- busy frame (new object) resets ALL windows ------------------------
    pipe.set_scene([(A, "Cola"), (B, "GABA"), (C, "Sprite")])
    r_busy = rec.process(img)
    check("busy frame: n_new == 1", r_busy.n_new == 1)
    st = rec.grasp_status("Cola")
    check("busy frame reset Cola window", st["ready"] is False
          and st["frames_stable"] == 0, str(st))
    check("busy frame reset GABA window",
          rec.grasp_status("GABA")["frames_stable"] == 0)
    # hold the 3-object scene still -> all three become ready again
    for _ in range(K):
        rec.process(img)
    check("quiescent 3-object scene: Sprite ready too",
          rec.grasp_status("Sprite")["ready"] is True)

    # ---- target leaves the scene -------------------------------------------
    pipe.set_scene([(B, "GABA")])
    rec.process(img)
    st = rec.grasp_status("Cola")
    check("target left: window cleared",
          st["ready"] is False and st["frames_stable"] == 0, str(st))

    # ---- stale session ------------------------------------------------------
    rec._last_frame_ts = time.monotonic() - 5.0   # simulate 5 s of no frames
    st = rec.grasp_status("GABA")
    check("stale session: not graspable",
          st["ready"] is False and "no frames processed recently"
          in st.get("reason", ""), str(st))

    # ---- reset() ------------------------------------------------------------
    rec._last_frame_ts = time.monotonic()
    rec.reset()
    st = rec.grasp_status("GABA")
    check("after reset(): windows cleared",
          st["ready"] is False and st["frames_stable"] == 0, str(st))

    print("GRASP TEST RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
