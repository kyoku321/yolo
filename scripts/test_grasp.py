"""Unit test for the Live grasp-readiness + temporal SKU fusion logic.

Feeds a LiveRecognizer a FAKE pipeline (fixed boxes / fixed names) and
checks the tracker/fusion behaviour:

  1. new tracks are UNCONFIRMED until CONFIRM_FRAMES votes agree
  2. CLIP runs for new/unconfirmed tracks (and only those, per frame)
  3. confirmed tracks hold their label and are NOT re-embedded every frame
  4. a static confirmed target becomes grasp-ready GRASP_WINDOW frames
     after its confirmation
  5. a small move is absorbed by tracking smoothing; a large jump
     restarts the window
  6. an unconfirmed newcomer does NOT reset any window; its confirmation
     (scene really changed a moment later) resets ALL windows
  7. unknown names report "no matched target"
  8. a stale session (no recent frames) is not graspable
  9. reset() clears the grasp state

Run:  python scripts/test_grasp.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from shelf_demo import config, live as L
from shelf_demo.detector import Box
from shelf_demo.pipeline import Recognition

A = (100, 100, 200, 200)          # "Cola"   - 100x100 box
B = (400, 100, 500, 200)          # "GABA"   - 100x100 box
A_SMALL = (105, 100, 205, 200)    # Cola shifted 5px: absorbed by smoothing
A_FAR = (150, 100, 250, 200)      # Cola jumped 50px: restarts the window
C = (600, 300, 700, 400)          # "Sprite" - new object for the busy-frame test


def _iou_xy(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = lambda t: (t[2] - t[0]) * (t[3] - t[1])
    return inter / (area(a) + area(b) - inter)


def xy_near(a: tuple, b: tuple) -> bool:
    return _iou_xy(a, b) > 0.5


class _FakeDetector:
    def __init__(self):
        self.boxes: list[Box] = []

    def detect(self, image, conf=None, imgsz=None):
        return list(self.boxes)


class _FakePipeline:
    """YOLO boxes from the scene; CLIP = name of the nearest scene box.

    The tracker returns Kalman-smoothed boxes, so the fake matches labels
    by box overlap (IoU >= 0.5) instead of exact coordinates.
    """

    def __init__(self):
        self.detector = _FakeDetector()
        self.clip_calls = 0
        self._scene: list[tuple[tuple, str]] = []

    def set_scene(self, scene: list[tuple[tuple, str]]) -> None:
        """scene: [(xyxy, name)] — name '' means Unknown."""
        self._scene = scene
        self.detector.boxes = [Box(*xy, 0.9) for xy, _ in scene]

    def match_boxes(self, image, boxes, threshold=None):
        recs = []
        for b in boxes:
            self.clip_calls += 1
            name, best = "", 0.5
            for xy, nm in self._scene:
                v = _iou_xy(b.xyxy, xy)
                if v > best:
                    best, name = v, nm
            recs.append(Recognition(b, 1 if name else None, name,
                                    0.9 if name else 0.2, bool(name)))
        return recs


def main() -> int:
    pipe = _FakePipeline()
    rec = L.LiveRecognizer(pipe)
    img = Image.new("RGB", (900, 500))   # big enough for all test boxes
    ok = True

    def check(label: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("" if cond else f"  [{extra}]"))

    K = L.GRASP_WINDOW
    CF = config.CONFIRM_FRAMES     # votes a new track needs to confirm

    # ---- confirmation phase (frames 1..CF) ----------------------------------
    pipe.set_scene([(A, "Cola"), (B, "GABA")])
    r1 = rec.process(img)
    check("frame 1: both boxes tracked", len(r1.recognitions) == 2)
    check("frame 1: CLIP ran once per new box", r1.n_new == 2
          and pipe.clip_calls == 2)
    check("frame 1: tracks UNCONFIRMED (scanning, no match shown)",
          not any(r.is_match for r in r1.recognitions))
    ids1 = {r.track_id for r in r1.recognitions}
    check("frame 1: track ids assigned (unique)", len(ids1) == 2, str(ids1))

    r2 = rec.process(img)
    check("frame 2: unconfirmed tracks re-verified (CLIP again)",
          r2.n_new == 2 and pipe.clip_calls == 4)
    check("frame 2: still unconfirmed (< CONFIRM_FRAMES votes)",
          not any(r.is_match for r in r2.recognitions))
    ids2 = {r.track_id for r in r2.recognitions}
    check("frame 2: track ids stable", ids1 == ids2,
          f"{ids1} vs {ids2}")

    r3 = r2
    for _ in range(CF - 2):                 # frames 3..CF
        r3 = rec.process(img)
    check(f"frame {CF}: tracks CONFIRMED via majority vote",
          all(r.is_match for r in r3.recognitions))
    names3 = sorted(r.name for r in r3.recognitions)
    check(f"frame {CF}: labels are Cola/GABA",
          names3 == ["Cola", "GABA"], str(names3))
    clip_after_confirm = pipe.clip_calls

    # ---- quiescent phase: confirm -> grasp-ready after K frames -------------
    for _ in range(K):
        rec.process(img)
    st = rec.grasp_status("Cola")
    check("after confirm + K stable frames: Cola ready",
          st["ready"] is True, str(st))
    check("Cola box == stable window mean", st["box"] == list(A),
          str(st.get("box")))
    check("Cola point == box centre", st["point"] == [150.0, 150.0],
          str(st.get("point")))
    check("Cola frames_stable == K", st["frames_stable"] == K)
    check("GABA ready too", rec.grasp_status("GABA")["ready"] is True)
    check("confirmed tracks NOT re-embedded before REEMBED_INTERVAL",
          pipe.clip_calls == clip_after_confirm, str(pipe.clip_calls))

    # ---- simdepth 3D smoke (web tab) -------------------------------------------
    sd = rec.simdepth_status("Cola", 0.4, 69.0)
    check("simdepth: 3D point from the de-jittered window",
          sd["point_cam_m"] is not None and sd["stable"] is True
          and sd["point_cam_m"][2] == 0.4, str(sd))
    sd_all = rec.simdepth_all_status(0.4, 69.0)
    check("simdepth: per-object 3D points", len(sd_all["points"]) == 2)

    # ---- unknown / bad names ----------------------------------------------------
    st = rec.grasp_status("Juice")
    check("unknown name: not ready", st["ready"] is False)
    check("unknown name: reason mentions no target",
          "no matched target" in st.get("reason", ""), str(st))
    check("available_names lists live confirmed matches",
          st["available_names"] == ["Cola", "GABA"],
          str(st.get("available_names")))

    # ---- small move: absorbed by tracking smoothing -----------------------------
    pipe.set_scene([(A_SMALL, "Cola"), (B, "GABA")])
    rec.process(img)
    st = rec.grasp_status("Cola")
    check("5px move: window NOT restarted (smoothing absorbs it)",
          st["frames_stable"] == K and st["ready"], str(st))

    # ---- large jump: restarts the window ------------------------------------------
    pipe.set_scene([(A_FAR, "Cola"), (B, "GABA")])
    rec.process(img)
    st = rec.grasp_status("Cola")
    check("50px jump: window restarted", st["ready"] is False
          and st["frames_stable"] <= 1, str(st))

    # ---- snap back and re-stabilise ------------------------------------------------
    pipe.set_scene([(A, "Cola"), (B, "GABA")])
    for _ in range(2 * K):
        rec.process(img)
    check("after snap-back + re-stabilising: Cola ready again",
          rec.grasp_status("Cola")["ready"] is True)

    # ---- busy frame (newcomer): tracker gates it one frame (ByteTrack
    # 'unconfirmed' state), then it appears as an unconfirmed track; no
    # grasp window resets until it is confirmed -------------------------------
    pipe.set_scene([(A, "Cola"), (B, "GABA"), (C, "Sprite")])
    r_busy = rec.process(img)
    gated = rec._tracker is not None   # legacy matcher emits instantly
    check("busy frame 1: newcomer gated consistently with tracker",
          r_busy.n_new == (0 if gated else 1), str(r_busy.n_new))
    r_busy = rec.process(img)
    check("busy frame 2: newcomer appears, CLIP embedded once",
          r_busy.n_new == 1, str(r_busy.n_new))
    st = rec.grasp_status("Cola")
    check("unconfirmed newcomer does NOT reset windows",
          st["ready"] is True and st["frames_stable"] == K, str(st))
    sprite_unconfirmed = not any(
        r.is_match for r in r_busy.recognitions
        if xy_near(r.box.xyxy, C))
    check("Sprite shows as unconfirmed (scanning)", sprite_unconfirmed)

    for _ in range(CF - 1):     # Sprite collects its remaining votes
        rec.process(img)
    st = rec.grasp_status("Cola")
    check("newcomer CONFIRMED -> all windows reset",
          st["ready"] is False and st["frames_stable"] < K, str(st))

    for _ in range(K):
        rec.process(img)
    check("quiescent 3-object scene: Sprite ready too",
          rec.grasp_status("Sprite")["ready"] is True)

    # ---- target leaves the scene ---------------------------------------------------
    pipe.set_scene([(B, "GABA")])
    rec.process(img)
    st = rec.grasp_status("Cola")
    check("target left: window cleared",
          st["ready"] is False and st["frames_stable"] == 0, str(st))

    # ---- stale session ---------------------------------------------------------------
    rec._last_frame_ts = time.monotonic() - 5.0   # simulate 5 s of no frames
    st = rec.grasp_status("GABA")
    check("stale session: not graspable",
          st["ready"] is False and "no frames processed recently"
          in st.get("reason", ""), str(st))

    # ---- reset() ------------------------------------------------------------------------
    rec._last_frame_ts = time.monotonic()
    rec.reset()
    st = rec.grasp_status("GABA")
    reset_ok = st["ready"] is False and st["frames_stable"] == 0
    check("after reset(): windows cleared", reset_ok)

    print("GRASP TEST RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
