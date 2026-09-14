"""Live (webcam) recognition: BYTE-track every frame, fuse SKU votes over time.

Frame loop (LiveRecognizer.process):
  1. YOLO detects boxes (every frame, detection-speed; full per-frame CLIP
     is too slow: ~150-300 ms/embed vs ~45 ms/frame for YOLO @960 on M1).
  2. A per-session ultralytics tracker (SHELF_TRACKER, ByteTrack by default)
     associates detections to persistent track ids. Its Kalman filter
     smooths per-frame box jitter, tolerates 1-2 dropped detections and its
     two-stage high/low-conf association avoids ID swaps on dense shelves.
     SHELF_TRACKER=off falls back to the legacy greedy-IoU matcher.
  3. CLIP + retrieval runs only for tracks that are DUE: brand-new tracks
     immediately, unconfirmed ones every frame until confirmed, confirmed
     ones round-robin every REEMBED_INTERVAL frames (so a bad first lookup
     self-corrects instead of sticking for the life of the track).
  4. Each track keeps the last VOTE_WINDOW retrieval outcomes and its SKU is
     decided by MAJORITY VOTE, never by a single frame. A new track is
     UNCONFIRMED (drawn as Unknown/scanning, never offered for grasping)
     until >= CONFIRM_FRAMES votes exist and >= VOTE_MIN of them agree on
     one SKU above the match threshold. Hysteresis: the held SKU only has to
     stay above MATCH_THRESHOLD_KEEP (< threshold), which stops label
     flapping while the score oscillates near the threshold.

Also contains the JSON plumbing for the Live tab: the browser posts webcam
frames to a FastAPI route; per-session LiveRecognizer state is kept
server-side keyed by a client-generated uuid (see handle_frame/handle_reset).

Grasp readiness (robotic arm integration): each product name carries a
stability window over its best track. Only CONFIRMED tracks participate and
windows reset only when a track is newly confirmed — a single spurious
detection no longer freezes the whole scene. See _update_grasp and
handle_grasp (GET /live/api/grasp?uuid=...&name=...).
"""
from __future__ import annotations

import base64
import io
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config
from .detector import Box
from .draw import summary
from .pipeline import Recognition, ShelfPipeline
from .pose3d import deproject

MATCH_IOU = 0.5    # legacy greedy matcher (only used when SHELF_TRACKER=off)
MAX_MISSES = 4     # frames a track may vanish before its vote state is dropped

SESSION_TTL_S = 120.0       # idle live sessions are dropped after this
IMGSZ_RANGE = (480, 1536)   # accepted detection sizes from the Live tab

# Grasp readiness (robotic arm, GET /live/api/grasp). A product name is
# ready when ONE specific confirmed track with that name has been present
# for GRASP_WINDOW consecutive frames, staying put frame-over-frame
# (IoU >= GRASP_IOU). A newly confirmed track (scene really changed)
# resets every window; single-frame spurious detections do not touch it.
# The returned box is the mean of the window (de-jittered).
# GRASP_MAX_AGE_MS: don't offer a grasp on a stale session.
GRASP_WINDOW = 5            # frames of stability required (~300 ms @16 FPS)
GRASP_IOU = 0.9             # min box IoU between consecutive window frames
GRASP_MAX_AGE_MS = 1000.0   # state older than this is not graspable

# Synthetic-depth 3D for the web Live tab (GET /live/api/simdepth). The
# browser stream carries no depth: z is a user-supplied constant plane in
# metres, intrinsics are synthesised from frame size + hfov. A web session is
# considered live for 3D purposes until the frame is this old.
SIMDEPTH_MAX_AGE_MS = 1500.0


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


# ---- box trackers ----------------------------------------------------------

# A vote outcome stored per track: (sku_id or None, score, name). None means
# the crop did not clear the match threshold on that verification.
_Vote = tuple[int | None, float, str]


class _ByteTrackWrapper:
    """Per-session ultralytics tracker (SHELF_TRACKER) over our Box list.

    The ultralytics tracker owns the Kalman filter, the two-stage high/low
    confidence association and the lost-track buffer; this wrapper only
    adapts Box <-> ultralytics Boxes and clamps output to image bounds.
    Per-session (not model.track(persist=True)) on purpose: the Detector is
    shared across live sessions / the RGB-D session, so tracker state must
    not live inside the shared model's predictor.
    """

    def __init__(self, tracker, args, low_yaml: float) -> None:
        self._tracker = tracker
        self._args = args          # namespace of tracker yaml + conf bridge
        self._low_yaml = low_yaml

    def set_conf(self, conf: float) -> float:
        """Re-derive the user-conf band; returns the DETECT conf to use.

        The UI 'Confidence' slider maps to the first-stage/new-track
        threshold; ByteTrack's low band below it keeps weak boxes alive for
        the second-stage association, so YOLO must run at the LOW threshold.
        """
        self._args.track_high_thresh = conf
        self._args.new_track_thresh = conf
        low = min(self._low_yaml, conf)
        self._args.track_low_thresh = low
        return low

    def update(self, image: Image.Image, boxes: list[Box]
               ) -> list[tuple[int, Box]]:
        from ultralytics.engine.results import Boxes as UBoxes  # lazy
        arr = (np.zeros((0, 6), dtype=np.float32) if not boxes
               else np.array(
                   [[b.x1, b.y1, b.x2, b.y2, b.conf, 0.0] for b in boxes],
                   dtype=np.float32))
        det = UBoxes(arr, orig_shape=(image.height, image.width))
        rows = self._tracker.update(det, img=np.asarray(image))
        w, h = image.size
        out: list[tuple[int, Box]] = []
        # row layout: x1, y1, x2, y2, track_id, score, cls, det_idx
        for r in rows:
            x1 = max(0, min(int(round(r[0])), w - 1))
            y1 = max(0, min(int(round(r[1])), h - 1))
            x2 = max(0, min(int(round(r[2])), w))
            y2 = max(0, min(int(round(r[3])), h))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((int(r[4]), Box(x1, y1, x2, y2, float(r[5]))))
        return out


class _GreedyIoUTracker:
    """Legacy matcher, kept as SHELF_TRACKER=off fallback.

    Associates detections to tracks by best IoU >= MATCH_IOU and drops
    tracks missed for more than MAX_MISSES frames. No motion model — this
    is exactly the jitter/ID-swap-prone behaviour ByteTrack replaces.
    """

    def __init__(self) -> None:
        self._rows: list[list] = []      # [id, box, misses]
        self._next_id = 0

    def update(self, boxes: list[Box]) -> list[tuple[int, Box]]:
        assigned: dict[int, list] = {}
        for row in self._rows:
            best_i, best_v = -1, MATCH_IOU
            for i, b in enumerate(boxes):
                if i in assigned:
                    continue
                v = _iou(row[1], b)
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                row[1], row[2] = boxes[best_i], 0
                assigned[best_i] = row
            else:
                row[2] += 1
        self._rows = [r for r in self._rows if r[2] <= MAX_MISSES]
        out: list[tuple[int, Box]] = []
        for i, b in enumerate(boxes):
            row = assigned.get(i)
            if row is None:
                row = [self._next_id, b, 0]
                self._next_id += 1
                self._rows.append(row)
            out.append((row[0], b))
        return out


def _build_tracker(conf: float):
    """Create the per-session tracker; (None, conf) -> legacy greedy-IoU."""
    if config.TRACKER.strip().lower() == "off":
        return None, conf
    try:
        from types import SimpleNamespace
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml
        cfg = dict(YAML.load(check_yaml(config.TRACKER)))
        low_yaml = float(cfg.get("track_low_thresh", 0.1))
        args = SimpleNamespace(**cfg)
        ttype = str(cfg.get("tracker_type", "bytetrack")).lower()
        if ttype == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT  # needs boxmot
            tracker = BOTSORT(args)
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker
            tracker = BYTETracker(args)
        wrapper = _ByteTrackWrapper(tracker, args, low_yaml)
        return wrapper, wrapper.set_conf(conf)
    except Exception as e:  # noqa: BLE001 - never kill the live stream
        print(f"[live] SHELF_TRACKER={config.TRACKER!r} unavailable "
              f"({e!r}); falling back to legacy greedy-IoU matching")
        return None, conf


@dataclass
class _Track:
    id: int                     # tracker-assigned id, exposed to clients
    box: Box
    votes: deque = field(
        default_factory=lambda: deque(maxlen=config.VOTE_WINDOW))  # _Vote
    misses: int = 0             # consecutive frames absent from tracker output
    confirmed: bool = False     # majority vote settled (see _reevaluate)
    sku_id: int | None = None
    name: str = ""
    score: float = 0.0
    last_embed_frame: int = -10**9   # frame idx of last CLIP verification


@dataclass
class _GraspState:
    """Stability window for one product name (see _update_grasp)."""
    track_id: int | None = None
    boxes: list[Box] = field(default_factory=list)

    def reset(self) -> None:
        self.track_id = None
        self.boxes.clear()


@dataclass
class LiveResult:
    recognitions: list[Recognition]
    changed: bool       # matched set changed -> caller should refresh summary
    n_new: int          # boxes that went through CLIP this frame


class LiveRecognizer:
    """Stateful per-session recognizer behind draw_recognitions/summary."""

    def __init__(self, pipeline: ShelfPipeline,
                 conf: float | None = None,
                 imgsz: int | None = None,
                 threshold: float | None = None) -> None:
        self.pipeline = pipeline
        self.conf = config.DETECT_CONF if conf is None else float(conf)
        self.imgsz = config.DETECT_IMGSZ if imgsz is None else int(imgsz)
        self.threshold = (config.MATCH_THRESHOLD if threshold is None
                          else float(threshold))
        self._tracks: dict[int, _Track] = {}   # tracker id -> vote state
        self._frame = 0                      # processed-frame counter
        self._greedy = _GreedyIoUTracker()
        # (ultralytics tracker wrapper | None, detector conf to run at)
        self._tracker, self._detect_conf = _build_tracker(self.conf)
        self._prev_n = 0
        self.fps = 0.0        # EMA of server-side processing speed
        self._grasp: dict[str, _GraspState] = {}   # name -> window
        self._last_frame_ts: float | None = None
        self._last_img_size: tuple[int, int] | None = None
        self._last_recs: list[Recognition] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_n = 0
        self._frame = 0
        # fresh tracker: no stale ids/Kalman states carried into a new run
        self._greedy = _GreedyIoUTracker()
        self._tracker, self._detect_conf = _build_tracker(self.conf)
        for st in self._grasp.values():
            st.reset()

    def process(self, image: Image.Image) -> LiveResult:
        image = image.convert("RGB")
        self._last_frame_ts = time.monotonic()
        self._last_img_size = (image.width, image.height)
        self._frame += 1
        pipe = self.pipeline
        detect_conf = (self.conf if self._tracker is None
                       else self._tracker.set_conf(self.conf))
        boxes = pipe.detector.detect(image, conf=detect_conf, imgsz=self.imgsz)
        if self._tracker is not None:
            # The auto detector's open-vocabulary fallback scores MUCH lower
            # than the trained specialist (0.05-0.4 vs 0.2+). Re-band the
            # tracker so such a box can still START a track: at the
            # specialist's threshold ByteTrack files it as a low-confidence
            # association-only box, never creates a track, and the fallback
            # would show no bounding box after all.
            if getattr(pipe.detector, "last_path", "") == "fallback":
                self._tracker.set_conf(min(self.conf, config.YOLO_WORLD_CONF))
            tracked = self._tracker.update(image, boxes)
        else:
            tracked = self._greedy.update(boxes)

        # 1. sync per-track vote state with the tracker output
        present: list[_Track] = []
        new_tracks: list[_Track] = []
        seen: set[int] = set()
        for tid, box in tracked:
            tr = self._tracks.get(tid)
            if tr is None:
                tr = _Track(id=tid, box=box)
                self._tracks[tid] = tr
                new_tracks.append(tr)
            else:
                tr.box, tr.misses = box, 0
            present.append(tr)
            seen.add(tid)
        expired = False
        for tid in list(self._tracks):
            if tid not in seen:
                tr = self._tracks[tid]
                tr.misses += 1
                if tr.misses > MAX_MISSES:
                    del self._tracks[tid]
                    expired = True

        # 2. CLIP + retrieval, only for tracks that are due: new tracks
        #    now, still-unconfirmed ones every frame, confirmed ones
        #    round-robin every REEMBED_INTERVAL frames (max
        #    REEMBED_MAX_PER_FRAME per frame to keep the stream smooth).
        new_ids = {t.id for t in new_tracks}
        embed_now: list[_Track] = list(new_tracks)
        reverify: list[_Track] = []
        for tr in present:
            if tr.id in new_ids:
                continue
            if not tr.confirmed:
                embed_now.append(tr)
            elif self._frame - tr.last_embed_frame >= config.REEMBED_INTERVAL:
                reverify.append(tr)
        reverify.sort(key=lambda t: t.last_embed_frame)
        embed_now.extend(reverify[:config.REEMBED_MAX_PER_FRAME])
        # Bound CLIP work per frame. Order is priority: new tracks first, then
        # unconfirmed (both every-frame), then the re-verification tail. Tracks
        # that miss the cap keep their vote state and are embedded next frame.
        if len(embed_now) > config.EMBED_MAX_PER_FRAME:
            embed_now = embed_now[:config.EMBED_MAX_PER_FRAME]

        n_clip = len(embed_now)
        transitions: list[str] = []
        if embed_now:
            recs = pipe.match_boxes(image, [t.box for t in embed_now],
                                    self.threshold)
            for tr, rec in zip(embed_now, recs):
                tr.votes.append((rec.sku_id if rec.is_match else None,
                                 rec.score, rec.name))
                tr.last_embed_frame = self._frame
                out = self._reevaluate(tr)
                if out:
                    transitions.append(out)
        n_new_confirmed = transitions.count("confirm")

        # 3. emit current recognitions in tracker order
        recognitions = [self._emit(tr) for tr in present]

        # 4. grasp readiness: update per-name stability windows
        self._update_grasp(present, n_new_confirmed)

        # 5. remember latest recognitions (read by simdepth_status)
        self._last_recs = recognitions

        changed = (bool(new_tracks) or expired or bool(transitions)
                   or len(recognitions) != self._prev_n)
        self._prev_n = len(recognitions)
        return LiveResult(recognitions, changed, n_clip)

    # ---- temporal SKU fusion ----------------------------------------------
    def _reevaluate(self, tr: _Track) -> str:
        """Re-decide a track's SKU from its recent votes.

        Returns a transition tag ("confirm" / "switch" / "drop") or "" --
        callers use it to mark the label set as changed and (for "confirm")
        to reset the grasp stability windows.
        Rules (parameters: config.VOTE_WINDOW / VOTE_MIN / CONFIRM_FRAMES /
        MATCH_THRESHOLD_KEEP):
          unconfirmed: lock SKU S once >= CONFIRM_FRAMES votes exist and
            >= VOTE_MIN of them agree on S with mean score >= threshold;
          confirmed SKU S: keep S while S appears in the window with mean
            score >= MATCH_THRESHOLD_KEEP (hysteresis);
          switch to another SKU only if it independently clears the full
            lock criteria; otherwise the label is dropped -> unconfirmed.
        """
        groups: dict[int, list] = {}   # sku_id -> [scores, names]
        for sku, sc, nm in tr.votes:
            if sku is None:
                continue
            g = groups.setdefault(sku, [[], []])
            g[0].append(sc)
            g[1].append(nm)
        if groups:
            dom_sku, g = max(groups.items(),
                             key=lambda kv: (len(kv[1][0]), sum(kv[1][0])))
            dom_mean = sum(g[0]) / len(g[0])
            dom_name = g[1][-1]
        else:
            dom_sku, dom_mean, dom_name = None, 0.0, ""

        if not tr.confirmed:
            if (len(tr.votes) >= config.CONFIRM_FRAMES and dom_sku is not None
                    and len(groups[dom_sku][0]) >= config.VOTE_MIN
                    and dom_mean >= self.threshold):
                self._set_label(tr, dom_sku, dom_mean, dom_name)
                tr.confirmed = True
                return "confirm"
            if tr.votes:
                tr.score = tr.votes[-1][1]     # best-guess score for display
            return ""

        held = groups.get(tr.sku_id) if tr.sku_id is not None else None
        if held:
            held_mean = sum(held[0]) / len(held[0])
            if held_mean >= config.MATCH_THRESHOLD_KEEP:
                tr.score = held_mean
                tr.name = held[1][-1]
                return ""
        if (dom_sku is not None and dom_sku != tr.sku_id
                and len(groups[dom_sku][0]) >= config.VOTE_MIN
                and dom_mean >= self.threshold):
            self._set_label(tr, dom_sku, dom_mean, dom_name)
            return "switch"
        # the held SKU lost support in the window: back to scanning
        tr.confirmed, tr.sku_id, tr.name = False, None, ""
        if tr.votes:
            tr.score = tr.votes[-1][1]
        return "drop"

    def _set_label(self, tr: _Track, sku_id: int, score: float,
                   vote_name: str) -> None:
        tr.sku_id = sku_id
        tr.score = score
        catalog = getattr(self.pipeline, "catalog", None)
        product = catalog.get(sku_id) if catalog is not None else None
        tr.name = product.name if product is not None else vote_name

    def _emit(self, tr: _Track) -> Recognition:
        if tr.confirmed and tr.sku_id is not None:
            return Recognition(tr.box, tr.sku_id, tr.name, tr.score, True,
                               track_id=tr.id)
        # unconfirmed -> "scanning": shown as Unknown, never graspable
        return Recognition(tr.box, None, "", tr.score, False, track_id=tr.id)

    def _update_grasp(self, present: list[_Track],
                      n_new_confirmed: int) -> None:
        """Maintain per-name stability windows (see GRASP_WINDOW comment).

        Only CONFIRMED tracks participate (flapping young tracks can't
        affect grasping). Locks onto ONE track per name (first in tracker
        order). The window slides as long as that same track stays put; a
        track change or a jump (IoU < GRASP_IOU) restarts it. A NEWLY
        CONFIRMED track means the scene really changed -> every window
        resets; single-frame spurious detections reset nothing.
        """
        current: dict[str, _Track] = {}
        for tr in present:
            if tr.confirmed and tr.name and tr.name not in current:
                current[tr.name] = tr

        if n_new_confirmed > 0:   # scene really changed: nothing graspable
            for st in self._grasp.values():
                st.reset()
            return

        for name in [n for n in self._grasp if n not in current]:
            self._grasp[name].reset()   # target left the scene

        for name, tr in current.items():
            st = self._grasp.setdefault(name, _GraspState())
            if st.track_id != tr.id:
                st.track_id, st.boxes = tr.id, [tr.box]
            elif _iou(st.boxes[-1], tr.box) >= GRASP_IOU:
                if len(st.boxes) >= GRASP_WINDOW:
                    st.boxes.pop(0)
                st.boxes.append(tr.box)
            else:
                st.track_id, st.boxes = tr.id, [tr.box]   # jumped: restart

    def grasp_status(self, name: str) -> dict:
        """Grasp readiness for one product name (GET /live/api/grasp).

        Read under LiveSessionManager.model_lock (state is mutated by
        process() while a frame is in flight).
        """
        age_ms = ((time.monotonic() - self._last_frame_ts) * 1000.0
                  if self._last_frame_ts is not None else float("inf"))
        available = sorted({t.name for t in self._tracks.values()
                            if t.confirmed and t.name})
        out: dict = {
            "name": name,
            "ready": False,
            "track_id": None,
            "box": None,
            "point": None,
            "frames_stable": 0,
            "age_ms": round(age_ms, 1) if age_ms != float("inf") else None,
            "img_w": self._last_img_size[0] if self._last_img_size else None,
            "img_h": self._last_img_size[1] if self._last_img_size else None,
            "available_names": available,
        }
        if self._last_frame_ts is None or age_ms > GRASP_MAX_AGE_MS:
            out["reason"] = ("no frames processed recently - "
                             "start the Live camera first")
            return out
        st = self._grasp.get(name)
        if st is None or st.track_id is None or not st.boxes:
            if name in available:
                out["reason"] = (f"target not stable yet: 0/{GRASP_WINDOW} "
                                 "frames (just appeared or scene changed)")
            else:
                out["reason"] = "no matched target with that name right now"
            return out
        out["track_id"] = st.track_id
        out["frames_stable"] = len(st.boxes)
        if len(st.boxes) < GRASP_WINDOW:
            out["reason"] = (f"target not stable yet: {len(st.boxes)}/"
                             f"{GRASP_WINDOW} frames (moving or just "
                             "appeared)")
            return out
        n = len(st.boxes)
        x1 = int(sum(b.x1 for b in st.boxes) / n)
        y1 = int(sum(b.y1 for b in st.boxes) / n)
        x2 = int(sum(b.x2 for b in st.boxes) / n)
        y2 = int(sum(b.y2 for b in st.boxes) / n)
        out["ready"] = True
        out["box"] = [x1, y1, x2, y2]
        out["point"] = [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]
        return out

    def _simdepth_base(self, depth_m: float, hfov_deg: float):
        """Freshness check + synthetic intrinsics shared by the 3D answers.

        Returns (out, K); K is None (with out["reason"] set) when there is no
        fresh frame to attach 3D to.
        """
        out: dict = {
            "has_frame": False, "frames_stable": 0,
            "stable_window": GRASP_WINDOW,
            "z_m": round(float(depth_m), 4), "hfov": round(hfov_deg, 1),
            "img_w": None, "img_h": None, "available_names": [],
            "reason": "",
        }
        if self._last_frame_ts is None:
            out["reason"] = "no frames processed yet - start the Live camera"
            return out, None
        age_ms = (time.monotonic() - self._last_frame_ts) * 1000.0
        if age_ms > SIMDEPTH_MAX_AGE_MS:
            out["reason"] = (f"last frame {age_ms:.0f} ms old - "
                             "camera stopped?")
            return out, None
        if self._last_img_size is None:
            out["reason"] = "no frame size yet"
            return out, None
        out["has_frame"] = True
        w, h = self._last_img_size
        out["img_w"], out["img_h"] = w, h
        out["available_names"] = sorted(
            {t.name for t in self._tracks.values()
             if t.confirmed and t.name})
        fx = fy = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = np.array([[fx, 0.0, w / 2.0],
                      [0.0, fy, h / 2.0],
                      [0.0, 0.0, 1.0]])
        return out, K

    def simdepth_status(self, name: str, depth_m: float,
                        hfov_deg: float) -> dict:
        """Camera-frame 3D for one product via a SYNTHETIC depth plane.

        The web stream has no depth, so z is the user-supplied constant
        plane (metres) and intrinsics are synthesised from the last frame
        size + hfov. The box is the de-jittered stability window when one
        exists for the name, else the latest matched box. point_ee_m is
        filled when data/handeye.json (T_ee_cam) is present; the arm-base
        frame additionally needs live FK and is therefore not offered here.
        Read under LiveSessionManager.model_lock.
        """
        out, K = self._simdepth_base(depth_m, hfov_deg)
        out["name"] = name
        out["stable"] = False
        out["track_id"] = None
        out["box"] = None
        out["point_cam_m"] = None
        out["point_ee_m"] = None
        if K is None:
            return out

        # box: de-jittered stability window if any, else latest matched rec
        box: Box | None = None
        st = self._grasp.get(name)
        if st is not None and st.track_id is not None and st.boxes:
            n = len(st.boxes)
            box = Box(sum(b.x1 for b in st.boxes) / n,
                      sum(b.y1 for b in st.boxes) / n,
                      sum(b.x2 for b in st.boxes) / n,
                      sum(b.y2 for b in st.boxes) / n, 1.0)
            out["track_id"] = st.track_id
            out["frames_stable"] = n
            out["stable"] = n >= GRASP_WINDOW
        else:
            for r in reversed(self._last_recs):
                if r.is_match and r.name == name:
                    box = r.box
                    out["track_id"] = r.track_id
                    break
        if box is None:
            out["reason"] = ("no matched target with that name right now "
                             "(see the names listed / summary below)")
            return out

        u, v = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
        p_cam = deproject(u, v, float(depth_m), K)
        out["box"] = [round(box.x1, 1), round(box.y1, 1),
                      round(box.x2, 1), round(box.y2, 1)]
        out["point_cam_m"] = [round(float(x), 4) for x in p_cam]
        T_ee_cam = _handeye_T()
        if T_ee_cam is not None:
            from .transforms import transform_points
            out["point_ee_m"] = [round(float(x), 4)
                                 for x in transform_points(T_ee_cam, p_cam)]
        return out

    def simdepth_all_status(self, depth_m: float, hfov_deg: float) -> dict:
        """Camera-frame 3D for EVERY recognition of the last frame.

        Same synthetic-plane geometry as simdepth_status, but no target
        name: `points` lists {id, box, label, match, point_cam_m} for every
        detected object so the browser can annotate all boxes. Read under
        LiveSessionManager.model_lock.
        """
        out, K = self._simdepth_base(depth_m, hfov_deg)
        out["name"] = ""
        out["stable"] = False
        out["track_id"] = None
        out["box"] = None
        out["point_cam_m"] = None
        out["point_ee_m"] = None
        out["points"] = []
        if K is None:
            return out
        for r in self._last_recs:
            u, v = (r.box.x1 + r.box.x2) / 2.0, \
                (r.box.y1 + r.box.y2) / 2.0
            p = deproject(u, v, float(depth_m), K)
            out["points"].append({
                "id": r.track_id,
                "box": [round(r.box.x1, 1), round(r.box.y1, 1),
                        round(r.box.x2, 1), round(r.box.y2, 1)],
                "label": r.label, "match": bool(r.is_match),
                "point_cam_m": [round(float(x), 4) for x in p]})
        return out


# ---- web session plumbing (used by the FastAPI routes of the Live tab) -----

class LiveSessionManager:
    """One LiveRecognizer per browser session id, with idle expiry.

    Two *separate* locks: `_map_lock` guards the session dict itself,
    `model_lock` is held while a LiveRecognizer mutates tracks / runs model
    inference (MPS does not like concurrent calls). Never nesting them avoids
    self-deadlock.
    """

    def __init__(self, ttl: float = SESSION_TTL_S) -> None:
        self.ttl = ttl
        self._map_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self._sessions: dict[str, tuple[LiveRecognizer, float]] = {}

    def _evict(self, now: float) -> None:
        stale = [sid for sid, (_, ts) in self._sessions.items()
                 if now - ts > self.ttl]
        for sid in stale:
            del self._sessions[sid]

    def get(self, sid: str, pipeline: ShelfPipeline) -> LiveRecognizer:
        now = time.monotonic()
        with self._map_lock:
            self._evict(now)
            if sid not in self._sessions:
                self._sessions[sid] = (LiveRecognizer(pipeline), now)
            else:
                rec, _ = self._sessions[sid]
                self._sessions[sid] = (rec, now)
            return self._sessions[sid][0]

    def reset(self, sid: str) -> None:
        # model_lock: don't clear tracks while a frame is mid-process
        with self.model_lock:
            entry = self._sessions.get(sid)
            if entry is not None:
                entry[0].reset()

    @property
    def n_sessions(self) -> int:
        with self._map_lock:
            return len(self._sessions)


SESSIONS = LiveSessionManager()


def _clamp(value, lo: float, hi: float, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _decode_frame(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")


def handle_frame(payload: dict, pipeline: ShelfPipeline) -> dict:
    """Process one webcam frame POSTed by the Live tab.

    Expects {"uuid", "image" (jpeg data-url), "imgsz", "conf", "threshold"}.
    Returns box coordinates in *sent-image pixels*; the browser draws the
    overlay itself, so the response stays tiny.
    """
    sid = str(payload.get("uuid") or "")[:64]
    if not sid:
        return {"ok": False, "error": "missing session id"}
    try:
        image = _decode_frame(payload["image"])
    except Exception as e:  # noqa: BLE001 - report to the browser instead
        return {"ok": False, "error": f"bad image payload: {e}"}

    rec = SESSIONS.get(sid, pipeline)
    t0 = time.perf_counter()
    with SESSIONS.model_lock:
        rec.imgsz = int(_clamp(payload.get("imgsz"), *IMGSZ_RANGE, 960))
        rec.conf = _clamp(payload.get("conf"), 0.01, 0.99, config.DETECT_CONF)
        rec.threshold = _clamp(payload.get("threshold"), 0.0, 1.0,
                               config.MATCH_THRESHOLD)
        result = rec.process(image)
    ms = (time.perf_counter() - t0) * 1000
    if ms > 0:
        fps_now = 1000.0 / ms
        rec.fps = fps_now if not rec.fps else 0.8 * rec.fps + 0.2 * fps_now

    matched = sum(1 for r in result.recognitions if r.is_match)
    return {
        "ok": True,
        "boxes": [
            {
                "id": r.track_id,
                "xyxy": [round(v, 1) for v in r.box.xyxy],
                "label": r.label,
                "match": bool(r.is_match),
            }
            for r in result.recognitions
        ],
        "img_w": image.width,
        "img_h": image.height,
        "n_objects": len(result.recognitions),
        "n_matched": matched,
        "n_new": result.n_new,
        "ms": round(ms, 1),
        "fps": round(rec.fps, 1),
        # Which detector produced these boxes: "custom" (trained SKU-110K) or
        # "world" (YOLO-World fallback). Shown in the Live status line.
        "detector": getattr(pipeline.detector, "active_backend", "?"),
        "summary": summary(result.recognitions) if result.changed else None,
    }


def handle_reset(payload: dict) -> dict:
    sid = str(payload.get("uuid") or "")[:64]
    if not sid:
        return {"ok": False, "error": "missing session id"}
    SESSIONS.reset(sid)
    return {"ok": True}


def handle_grasp(payload: dict, pipeline: ShelfPipeline) -> dict:
    """Grasp readiness for a product name (GET /live/api/grasp).

    Expects {"uuid", "name"}. Returns whether ONE track with that name has
    been stable for GRASP_WINDOW frames (see LiveRecognizer._update_grasp);
    a robotic arm should move using the returned de-jittered box/point, in
    the pixels of the frames sent to /live/api/frame (client must map them
    to 3D via camera calibration).
    """
    sid = str(payload.get("uuid") or "")[:64]
    if not sid:
        return {"ok": False, "error": "missing session id"}
    name = str(payload.get("name") or "").strip()
    rec = SESSIONS.get(sid, pipeline)
    with SESSIONS.model_lock:   # state is mutated by process() per frame
        status = rec.grasp_status(name)
    return {"ok": True, **status}


# T_ee_cam from data/handeye.json, cached by file mtime (None when absent).
_HE_MTIME: float = -1.0
_HE_T: np.ndarray | None = None


def _handeye_T() -> np.ndarray | None:
    global _HE_MTIME, _HE_T
    from .calibration import load_handeye
    try:
        mtime = config.HANDEYE_PATH.stat().st_mtime
    except OSError:
        return None
    if mtime != _HE_MTIME:
        try:
            _HE_T = load_handeye(config.HANDEYE_PATH)
        except Exception:   # noqa: BLE001 - uncalibrated -> camera frame only
            _HE_T = None
        _HE_MTIME = mtime
    return _HE_T


def handle_simdepth(payload: dict, pipeline: ShelfPipeline) -> dict:
    """Camera-frame 3D for a product name (GET /live/api/simdepth).

    Expects {"uuid", "name", "depth" (m plane), "hfov" (deg)}. The web stream
    carries no depth, so z is the synthetic plane and intrinsics are
    synthesised from the last frame size + hfov (see
    LiveRecognizer.simdepth_status). An empty `name` returns 3D points for
    every detected object (`points`). Intended for 3D cameras whose RGB UVC
    the browser opens directly (e.g. RealSense D435i); any webcam can be
    used, but the z value then carries no physical meaning.
    """
    sid = str(payload.get("uuid") or "")[:64]
    if not sid:
        return {"ok": False, "error": "missing session id"}
    name = str(payload.get("name") or "").strip()
    depth = _clamp(payload.get("depth"), 0.05, 5.0, 0.35)
    hfov = _clamp(payload.get("hfov"), 30.0, 120.0, 69.0)
    rec = SESSIONS.get(sid, pipeline)
    with SESSIONS.model_lock:   # process() mutates _last_recs/_grasp per frame
        if name:
            status = rec.simdepth_status(name, depth, hfov)
        else:
            status = rec.simdepth_all_status(depth, hfov)
    return {"ok": True, **status}
