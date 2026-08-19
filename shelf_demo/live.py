"""Live (webcam) recognition: detect every frame, embed only NEW objects.

Full per-frame recognition is too slow for video on this class of hardware:
YOLO @960 is ~45 ms/frame on an M1 (~22 FPS) but CLIP embedding costs
~150-300 ms and grows with box count. This tracker detects boxes on every
frame and only runs CLIP + retrieval for boxes that don't match an existing
track (IoU >= MATCH_IOU); labels are carried across frames by tracking.
On a static scene the stream runs at detection speed; panning to new
products costs one embedding batch, then it's smooth again.

Also contains the JSON plumbing for the Live tab: the browser posts webcam
frames to a FastAPI route; per-session LiveRecognizer state is kept
server-side keyed by a client-generated uuid (see handle_frame/handle_reset).

Grasp readiness (robotic arm integration): each product name carries a
stability window over its best track — a name is graspable only after one
specific track has held its position (IoU >= GRASP_IOU) for GRASP_WINDOW
consecutive frames while no new objects appeared. Exposed via handle_grasp
(GET /live/api/grasp?uuid=...&name=...).
"""
from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field

from PIL import Image

from . import config
from .detector import Box
from .draw import summary
from .pipeline import Recognition, ShelfPipeline

MATCH_IOU = 0.5    # min IoU to associate a detection with an existing track
MAX_MISSES = 4     # frames a track may be missing before being dropped

SESSION_TTL_S = 120.0       # idle live sessions are dropped after this
IMGSZ_RANGE = (480, 1536)   # accepted detection sizes from the Live tab

# Grasp readiness (robotic arm, GET /live/api/grasp). A product name is
# ready when ONE specific track with that name has been present for
# GRASP_WINDOW consecutive frames, staying put frame-over-frame
# (IoU >= GRASP_IOU) while no new objects appeared (a frame with new
# objects resets every window). The returned box is the mean of the window
# (de-jittered). GRASP_MAX_AGE_MS: don't offer a grasp on a stale session.
GRASP_WINDOW = 5            # frames of stability required (~300 ms @16 FPS)
GRASP_IOU = 0.9             # min box IoU between consecutive window frames
GRASP_MAX_AGE_MS = 1000.0   # state older than this is not graspable


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


@dataclass
class _Track:
    box: Box
    rec: Recognition
    misses: int = 0
    id: int = -1            # stable per-track id, exposed to clients


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
        self._tracks: list[_Track] = []
        self._prev_n = 0
        self.fps = 0.0        # EMA of server-side processing speed
        self._next_track_id = 0
        self._grasp: dict[str, _GraspState] = {}   # name -> stability window
        self._last_frame_ts: float | None = None
        self._last_img_size: tuple[int, int] | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_n = 0
        for st in self._grasp.values():
            st.reset()

    def process(self, image: Image.Image) -> LiveResult:
        image = image.convert("RGB")
        self._last_frame_ts = time.monotonic()
        self._last_img_size = (image.width, image.height)
        pipe = self.pipeline
        boxes = pipe.detector.detect(image, conf=self.conf, imgsz=self.imgsz)

        # 1. associate detections to existing tracks (greedy best IoU)
        assigned: dict[int, _Track] = {}
        for tr in self._tracks:
            best_i, best_v = -1, MATCH_IOU
            for i, b in enumerate(boxes):
                if i in assigned:
                    continue
                v = _iou(tr.box, b)
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                tr.box, tr.misses = boxes[best_i], 0
                assigned[best_i] = tr
            else:
                tr.misses += 1
        n_before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.misses <= MAX_MISSES]
        expired = len(self._tracks) != n_before

        # 2. CLIP + retrieval only for boxes with no track
        new_idx = [i for i in range(len(boxes)) if i not in assigned]
        if new_idx:
            recs = pipe.match_boxes(image, [boxes[i] for i in new_idx],
                                    self.threshold)
            for i, rec in zip(new_idx, recs):
                tr = _Track(boxes[i], rec, id=self._next_track_id)
                self._next_track_id += 1
                assigned[i] = tr
                self._tracks.append(tr)

        # 3. emit current recognitions in box order
        recognitions: list[Recognition] = []
        for i, b in enumerate(boxes):
            tr = assigned.get(i)
            if tr is None:
                continue
            recognitions.append(Recognition(b, tr.rec.sku_id, tr.rec.name,
                                            tr.rec.score, tr.rec.is_match,
                                            track_id=tr.id))

        # 4. grasp readiness: update per-name stability windows
        self._update_grasp(assigned, len(new_idx))

        changed = (bool(new_idx) or expired
                   or len(recognitions) != self._prev_n)
        self._prev_n = len(recognitions)
        return LiveResult(recognitions, changed, len(new_idx))

    def _update_grasp(self, assigned: dict[int, _Track],
                      n_new: int) -> None:
        """Maintain per-name stability windows (see GRASP_WINDOW comment).

        Locks onto ONE track per name (the first matched box in box order).
        The window slides as long as that same track stays put; a track
        change, a jump (IoU < GRASP_IOU) or a frame with new objects resets
        it.
        """
        current: dict[str, _Track] = {}
        for tr in assigned.values():
            if tr.rec.is_match and tr.rec.name and tr.rec.name not in current:
                current[tr.rec.name] = tr

        if n_new > 0:          # scene in flux: nobody is graspable yet
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
        available = sorted({t.rec.name for t in self._tracks
                            if t.rec.is_match and t.rec.name})
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
