"""Frontend assets for the Live (webcam) tab.

Gradio's built-in `gr.Image(streaming=True)` record/stop widget has two
problems for this demo (Gradio 6.22): the record button label is driven by an
internal `stream_state` that app code cannot reset, so the button stays stuck
on "Stop"; and the server-side stream session lingers, which confused users
("the right pane only shows a single frame").

Instead we render our own tiny UI with gr.HTML and talk to two FastAPI routes
(`/live/api/frame`, `/live/api/reset`):

* The browser captures webcam frames on a timer, POSTs them as JPEG data-URLs,
  and receives back detection boxes (sent-image pixel coords) + status.
* Boxes are drawn client-side onto a canvas overlaid on the raw video frame,
  at ~15 FPS redraw rate — the video looks live even while the detector is
  catching up, and the Start/Stop button always reflects real state.

IMPORTANT for gr.HTML templating: none of the three strings below may contain
the character sequence $(brace) (gradio would treat it as a template slot) — all
JS here uses plain string concatenation for that reason.
"""
from __future__ import annotations

from . import config

LIVE_HTML = """
<div class="slv" data-role="slv-root">
  <div class="slv-stage">
    <canvas class="slv-canvas"></canvas>
    <div class="slv-hint">Press Start and point the camera at the shelf</div>
  </div>
  <div class="slv-bar">
    <button class="slv-btn primary" data-role="toggle">&#9654; Start</button>
    <button class="slv-btn" data-role="reset">&#8635; Re-recognize</button>
    <span class="slv-status" data-role="status">Idle</span>
  </div>
  <div class="slv-controls">
    <label class="slv-row">
      <span class="slv-name">Camera</span>
      <select data-role="camera" class="slv-cam"></select>
    </label>
    <label class="slv-row">
      <span class="slv-name">Detection size</span>
      <select data-role="imgsz">
        <option value="640">640 (fastest)</option>
        <option value="960" selected="selected">960 (recommended)</option>
        <option value="1280">1280 (most accurate)</option>
      </select>
    </label>
    <label class="slv-row">
      <span class="slv-name">Confidence</span>
      <input type="range" min="0.05" max="0.8" step="0.01"
             data-role="conf" class="slv-range" />
      <span class="slv-val" data-role="conf-val"></span>
    </label>
    <label class="slv-row">
      <span class="slv-name">Match threshold</span>
      <input type="range" min="0.40" max="0.95" step="0.01"
             data-role="thr" class="slv-range" />
      <span class="slv-val" data-role="thr-val"></span>
    </label>
  </div>
  <pre class="slv-summary" data-role="summary">Recognition summary will appear here</pre>
</div>
"""

LIVE_CSS = """
.slv { width: 100%; font-family: inherit; }
.slv-stage {
  position: relative; width: 100%; background: #0a0e13;
  border-radius: 12px; overflow: hidden; min-height: 220px;
  display: flex; align-items: center; justify-content: center;
}
/* internal buffer keeps video's native resolution (detection quality is
   unaffected); only the DISPLAY box is capped so the whole tab fits one
   viewport without scrolling */
.slv-canvas {
  display: block; width: auto; height: auto;
  max-width: 100%; max-height: min(52vh, 480px);
}
.slv-hint {
  position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; color: #94a3b8; font-size: 15px;
  padding: 24px; text-align: center; pointer-events: none;
}
.slv-bar { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
.slv-btn {
  padding: 8px 18px; border-radius: 8px; border: 1px solid #334155;
  background: #1e293b; color: #e2e8f0; font-size: 14px; font-weight: 600;
  cursor: pointer;
}
.slv-btn:hover { filter: brightness(1.15); }
.slv-btn.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
.slv-btn.stop { background: #dc2626; border-color: #dc2626; color: #fff; }
.slv-status { color: #94a3b8; font-size: 13px; }
.slv-controls { display: flex; flex-wrap: wrap; gap: 8px 22px; margin: 4px 0 8px; }
.slv-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.slv-name { color: #64748b; white-space: nowrap; }
.slv-row select {
  background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
  border-radius: 6px; padding: 3px 6px;
}
.slv-cam { min-width: 200px; max-width: 320px; }
.slv-range { width: 120px; }
.slv-val { width: 42px; color: #94a3b8; font-variant-numeric: tabular-nums; }
.slv-summary {
  margin: 6px 0 0; padding: 10px 12px; min-height: 52px; max-height: 150px;
  overflow: auto; background: #0f172a; color: #cbd5e1;
  border-radius: 8px; font-family: ui-monospace, monospace; font-size: 12.5px;
  white-space: pre-wrap;
}
"""


def live_js() -> str:
    """Live-tab JS with the current env-configured defaults baked in."""
    conf = repr(round(config.DETECT_CONF, 3))
    thr = repr(round(config.MATCH_THRESHOLD, 3))
    # NOTE: no $(brace) sequences allowed in this string (gradio templates).
    js = """
(function () {
  var root = element.querySelector('[data-role="slv-root"]');
  if (!root || root.dataset.liveInit === "1") return;
  root.dataset.liveInit = "1";

  var canvas = root.querySelector(".slv-canvas");
  var hint = root.querySelector(".slv-hint");
  var btnToggle = root.querySelector('[data-role="toggle"]');
  var btnReset = root.querySelector('[data-role="reset"]');
  var statusEl = root.querySelector('[data-role="status"]');
  var sumEl = root.querySelector('[data-role="summary"]');
  var imgszSel = root.querySelector('[data-role="imgsz"]');
  var confRange = root.querySelector('[data-role="conf"]');
  var thrRange = root.querySelector('[data-role="thr"]');
  var confVal = root.querySelector('[data-role="conf-val"]');
  var thrVal = root.querySelector('[data-role="thr-val"]');
  var camSel = root.querySelector('[data-role="camera"]');

  var DEF_CONF = __DEF_CONF__, DEF_THR = __DEF_THR__;
  confRange.value = String(DEF_CONF); thrRange.value = String(DEF_THR);
  confVal.textContent = DEF_CONF; thrVal.textContent = DEF_THR;
  confRange.oninput = function () { confVal.textContent = confRange.value; };
  thrRange.oninput = function () { thrVal.textContent = thrRange.value; };

  // Camera picker: labels are empty until permission is granted, so the
  // list is rebuilt with real names after the first successful getUserMedia.
  function refreshCameraList(activeDeviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return;
    }
    navigator.mediaDevices.enumerateDevices().then(function (devs) {
      var cams = devs.filter(function (d) { return d.kind === "videoinput"; });
      var keepId = activeDeviceId || camSel.value;
      camSel.innerHTML = "";
      cams.forEach(function (d, i) {
        var opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || ("Camera " + (i + 1));
        camSel.appendChild(opt);
      });
      var ids = cams.map(function (d) { return d.deviceId; });
      if (ids.indexOf(keepId) >= 0) camSel.value = keepId;
    }).catch(function () {});
  }
  refreshCameraList();
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange",
        function () { refreshCameraList(); });
  }

  camSel.onchange = function () {
    if (!running) return;   // applies on next Start
    // hot-switch while running: release the old track and reopen
    if (media) {
      media.getTracks().forEach(function (t) { t.stop(); });
      media = null; video = null;
    }
    setStatus("Switching camera...");
    ensureCamera(function () { setStatus("Running..."); },
                 function (msg) { setStatus(msg, true); });
  };

  var SESSION = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : "srv-" + Date.now() + "-" + Math.floor(Math.random() * 1e9);
  root.dataset.slvSession = SESSION;   // exposed for external clients (e.g. arm controllers)
  // bumped every start/stop; in-flight responses from an older generation
  // are ignored so they cannot overwrite the stopped state
  var generation = 0;
  var API = location.pathname.replace(/\\/$/, "") + "live/api";

  var video = null, media = null, running = false, inflight = false;
  var pending = null;
  var grab = document.createElement("canvas");
  var gctx = grab.getContext("2d");
  var ctx = canvas.getContext("2d");
  var lastBoxes = [], lastImgW = 0, lastImgH = 0;

  function setStatus(text, isErr) {
    statusEl.textContent = text;
    statusEl.style.color = isErr ? "#f87171" : "";
  }

  function openStream(withDevice, onOk, onErr) {
    var c = {
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    };
    if (withDevice && camSel.value) {
      c.video.deviceId = { exact: camSel.value };
    }
    navigator.mediaDevices.getUserMedia(c).then(function (stream) {
      media = stream;
      video = document.createElement("video");
      video.muted = true; video.playsInline = true; video.autoplay = true;
      video.srcObject = stream;
      video.onloadeddata = function () { onOk(); };
      // fill in real camera names now that permission is granted
      var tk = stream.getVideoTracks()[0];
      var did = tk && tk.getSettings ? tk.getSettings().deviceId : null;
      refreshCameraList(did);
    }).catch(onErr);
  }

  function ensureCamera(onOk, onErr) {
    if (media) { onOk(); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      onErr("Camera not supported (needs localhost or HTTPS page)");
      return;
    }
    openStream(true, onOk, function (e) {
      // stale/unsupported deviceId (e.g. camera unplugged) -> default device
      if (e && e.name === "OverconstrainedError") {
        openStream(false, onOk, function (e2) {
          onErr("Cannot access camera: " + ((e2 && e2.message) || e2) +
                " (check permission; page must be on localhost/HTTPS)");
        });
      } else {
        onErr("Cannot access camera: " + ((e && e.message) || e) +
              " (check permission; page must be on localhost/HTTPS)");
      }
    });
  }

  function drawOverlay() {
    if (!running || !video || video.readyState < 2) return;
    var vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw; canvas.height = vh;
    }
    ctx.drawImage(video, 0, 0, vw, vh);
    var sx = lastImgW ? vw / lastImgW : 1;
    var fontH = Math.max(13, Math.round(vw / 90));
    ctx.font = "bold " + fontH + "px system-ui, sans-serif";
    ctx.lineWidth = Math.max(2, Math.round(vw / 480));
    for (var i = 0; i < lastBoxes.length; i++) {
      var b = lastBoxes[i];
      var x1 = b.xyxy[0] * sx, y1 = b.xyxy[1] * sx;
      var x2 = b.xyxy[2] * sx, y2 = b.xyxy[3] * sx;
      var color = b.match ? "#22c55e" : "#ef4444";
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      var label = b.label;
      var tw = ctx.measureText(label).width;
      var ty = Math.max(0, y1 - fontH - 6);
      ctx.fillStyle = color;
      ctx.fillRect(x1, ty, tw + 8, fontH + 6);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 4, ty + fontH + 1);
    }
  }

  function tick() {
    if (!running) return;
    if (inflight) { pending = setTimeout(tick, 60); return; }
    if (!video || video.readyState < 2 || !video.videoWidth) {
      pending = setTimeout(tick, 150); return;
    }
    var vw = video.videoWidth, vh = video.videoHeight;
    var scale = Math.min(1, 1280 / vw);
    var w = Math.round(vw * scale), h = Math.round(vh * scale);
    grab.width = w; grab.height = h;
    gctx.drawImage(video, 0, 0, w, h);

    var controller = new AbortController();
    var killer = setTimeout(function () { controller.abort(); }, 8000);
    var gen = generation;
    inflight = true;
    fetch(API + "/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uuid: SESSION,
        image: grab.toDataURL("image/jpeg", 0.7),
        imgsz: Number(imgszSel.value),
        conf: Number(confRange.value),
        threshold: Number(thrRange.value)
      }),
      signal: controller.signal
    }).then(function (r) { return r.json(); }).then(function (resp) {
      clearTimeout(killer); inflight = false;
      if (gen !== generation || !running) return;
      if (resp && resp.ok) {
        lastBoxes = resp.boxes;
        lastImgW = resp.img_w; lastImgH = resp.img_h;
        setStatus(resp.fps + " FPS · " + resp.n_objects + " objects · "
          + resp.n_matched + " matched · server " + resp.ms + "ms"
          + (resp.n_new ? " · " + resp.n_new + " new objects" : ""));
        if (resp.summary !== null && resp.summary !== undefined) {
          sumEl.textContent = resp.summary;
        }
      } else {
        setStatus("Server error: " + ((resp && resp.error) || "unknown"), true);
      }
      pending = setTimeout(tick, 20);
    }).catch(function (e) {
      clearTimeout(killer); inflight = false;
      if (gen !== generation || !running) return;
      setStatus("Connection failed: " + ((e && e.message) || e), true);
      pending = setTimeout(tick, 800);
    });
  }

  // setTimeout instead of rAF: keeps drawing when rAF is throttled
  window.__slv_draws = 0;   // e2e/diagnostic: how often drawOverlay ran
  window.setInterval(function () {
    window.__slv_draws += 1;
    drawOverlay();
  }, 66);

  btnToggle.onclick = function () {
    if (!running) {
      setStatus("Opening camera...");
      ensureCamera(function () {
        generation += 1;
        running = true;
        hint.style.display = "none";
        btnToggle.textContent = "■ Stop";
        btnToggle.classList.add("stop");
        btnToggle.classList.remove("primary");
        setStatus("Running...");
        tick();
      }, function (msg) { setStatus(msg, true); });
    } else {
      running = false;
      generation += 1;
      if (pending) { clearTimeout(pending); pending = null; }
      if (media) {
        media.getTracks().forEach(function (t) { t.stop(); });
        media = null; video = null;
      }
      btnToggle.textContent = "▶ Start";
      btnToggle.classList.add("primary");
      btnToggle.classList.remove("stop");
      hint.style.display = "flex";
      lastBoxes = []; lastImgW = 0; lastImgH = 0;
      canvas.width = 0; canvas.height = 0;
      setStatus("Stopped");
    }
  };

  btnReset.onclick = function () {
    lastBoxes = [];
    sumEl.textContent = "Reset: new objects will be re-embedded via CLIP";
    fetch(API + "/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uuid: SESSION })
    }).catch(function () {});
  };
})();
"""
    return js.replace("__DEF_CONF__", conf).replace("__DEF_THR__", thr)
