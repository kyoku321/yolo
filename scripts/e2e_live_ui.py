"""End-to-end test of the Live (webcam) tab with a fake camera (Playwright).

Verifies the two bugs reported against the old gr.Image-streaming version:
  1. Start/Stop button always reflects real state (was: stuck on "Stop").
  2. The annotated view is actually a live animation (was: a single frame).

Frame/reset traffic is counted SERVER-side (deterministic); the browser is
used for UI-state assertions only.

Usage:  python scripts/e2e_live_ui.py
Requires: pip install playwright && playwright install chromium-headless-shell
(dev-only dependency; not needed to run the app itself)
"""
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sys.exit("Playwright not installed. Run: pip install playwright && "
             "playwright install chromium-headless-shell")

import app as shelf_app
from shelf_demo import live as shelf_live


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PORT = free_port()
ERRORS = []


class Counters:
    def __init__(self):
        self.frames = 0
        self.resets = 0


COUNTERS = Counters()

_orig_frame = shelf_live.handle_frame
_orig_reset = shelf_live.handle_reset


def counting_frame(payload, pipeline):
    COUNTERS.frames += 1
    return _orig_frame(payload, pipeline)


def counting_reset(payload):
    COUNTERS.resets += 1
    return _orig_reset(payload)


shelf_live.handle_frame = counting_frame
shelf_live.handle_reset = counting_reset


def canvas_state(page):
    return page.evaluate(
        """() => {
        const c = document.querySelector('[data-role="slv-root"] .slv-canvas');
        if (!c || !c.width) return {size: 'empty', sum: 0,
                                    draws: window.__slv_draws || 0};
        const x = c.getContext('2d');
        const d = x.getImageData(0, 0, c.width, c.height).data;
        let s = 0;
        const step = Math.floor(d.length / 2000) || 1;
        for (let i = 0; i < d.length; i += step) s = (s * 31 + d[i]) % 1000000;
        return {size: c.width + 'x' + c.height, sum: s,
                draws: window.__slv_draws || 0};
    }"""
    )


def main() -> int:
    shelf_app.pipeline()  # pre-warm models so the first frame is fast
    demo, fastapi_app = shelf_app.build_app()
    demo.launch(server_port=PORT, prevent_thread_lock=True, quiet=True,
                show_error=True, _app=fastapi_app)
    time.sleep(2)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-fake-device-for-media-stream",
                  "--use-fake-ui-for-media-stream"],
        )
        ctx = browser.new_context(locale="zh-CN", permissions=["camera"])
        page = ctx.new_page()
        page.on("pageerror", lambda e: ERRORS.append(f"pageerror: {str(e)[:200]}"))
        page.on("console", lambda m: ERRORS.append(f"console.{m.type}: {m.text[:150]}")
                if m.type == "error" else None)

        page.goto(f"http://127.0.0.1:{PORT}", wait_until="domcontentloaded")
        page.get_by_role("tab", name="Live").click()
        page.wait_for_selector('[data-role="toggle"]', timeout=20000)

        toggle = page.locator('[data-role="toggle"]')
        status = page.locator('[data-role="status"]')
        reset = page.locator('[data-role="reset"]')

        check("initial button reads \"Start\"", "Start" in toggle.inner_text())

        toggle.click()
        deadline = time.time() + 20
        while time.time() < deadline and "FPS" not in status.inner_text():
            time.sleep(0.5)
        check("status shows FPS after start", "FPS" in status.inner_text())
        check("button reads \"Stop\" while recording", "Stop" in toggle.inner_text())

        time.sleep(4)
        check("frames POSTed to /live/api/frame", COUNTERS.frames > 5)

        # animation: redraw loop ticking at ~15 Hz and canvas has real pixels
        s1 = canvas_state(page)
        time.sleep(1.0)
        s2 = canvas_state(page)
        draws_per_s = s2["draws"] - s1["draws"]
        check("canvas sized from video (non-empty)", s2["size"] != "empty")
        print(f"   canvas size: %s" % s2["size"])
        check("canvas has painted pixels", s2["sum"] != 0)
        disp_h = page.locator(".slv-canvas").bounding_box()["height"]
        check(f"displayed canvas height capped ({disp_h:.0f}px <= 520)",
              disp_h <= 520)
        check(f"redraw loop running ({draws_per_s} draws/s, >=5)", draws_per_s >= 5)
        # response-driven stream: server keeps answering with fresh timings
        check("stream is flowing to server", COUNTERS.frames > 5 and s2["draws"] > 0)

        # camera dropdown: populated, and switching does not kill the stream
        cam_sel = page.locator('[data-role="camera"]')
        check("camera dropdown exists and lists >=1 camera",
              cam_sel.locator("option").count() >= 1)
        page.evaluate(
            "document.querySelector('[data-role=\"camera\"]')"
            ".dispatchEvent(new Event('change'))")
        deadline = time.time() + 8
        hot_ok = False
        while time.time() < deadline:
            time.sleep(0.5)
            if ("Stop" in toggle.inner_text()
                    and "FPS" in status.inner_text()):
                hot_ok = True
                break
        check("camera hot-switch keeps stream running", hot_ok)

        # grasp endpoint (robotic arm): session id is exposed by the tab and the
        # endpoint responds with readiness + available names
        session = page.evaluate(
            "() => document.querySelector('[data-role=\"slv-root\"]')"
            ".dataset.slvSession")
        check("session id exposed for external clients", bool(session))
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/live/api/grasp"
                f"?uuid={session}&name=__none__", timeout=5) as r:
            grasp = json.load(r)
        print("   grasp resp:", str(grasp)[:300])
        check("grasp endpoint responds ok", grasp.get("ok") is True)
        check("grasp endpoint reports not-ready + available_names",
              grasp.get("ready") is False and "available_names" in grasp)

        frames_before_stop = COUNTERS.frames
        toggle.click()  # stop
        time.sleep(2.5)
        check("button back to \"Start\" after stop", "Start" in toggle.inner_text())
        check("no frames after stop", COUNTERS.frames - frames_before_stop <= 1)
        check("status reads \"Stopped\"", "Stopped" in status.inner_text())

        reset.click()  # reset works with no errors
        deadline = time.time() + 5
        got_reset = COUNTERS.resets > 0
        while not got_reset and time.time() < deadline:
            time.sleep(0.3)
            got_reset = COUNTERS.resets > 0
        check("Re-recognize hits /live/api/reset", COUNTERS.resets > 0)

        frames_before_restart = COUNTERS.frames
        toggle.click()  # start again
        deadline = time.time() + 15
        while time.time() < deadline:
            time.sleep(0.5)
            if COUNTERS.frames > frames_before_restart + 5:
                break
        check("frame stream resumes after restart", COUNTERS.frames > frames_before_restart + 5)
        toggle.click()  # final stop

        browser.close()

    demo.close()
    real_errors = [e for e in ERRORS if "AbortError" not in e]
    check("no browser JS errors", not real_errors)
    for e in real_errors[:5]:
        print("  ", e)

    print(f"(frames={COUNTERS.frames} resets={COUNTERS.resets})")
    print("E2E RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
