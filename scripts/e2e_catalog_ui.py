"""End-to-end test of the Catalog tab custom table (Playwright).

Registers a throwaway SKU through the real pipeline, checks the custom HTML
table renders it (with photo thumbnails and a per-row Delete button), clicks
Delete, and verifies the SKU + its reference photos are gone. The user data
in the catalog is left untouched.

Usage:  python scripts/e2e_catalog_ui.py
Requires: pip install playwright && playwright install chromium-headless-shell
(dev-only dependency; not needed to run the app itself)
"""
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sys.exit("Playwright not installed. Run: pip install playwright && "
             "playwright install chromium-headless-shell")

from PIL import Image

import app as shelf_app
from shelf_demo import config


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


TEST_NAME = "__e2e_delete_me__"


def main() -> int:
    pre = shelf_app.pipeline()  # pre-warm models
    img = Image.open("data/shelf.jpg").convert("RGB")
    img.thumbnail((640, 640))
    sku_id = pre.register_product([img], name=TEST_NAME, price=3.5)
    photos_before = list(config.CROP_DIR.glob(f"sku{sku_id:04d}_*"))
    assert len(photos_before) == 1, photos_before
    print(f"(seeded test SKU #{sku_id}, crop saved: {photos_before[0].name})")

    demo, fastapi_app = shelf_app.build_app()
    port = free_port()
    demo.launch(server_port=port, prevent_thread_lock=True, quiet=True,
                _app=fastapi_app)
    time.sleep(2)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_context(locale="en-US").new_page()
            page.on("dialog", lambda d: d.accept())  # confirm() -> OK
            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
            page.get_by_role("tab", name="Catalog").click()
            page.wait_for_selector('[data-role="refresh"]', timeout=20000)

            # wait until the test SKU row renders (fetch on mount)
            sel = f'[data-role="del-{sku_id}"]'
            page.wait_for_selector(sel, timeout=15000)

            meta = page.locator('[data-role="meta"]').inner_text()
            check("meta shows product/vector counts",
                  "product(s)" in meta and "vectors" in meta)

            row = page.locator(sel).first.locator("xpath=ancestor::tr")
            check("test SKU row visible with name",
                  TEST_NAME in row.inner_text())
            check("row has photo thumbnail(s)",
                  row.locator(".sct-photos img").count() >= 1)
            check("photo thumb src served by /catalog/photos",
                  "/catalog/photos/" in (row.locator(".sct-photos img")
                                         .first.get_attribute("src") or ""))
            check("other user SKUs also listed",
                  page.locator(".sct-table tbody tr").count() >= 2)

            # click Delete; confirm dialog auto-accepted
            page.locator(sel).click()
            page.wait_for_selector(sel, state="detached", timeout=15000)
            check("row removed from table after delete", True)
            check("delete banner shown",
                  f"Deleted SKU #{sku_id}" in
                  page.locator('[data-role="msg"]').inner_text())

            browser.close()
    finally:
        demo.close()

    check("test SKU gone from catalog db", pre.catalog.get(sku_id) is None)
    check("crop photo files removed from disk",
          not list(config.CROP_DIR.glob(f"sku{sku_id:04d}_*")))
    print("E2E RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
