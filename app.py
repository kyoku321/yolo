"""Gradio web UI for the shelf recognition demo.

Tabs:
  1. Register  – upload reference photos + metadata to add a SKU to the catalog.
  2. Recognize – upload a shelf photo; get boxes + matched SKU names.
  3. Live      – webcam view with per-frame detection; CLIP only for new
                 objects; the tab is a small custom UI (gr.HTML) talking to
                 two FastAPI routes registered in build_app().
  4. Catalog   – view / delete registered products.

The heavy models (YOLO + CLIP) are loaded lazily on first use so the UI opens
instantly.
"""
from __future__ import annotations

import asyncio

import gradio as gr
from PIL import Image

from shelf_demo import catalog_api, config, live as shelf_live
from shelf_demo.catalog_web import CATALOG_CSS, CATALOG_HTML, catalog_js
from shelf_demo.draw import draw_recognitions, summary
from shelf_demo.live_web import LIVE_CSS, LIVE_HTML, live_js

_PIPELINE = None


def pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from shelf_demo.pipeline import ShelfPipeline
        _PIPELINE = ShelfPipeline()
    return _PIPELINE


# ---- register --------------------------------------------------------------
def do_register(files, name, barcode, price, category):
    if not files:
        return "⚠️ Please upload at least one reference image."
    if not name or not name.strip():
        return "⚠️ Product name is required."
    images = [Image.open(f.name if hasattr(f, "name") else f) for f in files]
    try:
        sku_id = pipeline().register_product(
            images, name=name, barcode=barcode or "",
            price=float(price or 0.0), category=category or "",
        )
    except Exception as e:  # noqa: BLE001 - surface the message to the user
        return f"❌ {e}"
    return (f"✅ Registered '{name}' as SKU #{sku_id} "
            f"({len(images)} reference image(s)). "
            f"Catalog now holds {pipeline().catalog.n_vectors} vectors.")


# ---- recognize -------------------------------------------------------------
def do_recognize(image, conf, threshold, imgsz):
    if image is None:
        return None, "⚠️ Please upload a shelf image."
    results = pipeline().recognize(image, conf=conf, threshold=threshold,
                                   imgsz=int(imgsz))
    if not results:
        return image, ("No products detected. If you are using the default "
                       "COCO yolov8n weights, train on SKU-110K for real "
                       "shelf detection (see README).")
    return draw_recognitions(image, results), summary(results)


# ---- live (webcam) + catalog: FastAPI routes -------------------------------
def register_api_routes(app) -> None:
    """Attach the custom-UI JSON endpoints to the FastAPI app gradio serves.

    Frames arrive as JSON {"uuid", "image", params...}; the heavy work runs
    in a thread (asyncio.to_thread) so gradio's event loop stays responsive,
    and per-session recognizer state lives in shelf_demo.live.SESSIONS.
    """
    if getattr(app.state, "_shelf_api_routes", False):
        return
    app.state._shelf_api_routes = True

    from starlette.staticfiles import StaticFiles
    config.ensure_dirs()   # CROP_DIR must exist before StaticFiles mounts it
    app.mount("/catalog/photos",
              StaticFiles(directory=str(config.CROP_DIR), check_dir=False),
              name="catalog_photos")

    @app.post("/live/api/frame")
    async def live_frame(payload: dict):
        # to_thread keeps ~50-300 ms of MPS work off the event loop
        return await asyncio.to_thread(shelf_live.handle_frame, payload,
                                       pipeline())

    @app.post("/live/api/reset")
    async def live_reset(payload: dict):
        return await asyncio.to_thread(shelf_live.handle_reset, payload)

    @app.get("/live/api/grasp")
    async def live_grasp(uuid: str = "", name: str = ""):
        """Grasp readiness for a product name (robotic arm integration).

        Poll this while the Live camera runs; it turns ready=true only after
        one track with that name has been stable for GRASP_WINDOW frames.
        """
        return await asyncio.to_thread(
            shelf_live.handle_grasp, {"uuid": uuid, "name": name}, pipeline())

    @app.get("/live/api/simdepth")
    async def live_simdepth(uuid: str = "", name: str = "",
                            depth: float = 0.35, hfov: float = 69.0):
        """Camera-frame 3D for a product name via a synthetic depth plane.

        Poll this while the Live camera runs on a 3D camera (e.g. RealSense
        D435i RGB over UVC). `depth` is the constant metres plane the target
        sits on; `hfov` is the camera's horizontal field of view in degrees.
        """
        return await asyncio.to_thread(
            shelf_live.handle_simdepth,
            {"uuid": uuid, "name": name, "depth": depth, "hfov": hfov},
            pipeline())

    @app.get("/catalog/api/list")
    async def catalog_list():
        return await asyncio.to_thread(catalog_api.list_catalog, pipeline())

    @app.post("/catalog/api/delete")
    async def catalog_delete(payload: dict):
        return await asyncio.to_thread(catalog_api.delete_sku, pipeline(),
                                       payload)


def build_ui():
    with gr.Blocks(title="Shelf Product Recognition Demo") as demo:
        gr.Markdown(
            "# 🛒 Shelf Product Recognition Demo\n"
            "**YOLO detection → CLIP embedding → vector retrieval.** "
            "Register products from a few photos; adding a SKU never retrains a "
            "model."
        )

        with gr.Tab("1 · Register product"):
            with gr.Row():
                with gr.Column():
                    reg_files = gr.File(
                        label="Reference photos (1–5, front view)",
                        file_count="multiple", file_types=["image"],
                    )
                    reg_name = gr.Textbox(label="Name *", placeholder="Coca-Cola 500ml")
                    reg_category = gr.Textbox(label="Category", placeholder="Beverages")
                    reg_barcode = gr.Textbox(label="Barcode", placeholder="6901234567890")
                    reg_price = gr.Number(label="Price", value=0.0)
                    reg_btn = gr.Button("Register", variant="primary")
                reg_out = gr.Markdown()
            reg_btn.click(
                do_register,
                [reg_files, reg_name, reg_barcode, reg_price, reg_category],
                reg_out,
            )

        with gr.Tab("2 · Recognize shelf"):
            with gr.Row():
                with gr.Column():
                    rec_in = gr.Image(label="Shelf photo", type="pil")
                    rec_imgsz = gr.Dropdown(
                        [640, 1280, 1920, 2560], value=config.DETECT_IMGSZ,
                        label="Detection image size (raise for 4K / dense shelves)",
                    )
                    rec_conf = gr.Slider(0.01, 0.9, value=config.DETECT_CONF,
                                         step=0.01, label="Detection confidence")
                    rec_thr = gr.Slider(0.0, 1.0, value=config.MATCH_THRESHOLD,
                                        step=0.01, label="Match threshold (cosine)")
                    rec_btn = gr.Button("Recognize", variant="primary")
                with gr.Column():
                    rec_out_img = gr.Image(label="Result", type="pil")
                    rec_out_txt = gr.Textbox(label="Summary", lines=10)
            rec_btn.click(do_recognize, [rec_in, rec_conf, rec_thr, rec_imgsz],
                          [rec_out_img, rec_out_txt])

        with gr.Tab("3 · Live (webcam)"):
            gr.Markdown(
                "Point your camera at the shelf: YOLO detects every frame, "
                "CLIP only processes newly-appeared objects, and boxes are "
                "painted directly on the video at ~15 FPS. Click **Start** and "
                "allow camera access when prompted (requires localhost or HTTPS)."
            )
            gr.HTML(html_template=LIVE_HTML, css_template=LIVE_CSS,
                    js_on_load=live_js())

        with gr.Tab("4 · Catalog"):
            gr.HTML(html_template=CATALOG_HTML, css_template=CATALOG_CSS,
                    js_on_load=catalog_js())

    return demo


def build_app():
    """Blocks demo plus the FastAPI app carrying the custom-tab routes.

    `demo.launch()` rebuilds gradio's FastAPI app internally, so routes added
    to `demo.app` beforehand would be lost; passing `_app=` keeps ours alive.
    """
    from gradio.routes import App
    demo = build_ui()
    app = App()
    register_api_routes(app)
    return demo, app


if __name__ == "__main__":
    # Build the models on the main thread up front, so heavy native init
    # (torch / MPS) never happens inside a Gradio worker thread.
    print("Loading detection + embedding models (first run downloads weights) ...")
    p = pipeline()
    d, e = p.detector, p.embedder
    _kind_label = {"custom": "SKU-110K 自训权重",
                   "world": "YOLO-World 零样本（文本 prompt）",
                   "auto": "自训权重 + 零样本兜底（密集货架/单品特写自适应）",
                   "coco": "COCO 80 类（检不出零食/饮料）"}
    print()
    print("=" * 62)
    print("  当前使用的模型 (current models in use)")
    print("-" * 62)
    print(f"  检测器 Detector  : {d.kind} — {_kind_label.get(d.kind, '')}")
    print(f"      权重 weights : {d.weights}")
    print(f"      设备 device  : {d.device}")
    if d.kind == "auto":
        # Load the open-vocab fallback now, not inside the first cam slab of
        # frames — YOLO-World init is seconds, and a live session must not
        # stall on it. See AutoDetector.
        print("      兜底 fallback: 已预热 (YOLO-World)，单品特写时自动启用")
        d.warm()
    if d.prompts:
        print(f"      prompts      : {', '.join(d.prompts)}")
    print(f"  嵌入 CLIP        : {e.model_name} / {e.pretrained}")
    print(f"      维度 dim     : {e.dim}    设备 device: {e.device}")
    print(f"  商品库 Catalog   : {p.catalog.n_vectors} 个 SKU   @{config.DATA_DIR}")
    print("=" * 62)
    demo, fastapi_app = build_app()
    demo.launch(_app=fastapi_app)
