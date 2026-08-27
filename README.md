# 🛒 超市货架商品识别 Demo

给新人写的完整文档：先讲清楚**原理和架构**，再讲**怎么跑起来**，最后讲
**怎么训练 SKU-110K 检测器**和**怎么调试**。

---

## 1. 这是什么？

输入一张货架照片，输出**每个商品的位置框 + 商品名**（以及价格、条码等元数据）：

```
货架照片 ──► YOLO 检测出 N 个商品框 ──► 逐框裁剪
                                          │
                              CLIP 把每张裁剪图变成 512 维向量
                                          │
                              和「商品库」里的向量比余弦相似度
                                          │
                                命中 → 显示商品名 / 不命中 → Unknown
```

这就是业界主流的 **检索式（retrieval-based）商品识别**方案（SKU-110K /
coarse-to-fine 论文的思路）：

- **检测**（YOLO）只回答「哪里有商品」，是一个**单类别**问题；
- **识别**（CLIP + 向量检索）回答「这是哪个 SKU」，是一个**检索**问题。

**核心优势**：上新一个商品，只需在 Web 页面上传 1–5 张参考照片入库，
**不用重训任何模型**。这正是检索式方案在零售业成为事实标准的原因。

### 新人必须先懂的 5 个概念

| 概念 | 一句话解释 |
|------|-----------|
| **embedding（向量）** | CLIP 把任意一张图变成一个 512 维数字向量。长得像的图，向量也接近。注册和识别用**同一个** CLIP 模型，所以向量在同一个「空间」里可以比。 |
| **余弦相似度** | 两个向量夹角的小数表示，0~1（本项目向量都做了 L2 归一化，所以点积 = 余弦）。≥ `MATCH_THRESHOLD`（默认 0.65）才算「认出」，否则显示 Unknown。 |
| **IoU** | 两个框重叠面积 ÷ 并集面积。NMS 用它去重，实时跟踪用它判断「这个框和上一帧的哪个目标是不是同一个商品」。 |
| **NMS（非极大值抑制）** | 检测器经常对同一个商品打出好几个几乎重合的框，NMS 按 IoU 阈值把重复框压掉。 |
| **开放词表检测（YOLO-World）** | 不训练、靠文字提示词（"bottle", "package"…）检测任意物体的检测器。零训练可用，但框偏松。 |

---

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────────────────┐
│ 浏览器（Gradio Web UI，http://127.0.0.1:7860）                          │
│                                                                        │
│  Tab 1 Register    Tab 2 Recognize   Tab 3 Live        Tab 4 Catalog  │
│  上传参考图入库     上传货架图识别    摄像头实时标注     商品库表格管理   │
│  (Gradio 表单)     (Gradio 表单)    (自研 HTML/JS)    (自研 HTML/JS)   │
└───────┬──────────────────┬─────────────────┬──────────────────┬────────┘
        │                  │                 │ JSON over HTTP   │ JSON over HTTP
        │                  │            POST /live/api/frame  GET /catalog/api/list
        │                  │            POST /live/api/reset  POST /catalog/api/delete
        │                  │            GET  /live/api/grasp
        ▼                  ▼           （机械臂抓取就绪接口）
┌────────────────────────────────────────────────────────────────────────┐
│ app.py — Gradio Blocks + FastAPI 路由（重模型懒加载，主线程预热）         │
└───────┬──────────────────┬─────────────────┬──────────────────┬────────┘
        │                  │                 │                  │
        ▼                  ▼                 ▼                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ shelf_demo/pipeline.py  —  ShelfPipeline（端到端编排，唯一入口）          │
│   register_product()   recognize()   match_boxes()   delete_product()  │
│  ┌──────────────┬───────────────┬──────────────────────┬────────────┐  │
│  ▼              ▼               ▼                      ▼            │  │
│ detector.py  embedder.py   database.py              crop 文件管理    │  │
│ YOLO 检测     CLIP 向量     SQLite 元数据            data/crops/     │  │
│ (3 种后端)    (512维)        + NumPy 向量库                              │  │
│                                        data/catalog.sqlite            │  │
│                                        data/vectors.npz               │  │
│  live.py  —  LiveRecognizer（实时跟踪：检测每帧跑、CLIP 只跑新目标）      │  │
│  live_web.py / catalog_web.py — 自研前端模板（HTML/CSS/JS）            │  │
│  catalog_api.py — Catalog 页 JSON handler                            │  │
│  draw.py — 画框/摘要    config.py — 全部配置（env 可覆盖）              │  │
└────────────────────────────────────────────────────────────────────────┘
```

**一句话总结模块职责**：

| 模块 | 职责 |
|------|------|
| `shelf_demo/config.py` | 所有可调参数（路径/检测/CLIP/检索/设备），全部可用环境变量覆盖 |
| `shelf_demo/detector.py` | `Detector`：封装 YOLO 推理，输出 `Box(x1,y1,x2,y2,conf)` 列表。3 种后端：`custom` / `world` / `coco` |
| `shelf_demo/embedder.py` | `Embedder`：open_clip 图像编码器，图片 → L2 归一化的 (N, 512) 向量 |
| `shelf_demo/database.py` | `Catalog`：SQLite 存 SKU 元数据 + `VectorStore`（NumPy 精确余弦近邻）存向量。`add / search / delete` |
| `shelf_demo/pipeline.py` | `ShelfPipeline`：把上面三块串起来。注册、识别、删除都从这里进出 |
| `shelf_demo/live.py` | 实时识别：`LiveRecognizer`（IoU 跟踪器）+ 会话管理 + 帧 JSON 处理 |
| `shelf_demo/live_web.py` | Live 页前端（canvas 画框、getUserMedia 抓帧、参数滑杆） |
| `shelf_demo/catalog_api.py` / `catalog_web.py` | Catalog 页后端 JSON handler / 前端表格 |
| `shelf_demo/draw.py` | 识别结果画框（绿=命中，红=Unknown）+ 文本摘要 |
| `app.py` | Gradio UI 4 个 Tab + FastAPI 自定义路由挂载 |
| `scripts/` | 训练（§8）、调试（§9）、端到端测试（§13）脚本 |

---

## 3. 目录结构

```
yolo/
├── app.py                    # Web 入口：Gradio 4 个 Tab + FastAPI 路由
├── requirements.txt
├── yolov8n.pt                # COCO 预训练权重（6 MB，coco 后端用）
├── yolov8s-worldv2.pt        # YOLO-World 权重（25 MB，world 后端用）
├── weights/
│   ├── sku110k_best.pt       # ★ 本项目在 SKU-110K 上训练的单类别检测器（存在则自动启用）
│   └── clip/ViT-B-32.pt      # 预下载的 CLIP 权重（默认 CLIP 是 ViT-B-16，首次运行自动下载）
├── shelf_demo/               # 核心代码（见 §2 职责表）
│   ├── config.py  detector.py  embedder.py  database.py
│   ├── pipeline.py  live.py  draw.py
│   ├── live_web.py  catalog_api.py  catalog_web.py
├── scripts/
│   ├── train_sku110k.py      # ★ 训练 SKU-110K 货架检测器（见 §8）
│   ├── smoke_test.py         # 无界面端到端自检
│   ├── debug_detect.py       # 「检测不到商品」分阶段诊断
│   ├── compare_detectors.py  # custom vs YOLO-World 框质量对比图
│   ├── sweep_detect.py       # imgsz/conf/NMS-iou 参数扫描
│   ├── test_grasp.py         # 抓取就绪接口（机械臂）单测，无模型秒跑
│   ├── e2e_live_ui.py        # [dev] Live 页 Playwright 端到端测试
│   └── e2e_catalog_ui.py     # [dev] Catalog 页 Playwright 端到端测试
├── datasets/SKU-110K/        # 训练数据（train.txt 8219 / val 588 / test 2936，单类 object）
├── runs/detect/...           # 训练产物（loss 曲线、best.pt 等）
├── data/                     # 运行期数据（见 §7）
└── docs/plans/               # 设计文档（实时视频标注的演进记录）
```

---

## 4. 快速开始

```bash
# 1. 环境（建议 Python 3.12）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 注：首次运行时 ultralytics 会自动补装 YOLO-World 需要的 openai CLIP 文本编码器

# 2. 自检（只验证「注册→检索」闭环，不需要检测器/摄像头；首次会下载 CLIP 权重 ~600 MB）
python scripts/smoke_test.py

# 3. 启动 Web UI
python app.py            # 打开 http://127.0.0.1:7860
```

启动时会在**主线程**预热 YOLO + CLIP 模型（MPS/torch 的原生初始化不适合放在
Gradio 工作线程里），之后 UI 秒开、识别不卡。

**使用流程**：

1. **Tab 1 · Register**：上传商品正面图 1–5 张 + 名称（必填）/价格/条码/类目 → 入库。
2. **Tab 2 · Recognize**：上传货架照片 → 左侧出画框结果图（绿框=认出，红框=Unknown），右侧出数量统计。可调 imgsz / conf / 匹配阈值。
3. **Tab 3 · Live**：点 Start 授权摄像头，对准货架 → 实时画框 + 商品名跟随画面。可调摄像头、检测分辨率、conf、阈值；Re-recognize 强制全量重识别。
4. **Tab 4 · Catalog**：表格查看所有 SKU（含注册照片缩略图，点击放大），每行 Delete 删除该 SKU（向量 + 元数据 + 照片文件一起清掉）。

> 浏览器摄像头只在「安全上下文」可用：`http://127.0.0.1:7860` 直接可用；
> 局域网 IP 访问需要 HTTPS 或 `gradio launch(share=True)`。

---

## 5. 每个功能背后的架构（逐 Tab 拆解）

### 5.1 Tab 1 · Register：商品入库

**调用链**：

```
gr.Button("Register")
  └─► app.py do_register(files, name, barcode, price, category)
        └─► pipeline.register_product(images, ...)
              ├─► Embedder.embed(images)          # ① CLIP：N 张图 → (N, 512) L2 归一化向量
              ├─► Catalog.add_product(...)        # ② SQLite 插一行 + vectors.npz 追加 N 个向量（打同一个 sku_id）+ 落盘
              └─► _save_reference_crops(...)      # ③ 参考照片存到 data/crops/skuNNNN_<时间戳>_i.jpg（供 Catalog 页展示）
```

**关键点**：

- 一个 SKU 拥有 **N 个向量**（N = 参考照片张数，1–5 张）。多张照片 = 多角度
  覆盖，检索时「该 SKU 的最优向量」胜出（见 5.2）。
- 入库**不碰任何模型权重**——这就是检索式方案的卖点：新品上线秒级完成。
- 注册照片保存为文件只是为了 Catalog 页展示与人工核对，检索本身只认向量。

### 5.2 Tab 2 · Recognize：货架识别（核心管线）

**调用链**：

```
gr.Button("Recognize")
  └─► app.py do_recognize(image, conf, threshold, imgsz)
        └─► pipeline.recognize(shelf_image, ...)
              ├─► Detector.detect(image, conf, imgsz)      # ① YOLO → [Box]（单类别，agnostic NMS）
              └─► pipeline.match_boxes(image, boxes, thr)  # ② 逐框裁剪 → CLIP 批量编码 → 向量检索
                        └─► Catalog.search(embeddings, topk=5)
                              即 (n,512) @ (512,N) 一次矩阵乘法 + 按 SKU 取最优
              └─► draw_recognitions(...) + summary(...)    # ③ 画框 + 统计文本
```

**关键点**：

- **检测与识别彻底解耦**：检测器只管「框」，SKU 是谁完全由向量检索决定。
  所以换检测器/换 CLIP 模型/增删商品，互相不影响。
- 一个框 → 一个 crop → 一个向量 → 与库内所有向量算余弦，**按 SKU 聚合取
  最高分**（一个 SKU 有 5 张参考图就有 5 个向量，谁高算谁的），top1 分数
  ≥ `MATCH_THRESHOLD` 判为命中，否则 `Unknown (score)`。
- `match_boxes` 从 `recognize` 里拆出来，是为了让 Live 页（5.3）只对
  **新目标**跑 CLIP——照片页和实时页共用同一套识别逻辑。
- `agnostic_nms=True`：YOLO-World 一个物体可能被 "package"/"box"/"pouch"
  多个提示词重复命中，按类别分开的 NMS 去不掉，跨类 NMS 才能压成 1 框。

### 5.3 Tab 3 · Live：摄像头实时标注（性能设计的核心）

**为什么不能每帧全量跑？** M1 MPS 实测：YOLO @960 ≈ 45 ms/帧（~22 FPS），
但 CLIP 一个 crop ≈ 150–300 ms，一帧 30 个框就是 5–9 秒。所以核心策略是：

> **YOLO 每帧都跑（快），CLIP 只对「新出现的目标」跑（贵）。**
> 目标身份靠跟踪维持，标签跨帧复用。

**前端**（自研 HTML/JS，`live_web.py`）：

```
浏览器 getUserMedia 抓摄像头帧（缩到 ≤1280px，JPEG quality 0.7）
  → 每帧 POST /live/api/frame {uuid, image(data-url), imgsz, conf, threshold}
     （同一时刻只允许 1 个在途请求，没回来就跳过本轮，防积压；8s 超时中止）
  → 服务端回 JSON {boxes[{xyxy,label,match}], n_objects, n_matched, n_new, ms, fps}
  → 浏览器把视频帧 + 框/标签画在 <canvas> 上，独立 66 ms 定时器重绘（~15 FPS）
```

覆盖框由**浏览器画**而不是服务端回传绘制好的 JPEG——网络负载极小（只有
坐标和文字），且检测慢一点画面照样流畅。

**后端**（`live.py`）：

```
POST /live/api/frame  (asyncio.to_thread 把 50–300ms 的 MPS 计算踢出事件循环)
  └─► handle_frame(payload, pipeline)
        ├─► SESSIONS.get(uuid) → LiveRecognizer     # 每浏览器会话一个，uuid 由前端 crypto.randomUUID() 生成
        │     （空闲 120s 自动回收；_map_lock 保护会话表，model_lock 串行化模型调用——
        │      MPS 不喜欢并发推理，两把锁绝不嵌套，避免自死锁）
        └─► rec.process(image)
              1) YOLO 检测（每帧，imgsz 默认 960）
              2) 贪婪 IoU 关联：每个 track 找 IoU≥0.5 的最佳检测框
                 - 关联上 → 框更新，沿用 track 里的 SKU 标签（零 CLIP 开销）
                 - 关联不上 → misses+1；连续 4 帧丢失 → track 淘汰
              3) 未关联上的「新框」批量走 pipeline.match_boxes（一次 CLIP 前向）
              4) 输出当前所有框的识别结果 + changed 标志（没变化就不回传摘要文本）
```

**效果**：货架静止时帧率 ≈ 纯检测上限（静态帧 ~61 ms/帧 ≈ 16 FPS）；
镜头移到新品上时，那一帧多一次批量 embedding（~1.5 s），之后又恢复流畅。

**为什么前端不用 Gradio 自带的 `gr.Image(streaming=True)`？** Gradio 6 的
录制按钮文案由内部 `stream_state` 驱动，app 层无法复位——点停止后按钮一直
卡在 "Stop"，服务端会话也不真正结束（源码级诊断确认，详见
`docs/plans/2026-08-12-realtime-video-annotation-design.md`）。自研前端后
Start/Stop 状态始终真实；前端还带 **generation 计数**：停止后晚到的在途
响应会被丢弃，不会把「已停止」状态覆盖回去。

**Re-recognize 按钮**：POST `/live/api/reset` 清空该会话的 track 缓存，
下一帧所有目标重新过 CLIP。什么时候需要点它？库里的 SKU 被删除/重新注册后
（track 里的标签是注册时刻的快照）。

**响应里的 `id` 字段**：每个框带稳定 `track_id`（会话内自增，track 存活期间不变）。
货架上多个同款商品时，靠它跨帧区分"哪一个物理目标"——机械臂必须用它，不能靠名字。

#### 机械臂集成：`GET /live/api/grasp`

想让机械臂去抓某个名称的商品，服务端提供「抓取就绪」接口，把时机判断逻辑
封装在跟踪器内部（`LiveRecognizer._update_grasp` 每帧维护每个名称的稳定性窗口）：

```
GET /live/api/grasp?uuid=<Live 会话 id>&name=<商品名>
→ {ok, ready, reason, track_id, box[x1,y1,x2,y2], point[cx,cy],
   frames_stable, age_ms, img_w, img_h, available_names[]}
```

**`ready=true`（= 最佳抓取时刻）的条件**：该名称的**同一个 track** 连续
`GRASP_WINDOW=5` 帧（~300 ms @16 FPS）出现，且：

- 相邻两帧框 IoU ≥ `GRASP_IOU=0.9`（位置收敛，滤掉 YOLO 框抖动）；
- 期间没有新目标入画（`n_new>0` 的帧会重置所有窗口 → 场景静止才算数）；
- 会话 1 秒内（`GRASP_MAX_AGE_MS`）处理过帧（防止对着停掉的摄像头抓）。

返回的 `box` 是窗口 5 帧的**平均框**（消抖），`point` 是其中心点。

**机械臂控制器推荐流程**：

1. 在 Live 页开着摄像头（会话 id 读
   `document.querySelector('[data-role="slv-root"]').dataset.slvSession`）；
2. 每 100–200 ms 轮询 `grasp?name=X`（用 `available_names` 核对名称是否打错）；
3. `ready=true` 后，再同步发一帧 `/live/api/frame` 并用其响应的框（最新鲜，
   滞后 ≤1 帧）触发抓取动作；
4. 抓取前相机与货架必须固定；`reason` 字段会告诉你当前为什么还没就绪
   （未稳定 / 场景有变化 / 摄像头没开）。

> ⚠️ `box`/`point` 是**发送图像像素**里的 2D 坐标。机械臂还需要相机内参 +
> 手眼标定把它们换算成 3D 位姿（有深度相机时取框内中心点深度更稳）。

### 5.4 Tab 4 · Catalog：商品库管理

**调用链**：

```
前端 JS (catalog_web.py, gr.HTML 注入)
  ├─► GET  /catalog/api/list
  │     └─► catalog_api.list_catalog → 每个 SKU 一行 + 照片 URL 列表（/catalog/photos/<文件名>）
  │        （照片目录 data/crops 用 Starlette StaticFiles 挂载为 /catalog/photos）
  └─► POST /catalog/api/delete {sku_id}
        └─► catalog_api.delete_sku → pipeline.delete_product(sku_id)
              ├─► VectorStore.remove(sku_id)   # 删掉该 SKU 的全部向量
              ├─► SQLite DELETE products       # 删元数据行
              └─► 删 data/crops/skuNNNN_*.jpg  # 删参考照片文件
```

纯 JSON 进出的 handler（`catalog_api.py`）不依赖任何框架，方便单独测试；
`app.py` 负责把它们包成 FastAPI 路由。

### 5.5 app.py 的两个「不写就会踩的坑」

1. **路由要用 `_app=` 传入**：`demo.launch()` 内部会重建 Gradio 的 FastAPI
   app，预先挂在 `demo.app` 上的自定义路由会被丢掉。所以 `build_app()`
   新建 `gradio.routes.App()`、挂好全部路由，再以 `_app=fastapi_app` 启动。
2. **重活必须 `asyncio.to_thread`**：YOLO/CLIP 是同步阻塞调用，不挪线程
   会卡死 Gradio 事件循环（整个 UI 无响应）。

---

## 6. 数据流向与持久化

注册和识别共用同一个向量空间（同一个 CLIP 模型编码），这是整条链路成立的根本：

```
          注册（一次性）                          识别（每次查询）
  ┌────────────────────┐              ┌──────────────────────────┐
  │ 参考照片 ×N         │              │ 货架照片                   │
  │      ▼              │              │ YOLO → Box ×N → crops ×N │
  │ CLIP.encode_image   │              │      ▼                   │
  │      ▼              │              │ CLIP.encode_image（同一模型）│
  │ (N, 512) 归一化向量  │   同一个      │      ▼                   │
  └──────┬─────────────┘   向量空间 ──► │ 余弦相似度 vs 库内全部向量  │
         │                             └────────────┬─────────────┘
         ▼                                          ▼
   ┌─────────────────────────── data/ 目录 ──────────────────────┐
   │ vectors.npz     {vectors: (N,512) float32, ids: (N,) int64} │
   │                   ↑ N 是参考照片总数（不是 SKU 数），ids 记录每个向量属于哪个 sku_id │
   │ catalog.sqlite  products(sku_id, name, barcode, price,      │
   │                   category, n_refs, created_at)             │
   │ crops/          sku0007_20260811113440_0.jpg …（展示用照片）  │
   └─────────────────────────────────────────────────────────────┘
```

- 每次 add/delete 后 `vectors.npz` 整表重写（几千个向量的规模，毫秒级）。
- 换 CLIP 模型 = 换了向量空间 → 旧向量全部失效，**需要重新注册所有商品**
  （`VectorStore._load` 以文件里存的维度为准，防止维度不一致时静默错配）。

---

## 7. 检测器选型（直接决定「能不能检测到商品」）

用 `DETECTOR` 环境变量切换后端（`shelf_demo/detector.py` 内部分发）：

| `DETECTOR` | 模型 | 原理 | 框质量 | 何时用 |
|-----------|------|------|--------|--------|
| `custom`（**默认**，需权重存在） | 本项目在 SKU-110K 训练的 YOLOv8n（单类别 "product"） | 有监督微调 | ★ 又多又紧 | **本仓库已附带权重，开箱即用** |
| `world`（无 custom 权重时的默认） | YOLO-World (`yolov8s-worldv2.pt`) | 开放词表，文字提示词驱动，**零训练** | 偏松，爱把整排/整盒框在一起 | 想立刻跑通但没有训练权重时 |
| `coco` | 普通 YOLOv8n (`yolov8n.pt`) | COCO 80 类预训练 | 对零食/包装**几乎检不到** | 只是先把 UI 跑起来 |

**默认逻辑**（`config.py`）：`weights/sku110k_best.pt` 存在 → 自动 `custom`；
否则回退 `world`。

**为什么默认参数随检测器变化**（这是"检测不到商品"的第一大坑）：

| 参数 | `custom` | `world`/`coco` | 原因 |
|------|----------|----------------|------|
| `DETECT_CONF` | 0.2 | 0.05 | 训练过的模型置信度校准良好；YOLO-World 零样本分数普遍偏低，阈值高了直接 0 框 |
| `DETECT_IOU` | 0.3 | 0.5 | 单类训练模型会在同一商品上叠出偏移半格的重复框（IoU≈0.4），0.5 去不掉，0.3 可以，且不会误并真正相邻的商品 |
| `DETECT_IMGSZ` | 1280 | 1280 | 4K 货架照片压到 640 后商品太小；密集货架建议 1920 |

**已验证的对比**（`python scripts/compare_detectors.py data/shelf.jpg`，
输出 `data/compare_custom.jpg` / `data/compare_world.jpg`）：同一张 4K 货架图，
custom 检出 **61** 个紧贴单品边框（中位 200×148 px），YOLO-World 只有 **21**
个松散大框（中位 412×333 px，经常框住整排商品）。

YOLO-World 提示词可用 `YOLO_WORLD_PROMPTS` 自定义（逗号分隔），默认：
`product,package,snack bag,pouch,box,bottle,can,carton,chocolate bar,bag of chips,candy`。

---

## 8. 如何训练自己的检测器：SKU-110K 完整指南

> 目标：得到一个**单类别、密集货架上框又多又紧**的商品检测器，
> 替代框偏松的零样本 YOLO-World。

### 8.1 为什么是 SKU-110K 数据集？

- **SKU-110K** 是零售货架密集目标检测数据集：约 1.17 万张货架图
  （本仓库 `datasets/SKU-110K/`：train 8219 / val 588 / test 2936），
  **11 万+ 个标注框**，图像 3024×3024。
- 它只有一个类别 `object`（= 商品），**恰好就是我们要的**：检测器只学
  「哪里有商品」，商品是谁交给后面的检索管线。
- 标注是 CSV（`image_name,x1,y1,x2,y2,class,image_width,image_height`），
  Ultralytics 用 `data="SKU-110K.yaml"` 引用它。

### 8.2 训练命令

```bash
# 完整质量训练（推荐在 CUDA 机器上；M 系列 Mac 用 mps 也可以，慢一些）
python scripts/train_sku110k.py --model yolov8n.pt --epochs 50

# 快速 demo 级训练（本仓库现有权重就是这样练的：15% 数据子集，~1.2 小时 @ M1 MPS）
python scripts/train_sku110k.py --model yolov8n.pt --epochs 15 --fraction 0.15 --batch 8
```

脚本参数（`scripts/train_sku110k.py`）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `yolov8n.pt` | 基础权重（n 最快，s/m 精度更高）。首次会自动下载 |
| `--epochs` | 15 | 训练轮数。全量数据建议 50+ |
| `--imgsz` | 640 | 训练分辨率（推理时再用 `DETECT_IMGSZ=1280/1920`） |
| `--batch` | 16 | 显存/MPS 内存不足就调小（M1 实测 8 稳妥） |
| `--fraction` | 1.0 | 每轮只用训练集的比例；0.15 = 快速验证 |
| `--device` | auto | `cuda` / `mps` / `cpu` |
| `--patience` | 10 | 早停：10 轮没有提升就停 |

**首次运行** Ultralytics 会自动下载 SKU-110K 数据集（**约 13 GB**）到
`datasets/`，请耐心等待；之后重复训练直接用本地缓存。

### 8.3 训练过程中会发生什么

所有产物在 `runs/detect/sku110k/`：

```
runs/detect/sku110k/
├── results.csv        # 每轮指标（loss、P、R、mAP50、mAP50-95）
├── results.png        # loss/精度曲线
├── BoxPR_curve.png    # 精度-召回曲线
├── confusion_matrix*.png
├── train_batch0.jpg   # 训练采样可视化（看数据增广效果）
├── val_batch0_pred.jpg# 验证集预测可视化（肉眼看框质量）
└── weights/
    ├── best.pt        # ★ 验证集上最好的权重（要用的就是它）
    └── last.pt        # 最后一轮权重
```

**怎么读指标**（新人版）：

- **P（precision 精确率）**：检测出的框里有多少是对的。0.85 = 检 100 个框，85 个准。
- **R（recall 召回率）**：真实商品里有多少被检到。0.76 = 漏了 24%。
- **mAP@50**：IoU 阈值 0.5 下的平均精度，密集检测的"总分"。>0.8 说明框又多又基本压得住商品。
- **mAP@50-95**：IoU 0.5~0.9 全档位的平均，更苛刻地衡量**框的紧致度**。
  本仓库现有权重是 0.473——还能提升（见 8.5）。

### 8.4 本仓库已完成的一次训练（可复现）

| 项 | 值 |
|----|----|
| 基础模型 | yolov8n.pt（预训练初始化） |
| 数据 | SKU-110K，`fraction=0.15`（每轮用 15% 训练集） |
| 超参 | 15 epochs, imgsz 640, batch 8, patience 10 |
| 硬件 | M1 MPS，约 1.2 小时 |
| 结果 | **P = 0.852 / R = 0.761 / mAP@50 = 0.818 / mAP@50-95 = 0.473** |
| 权重 | 已复制到 `weights/sku110k_best.pt`，app 自动启用 |

### 8.5 训练完怎么部署 + 怎么提质

```bash
# 1) 把最优权重放到标准位置（app 检测到它会自动切到 custom 后端）
cp runs/detect/sku110k/weights/best.pt weights/sku110k_best.pt

# 2) 启动并肉眼验证
python app.py

# 3) 量化验证：和 YOLO-World 对比框质量
python scripts/compare_detectors.py data/shelf.jpg
# 期望：custom 框数更多、中位框尺寸更小（更贴单品）
```

如果不想自动启用，也可以显式指定：

```bash
export DETECTOR=custom
export YOLO_WEIGHTS=runs/detect/sku110k/weights/best.pt
python app.py
```

**想再提质，按性价比排序**：

1. **全量数据 + 更多轮**：`--fraction 1.0 --epochs 50`（最直接的收益，R 和 mAP 都会上来）；
2. **换更大的模型**：`--model yolov8s.pt`（甚至 m），换框紧致度（mAP@50-95）；
3. **换 CUDA 机器**：M1 MPS 上 batch 只能开 8，CUDA 上 batch 16+ 训练更快更稳；
4. **推理侧**：4K 密集货架把 `DETECT_IMGSZ` 提到 1920（注意帧率下降）。

> 注意：检测器只决定「框的质量」，**不决定认不认得对**——认不认得对由
> CLIP + 商品库决定（见 §11 生产化建议）。

---

## 9. 调试指南：出问题了先看哪里

### 9.1 「No products detected」——一定是检测的问题

"检测不到" 和 "认不出" 是两回事，先用 `debug_detect.py` 分阶段定位：

```bash
python scripts/debug_detect.py data/shelf.jpg                # 默认检测器
python scripts/debug_detect.py data/shelf.jpg --conf 0.05    # 降低阈值多检点
python scripts/debug_detect.py data/shelf.jpg --detector coco  # 对比（通常 0 框，证明是类别问题）
python scripts/debug_detect.py data/shelf.jpg --imgsz 1920   # 4K 密集货架提高分辨率
```

它会：① 打印检到几个框（0 个 → 检测器问题，按提示调 imgsz/conf/训练 custom）；
② 把**所有**原始检测框画到 `data/debug_boxes.jpg`（直接看检测器"看到"了什么）；
③ 若商品库非空，打印每个框的最佳匹配 SKU + 相似度，并告诉你 `MATCH_THRESHOLD`
该调到多少（检测正常但全 Unknown → 检索/阈值/参考图质量问题，不是检测问题）。

### 9.2 「一个商品出多个框」

训练好的单类模型会对同一商品打偏移半格的重复框（IoU≈0.4）。默认
`DETECT_IOU=0.3` 已处理；仍有残留时用 `sweep_detect.py` 扫描
imgsz × conf × NMS-iou 组合，并输出三个货架区域的放大对比图到 `data/sweep/`：

```bash
python scripts/sweep_detect.py data/shelf.jpg
```

### 9.3 其他常用命令

```bash
python scripts/smoke_test.py                 # 检索闭环自检（红/蓝两个假 SKU）
python scripts/compare_detectors.py data/shelf.jpg   # 两种检测器框质量对比
```

---

## 10. 配置项（全部环境变量，`shelf_demo/config.py`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `DETECTOR` | 有 `weights/sku110k_best.pt` 时 `custom`，否则 `world` | `custom` / `world` / `coco`（§7） |
| `DETECT_IMGSZ` | `1280` | 检测推理分辨率；4K 密集货架用 `1920` |
| `DETECT_CONF` | `custom` 0.2，否则 0.05 | 检测置信度阈值（§7 表格解释了差异原因） |
| `DETECT_IOU` | `custom` 0.3，否则 0.5 | NMS 去重阈值 |
| `DETECT_MAX_DET` | `1000` | 单图最大框数（货架很密集） |
| `DETECT_CLASS_AGNOSTIC` | `1` | 跨类 NMS（YOLO-World 多提示词重复框必需） |
| `YOLO_WEIGHTS` | `weights/sku110k_best.pt`（存在时）否则 `yolov8n.pt` | `coco`/`custom` 权重路径 |
| `YOLO_WORLD_WEIGHTS` | `yolov8s-worldv2.pt` | YOLO-World 权重路径 |
| `YOLO_WORLD_PROMPTS` | `product,package,...` | 开放词表提示词（逗号分隔） |
| `CLIP_MODEL` / `CLIP_PRETRAINED` | `ViT-B-16` / `laion2b_s34b_b88k` | embedding 模型。**换模型必须重新注册所有商品**（向量空间变了）。想更强可试 `ViT-B-16-SigLIP` + `webli` |
| `EMBED_DIM` | `512` | 向量维度（实际以加载的 CLIP 模型为准自动校正） |
| `MATCH_THRESHOLD` | `0.65` | 余弦相似度低于此值 → Unknown（按 ViT-B-16 校准） |
| `SEARCH_TOPK` | `5` | 每个框返回 top-k 个候选 SKU |
| `SHELF_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `SHELF_DATA_DIR` | `./data` | 向量库/SQLite/照片的根目录 |

Live 页内置常量（`shelf_demo/live.py`，暂不走 env）：track 关联 `MATCH_IOU=0.5`、
track 丢失容忍 `MAX_MISSES=4` 帧、会话空闲回收 `SESSION_TTL_S=120s`、
前端可选 imgsz 范围 480–1536；抓取就绪 `GRASP_WINDOW=5` 帧 / `GRASP_IOU=0.9` /
`GRASP_MAX_AGE_MS=1000` ms（含义见 §5.3 机械臂集成）。

---

## 11. 关键设计决策（为什么这么写）

### 11.1 向量检索为什么用 NumPy 而不是 FAISS？

macOS 上 `faiss-cpu` 和 `torch` 各自打包了一份 OpenMP 运行时（libomp），
同一进程同时加载会 `OMP: Error #15` 报错、在 Gradio 多线程服务下**偶发段错误**。
`KMP_DUPLICATE_LIB_OK=TRUE` 只是压制报错，官方明确说"可能崩溃或静默算错"，不可靠。

Demo 规模（几千个 512 维向量）的最近邻检索就是一次矩阵乘法，NumPy 亚毫秒完成、
零原生依赖冲突。`VectorStore` 的 API 特意做成 `add / search / delete` 的
标准向量库形态——上量级（10 万+ SKU）时直接换成 FAISS / Milvus / Qdrant
即可，上层代码不动。

### 11.2 为什么 Live 页和 Catalog 页是自研前端？

- **Live**：Gradio 6 的 `gr.Image(streaming=True)` 内部状态机不可控
  （停止按钮不复位、会话不结束，源码级确认应用层修不了）。自研后：
  前端 canvas 独立 15 FPS 重绘（检测慢画面不卡）、generation 计数丢弃
  停止后的迟到响应、摄像头热切换不断流。
- **Catalog**：Gradio 原生表格放不了「照片缩略图 + 行内删除按钮」的交互，
  用 `gr.HTML` 注入纯 JS 表格，数据走两条 JSON 路由。
- 两者共同的坑：Gradio 的 `gr.HTML` 模板字符串里**不能出现 `${`**
  （会被当成模板插值槽），所以 JS 一律用字符串拼接 / DOM API。

### 11.3 Live 会话为什么用两把锁？

`_map_lock` 保护会话 dict 本身；`model_lock` 在整个帧处理（改 track + 跑
MPS 推理）期间持有——MPS 不接受并发推理。两把锁**绝不嵌套**，否则
`handle_reset` 里会自死锁（初版踩过，faulthandler 定位，见设计文档）。

---

## 12. 从 Demo 到生产：还需要做什么

这个 Demo 用**零样本 CLIP**，链路完整但细粒度 SKU 区分有限。生产系统的进阶方向：

1. **微调 embedding 模型**：零样本 CLIP 在零售数据上 top-1 只有 ~40%，
   用零售图微调后可达 ~89–92%。可换 **SigLIP / DINOv2** 作 backbone。
2. **细粒度消歧**：同款不同规格（500ml vs 750ml）最容易翻车，
   生产常加**第二阶段 reranker / 关键点匹配 / OCR（读包装文字）**。
3. **向量库升级**：10 万+ SKU 换 FAISS / Milvus / Qdrant 做 ANN
   （`VectorStore` 接口已对齐，直接替换）。
4. **自动造标注**：用 Grounding DINO + Autodistill 零样本标注货架图，
   再蒸馏训练小 YOLO，省人工。
5. **VLM 兜底**：检索置信度低的 crop 送生成式 VLM（Qwen-VL / GPT-4o）二次确认。

> 生成式 VLM 目前**不是**货架识别的线上主流（成本高、延迟大、SKU 级会幻觉），
> 主要用在"造数据"和低置信度兜底。

---

## 12.5 Eye-in-hand 机械臂抓取（RGB-D 扩展，进行中）

目标形态：机器狗驮机械臂在货架前抓/理商品，深度相机（RealSense）装在腕部。
设计文档：**`docs/plans/2026-08-27-eye-in-hand-design.md`**。
识别层（YOLO+CLIP、2D 稳定窗口）完全不动，新增几何层：

| 模块 | 职责 |
|------|------|
| `shelf_demo/transforms.py` | SE3 数学：变换链 `P_base = T_base_ee·T_ee_cam·P_cam` |
| `shelf_demo/camera.py` | RealSense 采帧（深度对齐 RGB、转米、读内参） |
| `shelf_demo/pose3d.py` | 2D 框 + 深度 → 相机系 3D 目标点（中值滤波） |
| `shelf_demo/calibration.py` | 手眼标定：纯 numpy Park 解法（cv2≥5 删了 calibrateHandEye） |
| `shelf_demo/rgbd_live.py` | `RGBDGraspSession.grasp3d(name)` → 米制坐标 |
| `shelf_demo/robot.py` | `ArmBase` 接口 + `MockArm` |

```bash
python scripts/test_transforms.py            # SE3 数学自检（秒级，无硬件）
python scripts/test_rgbd_grasp.py            # 全链路合成数据测试（同上）
python scripts/calibrate_handeye.py simulate # 手眼解算自检（无硬件）
python scripts/rs_live.py --target "可乐"     # 上机回路：实时打印目标 3D
# 臂到货后：collect（采 10+ 组）→ solve → data/handeye.json，
# rgbd_live 的 point_base_m 随即可用
```

---

## 13. 开发与测试

```bash
# 抓取就绪接口（机械臂）单测：假 pipeline 驱动 LiveRecognizer，无需模型，秒级
python scripts/test_grasp.py

# 开发依赖（app 运行不需要）
pip install playwright && playwright install chromium-headless-shell

# Live 页端到端：Playwright 假摄像头，验证开始/停止/重识别/帧流连续性/摄像头热切换
python scripts/e2e_live_ui.py

# Catalog 页端到端：真实注册一个测试 SKU → 验证表格渲染/缩略图/删除（含照片文件清理）
python scripts/e2e_catalog_ui.py
```

设计演进记录见 `docs/plans/2026-08-12-realtime-video-annotation-design.md`
（含实测耗时表、Gradio 流媒体问题的源码级诊断）。
