# 实时视频标注设计（2026-08-12）

## 需求
在现有"上传照片 / 相机拍照识别"之外，为货架识别 demo 增加**摄像头实时视频标注**。

## 关键约束（M1 MPS 实测）
| 阶段 | 耗时 @960 | 结论 |
|------|-----------|------|
| YOLO 检测 | ~45 ms/帧（≈23 FPS） | 可以逐帧跑 |
| CLIP embedding | 1 crop ≈156 ms，40 crop ≈1.35 s | **逐帧全量不现实** |
| 向量检索 | 亚毫秒（NumPy） | 忽略 |

瓶颈在 CLIP，因此核心设计是"**检测每帧跑、识别只跑新目标**"。

## 架构
1. **`shelf_demo/pipeline.py`**：把 recognize 拆出 `match_boxes(image, boxes, threshold)`
   （crop→embed→检索），照片 Tab 与实时 Tab 共用同一识别逻辑。
2. **`shelf_demo/live.py`（新增）**：`LiveRecognizer`，每用户一份：
   - 每帧检测框（imgsz 默认 960，与照片 Tab 解耦）；
   - 贪婪 IoU≥0.5 关联到既有 track，命中的框复用其 SKU 标签；
   - 仅无关联的新框批量过 `match_boxes`；track 连续 4 帧未命中即丢弃；
   - 返回 `LiveResult(recognitions, changed, n_new)`，`changed=false` 时
     UI 侧不刷新摘要文本。
3. **app.py 新 Tab「3 · Live (webcam)」**：
   `gr.Image(sources=["webcam"], streaming=True)` + `.stream(...)`，
   `stream_every=0.05` + `trigger_mode="always_last"`（积压帧直接丢，
   永远处理最新帧，防止延迟雪球）。UI 显示 FPS · 框数 · 匹配数，
   「↺ Re-recognize scene」清空跟踪器强制全量重识别。

## 状态管理
- 跟踪器放 `gr.State`（每会话独立）；检测/匹配参数（conf/threshold/imgsz）
  变化时自动重建会话，避免旧缓存标签串味。

## 已验证结果（headless）
- 新场景首帧：31 目标一次批量嵌入（~1.6 s），之后静态帧 61 ms/帧（≈16 FPS）；
- 轻微抖动帧最多补嵌 4 个边缘框；`reset()` 后全量重识别；
- 参数变更重建、空帧、UI 启动（端口 7865 冒烟）全部通过。

## 已知取舍
- 标签在 track 生命周期内固定：商品被移走后新出现的同款没问题，
  但**库内商品被删除/重注册后需点 Re-recognize** 刷新；
- 浏览器摄像头要求安全上下文：localhost/127.0.0.1 可用；
  局域网 IP 访问需 HTTPS 或 `share=True`。

---

## 增补：前端替换为自研 HTML/JS（2026-08-12，最终落地版）

初版实现按计划用了 `gr.Image(sources=["webcam"], streaming=True)`，
实测暴露两个**无法在应用层修复**的 Gradio 6 内部问题（经源码级诊断确认）：

1. **录制/停止按钮显示状态不由 app 控制**：按钮文案由前端 `loading_status.stream_state`
   推流驱动；stream session 在服务端按 "awake 事件 + 30s 窗口" 保活，
   app 侧点击停止后按钮仍长时间显示 "Stop"。
2. 停止后浏览器仍偶发继续收帧（会话未真正终止）。

**替换方案（已实现）**：

- Live Tab 改为 `gr.HTML` 自研组件（`shelf_demo/live_web.py`：HTML/CSS/JS 模板，
  通过 `js_on_load` 注入；注意模板字符串里**不能出现 `${`**，会触发 Gradio 模板插值）。
- 标注接口走两条 FastAPI 路由（`register_live_routes()`，经 `launch(_app=App)` 挂载，
  因为 `App.create_app` 会丢弃预先加在 `demo.app` 上的路由）：
  - `POST /live/api/frame` `{uuid, image(data-url jpeg), imgsz, conf, threshold}`
    → `{boxes[], n_objects, n_matched, n_new, ms, fps, summary?}`
  - `POST /live/api/reset` `{uuid}` → 清空该会话的跟踪缓存。
- 覆盖框**由浏览器**在 `<canvas>` 上绘制（服务端不再回传绘制后的 JPEG），
  重绘循环 15 次/秒，服务端检测慢一点也依然流畅。
- 会话管理 `LiveSessionManager`：按 uuid 一份 `LiveRecognizer`，空闲 120s 回收；
  `_map_lock`（会话表）与 `model_lock`（模型串行，MPS 不喜并发调用）分离，
  避免同一线程内嵌套加锁自死锁（初版实现踩过此坑，faulthandler 定位为
  `handle_reset` 内 `SESSIONS.lock` 与 `reset()` 的双重获取）。
- 前端附带 "generation 计数"，停止后晚到的在途响应被丢弃，
  保证「停止」状态不会被覆盖——按钮状态始终与真实状态一致。

验证：`scripts/e2e_live_ui.py`（Playwright 假摄像头）端到端断言
开始/停止/重新识别/帧流连续性，全部通过。
