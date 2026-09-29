# リアルタイム映像注釈機能 — 設計書

**日付**: 2026-08-12
**状態**: 完了（v3 — 性能最適化版 + ストリーミング問題のソースレベル診断）
**対象**: `yolo/`（supermarket-shelf-product-recognition）プロジェクトの「商品認識」Web アプリ

---

## 1. 背景と問題

### 1.1 現状

プロジェクトには 4 つの Gradio タブが既に存在：

| タブ | 実装 | 状態 |
|------|------|------|
| Register | `app.py` + `shelf_demo/pipeline.py` | 動作確認済み |
| Recognize | 同上（画像アップロード） | 動作確認済み |
| **Live** | `app.py` 内の `live_html` / `handle_frame` | スケルトンのみ（HTML、フレームハンドラは未接続） |
| Catalog | `shelf_demo/catalog_api.py` + `catalog_web.py` | 動作確認済み |

`shelf_demo/live.py` に既に実装済み・**検証済み**のコアロジックがある：

- `LiveRecognizer`：`Tracker` + 追跡ターゲットごとの認識キャッシュ + `match_boxes` の呼び出し
- `LiveSession`：uuid キーのセッション管理、120 秒 TTL、`_map_lock`（dict 保護）+ `model_lock`（モデル呼び出し直列化）
- `handle_frame(payload, pipeline) → (json_str, changed)`：JPEG base64 受信 → 認識 → JSON 返却（`boxes`/`n_matched`/`n_new`/`changed`/`ms`）

**検証済み**（`scripts/e2e_live_ui.py` の実測、`scripts/smoke_test.py`）：

- `recognize()` のフルパイプラインは 3–7 秒/回（YOLO 検出 + 各ボックスの CLIP + 検索）
- `LiveRecognizer.process()` の 2 回連続呼び出し：2 回目は `n_new=0`（キャッシュが機能、CLIP を再実行しない）
- `handle_frame` の JSON 形式が正確、`ms` フィールドが認識の所要時間を報告

### 1.2 タブ 3 の現状ギャップ

- フロントエンドは `live_web.py` 内の 300 行の HTML/JS（canvas 描画 + `fetch` フレーム送信、**Gradio なし**）
- 後端は `catalog_api.py` のような純 JSON ハンドラを**持っていない**：`handle_frame` は Gradio の `gr.Image(streaming=True)` + `gr.Textbox` のイベントモデルに紐づいた形で書かれている
- `app.py` は Live タブに `live_html` のみを提供し、`/live/api/frame` POST ルートを**公開していない**

### 1.3 なぜ `gr.Image(streaming=True)` を使わないか（ソースレベルの診断）

当初は Gradio 標準のストリーミング画像コンポーネントを使う案を評価したが、Gradio 6.19.2 のソースレベルの診断により**アプリ層からは制御不能**であることを確認した：

- `components/image.py` の `postprocess_data`：録画状態の判定は `self.stream_state["streaming"]`（内部状態）に依存。フロント JS（`streaming_image.js`）は「停止」をクリックすると `stop()` を呼び、**サーバ側の state をリセットする**が——
- 停止ボタンが再度「Start」に戻る条件（`update_streaming_state` → `stream_state["should_stop"]`）は、**サーバがストリーミングを「正常終了」したときのみ**に真になる。アプリが `stop()` を受けても、イベントは返ってこないため `should_stop` は永遠に False のまま。
- 結論：ストリームが「正常終了」しない限り（Gradio のストリームは正常終了しない）、**ボタンは常に "Stop" に固定される**。アプリ層では修正不能。

これが Tab 3 で自作フロントエンド（純 HTML/JS + FastAPI ルート）を採用した理由：
開始/停止の状態は常に真実を反映し、停止は確実にサーバのセッションを破棄し、フレーム送信の周波数/タイムアウト/描画周波数をすべて制御できる。

---

## 2. 目標

1. **ブラウザのカメラから**リアルタイムに棚映像を取得し、各フレームで商品を検出・認識し、映像の上に bounding box + 商品名をオーバーレイ表示
2. **レイテンシ目標**：M1 (MPS) 上で 15–20 FPS の推論 + 描画（YOLO @640px）
3. **既存の認識パイプラインを再利用**：`pipeline.recognize()` を変更しない（その内部の `detect + match_boxes` を呼ぶ）
4. **セッション分離**：複数のブラウザセッションは相互に干渉しない（各セッションに独立した `LiveRecognizer`）
5. **ゼロ新規依存**：Gradio 標準コンポーネント + 標準ライブラリのみ、追加パッケージなし
6. **UI パラメータ調整**：フレームレート、検出解像度 (imgsz)、conf、閾値を UI から変更可能

### 非目標

- 多カメラ / RTSP / OBSBOT 直接接続（既存 `scripts/rs_live.py` の機能と重複、将来の拡張点）
- GPU 推論のマルチスレッド化 / バッチ化（デモ規模で不要）
- モバイルブラウザの最適化（カメラアクセスはデスクトップ Chromium / Safari を主とする）

---

## 3. アーキテクチャ

### 3.1 全体データフロー

```
┌─────────────────────────── ブラウザ ───────────────────────────┐
│  <video> ← getUserMedia (カメラ)                               │
│  JS タイマー (33ms) → canvas.drawImage(video)                  │
│      → 縮小 (≤1280px, JPEG q=0.7) → POST /live/api/frame      │
│  サーバの JSON 返却 → canvas に box + label を描画              │
│  (動画自体はローカルで描画し、サーバが描画済みフレームを返さない) │
└──────────────────────────────┬─────────────────────────────────┘
                               │ JSON: {uuid, image(b64), imgsz, conf, threshold}
                               ▼
┌────────────────────── FastAPI (asyncio.to_thread) ─────────────┐
│  handle_frame(payload, pipeline)   ← shelf_demo/live.py 既存   │
│    └─► LiveSession.process(uuid)                               │
│          └─► LiveRecognizer.process(img)                       │
│                ├─► detector.detect()        (YOLO, 全フレーム) │
│                ├─► 追跡 (ByteTrack)          (全フレーム)      │
│                ├─► 認識キャッシュ参照         (既存ターゲット)  │
│                └─► match_boxes(新規のみ)     (CLIP+検索)       │
└────────────────────────────────────────────────────────────────┘
```

**重要**：映像フレームは**ブラウザでローカル描画**され、サーバは認識 JSON（ボックス座標 + ラベル）を返すだけ。
サーバが描画済み JPEG を返却する方式ではないため、ネットワーク負荷が極小（1 フレームの JSON 数 KB vs 描画済み JPEG 100–500 KB）、
描画フレームレートはブラウザで独立制御（30 FPS）でき、推論が 15 FPS に落ちても動画自体は滑らかに保てる。

### 3.2 各レイヤーの責務

| レイヤー | ファイル | 責務 | 新規/既存 |
|------|------|------|------|
| フロントエンド | `shelf_demo/live_web.py` | HTML/JS：canvas、getUserMedia、フレーム送信、描画 | **新規** |
| ルーティング | `app.py`（`build_app` に追記） | `/live` ルート + `/live/api/frame` + `/live/api/reset` | **新規** |
| 後端 API | `shelf_demo/catalog_api.py`（追記）または新規 `live_api.py` | JSON ハンドラ（純関数、テストしやすく） | **新規** |
| セッション/認識 | `shelf_demo/live.py` | 既存の `LiveSession` / `LiveRecognizer` / `handle_frame` | 既存（**変更なし**） |
| パイプライン | `shelf_demo/pipeline.py` | `recognize` / `match_boxes` | 既存（**変更なし**） |

> **設計判断**：`shelf_demo/live.py` は既に動作確認済みのため**変更しない**。
> フロントエンドと API ルートは既存の Catalog パターン（純 JSON ハンドラ + FastAPI 注入）に倣い、
> 既存ロジックに「フレームの送受信」の薄いラッパーを 1 つ加えるだけにする。

### 3.3 API 契約

```
POST /live/api/frame
Request:  {"uuid": "...", "image": "data:image/jpeg;base64,...",
           "imgsz": 640, "conf": 0.2, "threshold": 0.65}
Response: {"boxes": [{"xyxy":[x1,y1,x2,y2], "label":"Coke", "match":0.81}, ...],
           "n_objects": 7, "n_matched": 5, "n_new": 0, "changed": false, "ms": 68.3}
```

```
POST /live/api/reset   # 「再認識」ボタン → そのセッションの認識キャッシュをクリア
Request:  {"uuid": "..."}
Response: {"ok": true}
```

- `uuid`：ページロード時に `crypto.randomUUID()` で生成し、JS 内で保持
- `changed`：ラベル変化の有無（フロントエンドはサマリーテキストの更新頻度を制御するために使用）
- `ms`：このフレームの認識の所要時間（フロントエンドの FPS 表示に使用）

### 3.4 セッションライフサイクル

```
ページロード  → JS: uuid = crypto.randomUUID()
[Start]       → JS: getUserMedia() でカメラ起動 + タイマー開始
                → 毎フレーム POST /live/api/frame
                → サーバ: SESSIONS.get_or_create(uuid) → LiveRecognizer
[Stop]        → JS: タイマー停止 + stream.stop() + track.stop()
                → POST /live/api/reset（オプション：セッションキャッシュクリア）
                → サーバ: セッションは 120s 放置後に自動回収（既存 SESSION_TTL_S）
```

---

## 4. 詳細設計

### 4.1 フロントエンド（`live_web.py`）

**HTML 構造**（既存 `live_html` スケルトンの拡張）：

```
┌────────────────────────────────────────────────────────┐
│  [Start camera]  [Stop]  [Re-recognize]                │
│  Status: ● Running | FPS: 17.2 | 5/7 matched | 68 ms  │
├────────────────────────────────────────────────────────┤
│                                                        │
│              <video> (カメラ映像)                       │
│              <canvas> (box + label オーバーレイ)        │
│                                                        │
└────────────────────────────────────────────────────────┘
│  Controls:                                             │
│  Resolution [640▾]   Conf [0.20]  Threshold [0.65]    │
└────────────────────────────────────────────────────────┘
```

**JS コアロジック**（`live_web.py` 内の `LIVE_JS` 定数、既存 `live_html` と同じ方式）：

```javascript
// 1) 開始
async function start() {
  uuid = crypto.randomUUID();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } }
  });
  video.srcObject = stream;
  await video.play();
  tick();  // タイマー開始
}

// 2) 毎フレーム送信 (33ms タイマー ≈ 30 FPS 上限)
async function tick() {
  if (!running || sending) { schedule(); return; }  // 1 回の送信がまだ返ってこなければスキップ
  sending = true;
  const b64 = grabFrame(video);   // canvas → JPEG base64 (quality 0.7, ≤1280px)
  const res = await fetch('/live/api/frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({uuid, image: b64,
                          imgsz: sel_imgsz.value|0,
                          conf: +sl_conf.value, threshold: +sl_thr.value}),
    signal: AbortSignal.timeout(8000)
  });
  const data = await res.json();
  results = data;                 // 直近の結果を保存
  drawOverlay();                  // 即時再描画
  updateStatus(data);
  sending = false;
  schedule();
}

// 3) オーバーレイ描画：video フレーム + box/label（66ms タイマーで独立再描画 ≈ 15 FPS）
function drawOverlay() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  for (const b of results.boxes) {
    const [x1, y1, x2, y2] = b.xyxy;  // 送信画像の座標系
    // canvas と送信画像は同一スケール（getUserMedia で 1280px に縮小済み）
    ctx.strokeStyle = b.match >= threshold ? '#22c55e' : '#ef4444';
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    ctx.fillText(b.label, x1, y1 - 4);
  }
}

// 4) 停止
async function stop() {
  running = false;
  stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  await fetch('/live/api/reset', {method:'POST', body: JSON.stringify({uuid})});
}
```

**フレーム取得の詳細**：

- `grabFrame(video)`：オフスクリーンの canvas（`document.createElement('canvas')`）に video を描画し `toDataURL('image/jpeg', 0.7)`
- 解像度上限 1280px：ネットワーク帯域と推論速度のバランス（YOLO の imgsz は後端で別途制御）
- **バックプレッシャー制御**：`sending` フラグ——前回送信がまだ返ってこなければ、そのフレームはスキップ
  （推論が 15 FPS だとすると、30 FPS の取得は 1:2 で削減される。キューが無限に膨らむことを防止）
- タイムアウト：8 秒（初回フレームはモデルウォームアップで数秒かかることがある）

### 4.2 後端（`app.py` に追記）

既存 `build_app()` に 3 つのルートを追記（Catalog パターンと同じ）：

```python
# ── Live routes ──
fastapi_app.add_api_route(
    "/live",
    lambda: HTMLResponse(live_html),
    methods=["GET"],
)
fastapi_app.add_api_route(
    "/live/api/frame",
    lambda req: JSONResponse(asyncio.to_thread(handle_frame, req.body, pipeline)),
    methods=["POST"],
)
fastapi_app.add_api_route(
    "/live/api/reset",
    handle_live_reset,   # 純関数：SESSIONS.get(uuid)?.reset() → {"ok": True}
    methods=["POST"],
)
```

> `handle_frame` は**同期的・ブロッキング**（YOLO + CLIP の 50–300ms の計算を含む）ため、
> `asyncio.to_thread` でワーカースレッドに投げる必要があり、これは既存の `recognize_api` と同じパターン。
> `LiveSession.process` 内部の `model_lock` が複数リクエストのモデル呼び出し直列化を保証（MPS は並行推論を嫌う）。

### 4.3 UI コントロール

| コントロール | 型 | 値 | 影響先 |
|------|------|----|--------|
| Start / Stop | button | — | `getUserMedia` のライフサイクル |
| Re-recognize | button | — | `POST /live/api/reset`（認識キャッシュクリア、全ターゲットを再認識） |
| Resolution | dropdown | 480 / 640 / 960 / 1280 | `imgsz`（YOLO 推論サイズ） |
| Confidence | slider | 0.05 – 0.9 | `conf` |
| Threshold | slider | 0.50 – 0.80 | `MATCH_THRESHOLD`（このセッションの閾値） |
| FPS 表示 | statusbar | 実測 | サーバ返却の `ms` に基づく |

パラメータ変更は**次のフレームから即時有効**（`handle_frame` は `pipeline.recognize` を毎回現在の値で呼び出すため）、
ページリロードは不要。

### 4.4 ステータスバー

```
● Running | 17.2 FPS | 5/7 matched | 68 ms | imgsz=640 conf=0.20
```

- `17.2 FPS`：クライアントが直近 10 フレームの送信間隔から計算（ブラウザで得られた実際のエンドツーエンドフレームレート）
- `5/7 matched`：`n_matched` / `n_objects`
- `68 ms`：サーバ返却の `ms`（純粋な認識の所要時間、ネットワーク往復を含まない）

---

## 5. 性能見積もり（M1 / MPS）

| ステージ | 所要時間（実測） | 備考 |
|------|------|------|
| ユーザーのカメラフレーム取得 (30 FPS) | 33 ms | ブラウザ |
| フレーム送信 (≤1280px JPEG) | 5–20 ms | localhost |
| YOLO 検出 @640 | ~45 ms | 実測 22 FPS @960、@640 はより速い |
| 追跡 (ByteTrack) | <1 ms | 純 NumPy |
| 新規ターゲットの CLIP+検索 | ~200–500 ms / 新規 | 棚が静止している場合新規は毎フレーム 0–2 個 |
| **合計（棚静止時）** | **~50–100 ms** | 10–20 FPS 可能 |
| **合計（カメラパン時）** | **~300–800 ms** | 多数の新規ターゲット、自動的に低速 |

**結論**：棚が静止している場合 15 FPS 以上は安定達成可能；カメラを大きく振った場合、新規ターゲットの認識がボトルネックになり、
これは**認識キャッシュ（既存 `LiveRecognizer` の機能）の設計意図通り**——一度認識したターゲットは再認識しない。

> v2（完了）：`DETECTOR=custom` 切替 + 300 枚の商品クロップ（155 SKU）を登録した後の実測：
> 静止棚 ~61 ms/フレーム（≈16 FPS）、カメラをゆっくり振った場合 237–1004 ms（新規ターゲットが多いとき）、
> 静止フレームの 61 ms は推論のみの所要時間であり、ネットワーク往復（20–40 ms）は別途必要。
> `scripts/e2e_live_ui.py` の実測（Playwright 偽カメラ、imgsz 640、静止棚）：
> **61–97 ms/フレーム（10–16 FPS）**。

---

## 6. ファイル変更リスト

| ファイル | 操作 | 規模 | 内容 |
|------|------|------|------|
| `shelf_demo/live_web.py` | 新規 | ~300 行 | HTML/JS/CSS：canvas、getUserMedia、フレーム送信、描画、UI コントロール |
| `shelf_demo/live.py` | 修正 | ~10 行 | `handle_frame` は既に `imgsz/conf/threshold` を受け取っており、追加は不要；`LiveSession.reset` のみ確認/追記 |
| `app.py` | 修正 | ~20 行 | `/live` + `/live/api/frame` + `/live/api/reset` の 3 ルートを追加 |
| `scripts/e2e_live_ui.py` | 新規 | ~120 行 | Playwright 自動テスト：偽カメラ → フレーム送信 → JSON 検証 → スタート/停止/再認識 |
| `scripts/test_grasp.py` | 新規 | ~150 行 | `grasp` 準備インターフェースの単体テスト（偽 pipeline で `LiveRecognizer` を駆動、モデル不要、秒オーダー） |

**推計総量**：~600 行（テスト込み）

---

## 7. テスト計画

### 7.1 ユニットレベル

- `handle_frame` の JSON 契約：既存の `e2e_live_ui.py` に追加（`changed` / `ms` フィールドを検証）

### 7.2 E2E（Playwright、既存インフラの再利用）

```
Playwright 偽カメラ (canvas → webm → 1080p)
  → /live ページロード
  → [Start] クリック → getUserMedia 許可
  → 50 フレーム送信
  → 検証：
    ✓ JSON 形式が正確（boxes/n_objects/ms フィールド）
    ✓ ボックス座標は画像範囲内
    ✓ 静止偽映像下 n_new は 50 フレーム後に 0 に収束
    ✓ [Stop] クリック後にセッションが破棄される（2 回目の Start で n_new が再出現）
    ✓ [Re-recognize] 後に n_new が 1 回スパイクし、その後収束
```

既存の `scripts/e2e_catalog_ui.py`（Catalog の Playwright テスト）と同一パターンで、
`scripts/e2e_live_ui.py` を複製して改造すればよい。

### 7.3 手動受け入れチェックリスト

- [ ] 本物のカメラ（OBSBOT）で棚に向けて、枠が滑らかに移動する
- [ ] 棚の前に商品を 1 個追加：~3 フレーム後にラベルが出現し、その後は安定
- [ ] 商品を棚に戻す：ラベルが消失、残らない
- [ ] Resolution を 480 → 1280 に変更：FPS が低下するが、枠がより正確
- [ ] Threshold を 0.65 → 0.5 に下げる：Unknown が減少
- [ ] 2 つのブラウザウィンドウ：それぞれが独立したセッション（一方の操作が他方に影響しない）
- [ ] ページの再読み込み：uuid が再生成され、古いセッションは 120 秒後に自動回収

---

## 8. リグレッション防止

| 既存機能 | 影響 | 説明 |
|------|------|------|
| Register / Recognize / Catalog タブ | **なし** | 新規追加ルートは既存コードに触れない |
| `shelf_demo/live.py` | 最小 | `handle_frame` / `LiveRecognizer` / `LiveSession` には**変更なし** |
| `scripts/smoke_test.py` | **なし** | 引き続き `pipeline.recognize` を直接使用 |
| 既存 3 個の E2E テスト | **なし** | 新規 `e2e_live_ui.py` は独立ファイル |
| 起動時間 | +0 ms | Live は遅延初期化（最初のフレームでモデルは既にウォームアップ済み） |

---

## 9. 既知の制約

### 9.1 既知の制約

1. **ブラウザのカメラはセキュアコンテキスト必須**：`http://127.0.0.1:7860` はローカルホストに例外があり動作可能；
   LAN IP でアクセスする場合は HTTPS（`gradio launch(share=True)` または自己署名証明書）が必要
2. **iOS Safari**：`getUserMedia` は動作可能だが、バックグラウンド移動でカメラが中断される
   （iOS の制限、復帰時にユーザーの再許可が必要）
3. **複数 GPU / 複数プロセスの水平スケーリング**：デモ規模では非目標；将来の要件時は `LiveSession` を Redis に外に出し、
   複数の worker プロセスに分割可能（`uuid` は既にグローバルに一意）

### 9.2 将来拡張（今回は実施しない）

- **OBSBOT 直接接続**：既存 `scripts/rs_live.py`（pyrealsense2）はサーバでカメラを直接取得し、
  ブラウザへ WebSocket でストリーミング——「カメラを PC に直接接続」のシナリオに適している。Tab 3 と補完関係
- **3D 座標出力**：RealSense 深度カメラがあれば、`pose3d` により bounding box 中心を 3D ポーズに変換し、
  ロボットアームの grasp に使用可能（既存 `shelf_demo/pose3d.py`）
- **推論パイプラインの並列化**：YOLO と CLIP を別スレッド（別デバイス）に配置
- **フレーム間差のスキップ**：フレームの perceptual hash が直近 N フレームと同一なら認識をスキップ

---

## 10. マイルストーン

| ステップ | 内容 | 推定 | 完了基準 |
|------|------|------|---------|
| M1 | `live_web.py` フロントエンド（HTML/JS） | 2h | /live で手動開始/停止可能、動画表示可能 |
| M2 | `app.py` の API ルート 3 つ + `live.py` の微修正 | 1h | フレーム送信 → JSON 返却の閉ループ |
| M3 | Playwright E2E テスト | 1h | `e2e_live_ui.py` 全パス |
| M4 | 本物カメラの手動検証 + パフォーマンス調整 | 1h | 実測 15+ FPS、受け入れチェックリスト全項目 |

**総計**：~5–6 人時間、1 日以内で完了可能。

---

## 11. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|------|------|------|------|
| ブラウザの `getUserMedia` が権限ポップアップに拒否される | 高（初回） | 機能が使用不可 | UI に明確な権限説明、ローカルホスト URL を使用 |
| 初回フレームのモデルウォームアップで 1–3 秒の遅延 | 高 | 初回フレームが詰まる | 起動時に `app.py` は既にモデルをウォームアップ；フロントエンドの 8s タイムアウトが許容 |
| MPS の並行推論競合（複数セッション） | 中 | レイテンシ変動 | 既存 `model_lock` がモデル呼び出しを直列化 |
| 高解像度フレームの帯域逼迫 | 低（localhost） | フレームレート低下 | 1280px 上限 + JPEG q=0.7 + 1 送信のバックプレッシャー制御 |
| Gradio 6 の `HTMLResponse` エスケープ問題 | 低 | JS が壊れる | 既存 Catalog のパターンに倣い、`gr.HTML` を経由せず直接 Starlette の `HTMLResponse` を返す |

---

## 12. 決定事項記録（v3 追記）

1. **ストリーミングフロントエンドは自作**（§1.3）：`gr.Image(streaming=True)` は Gradio 6 の内部状態機械により、
   アプリ層から停止/リセット不能と確認。
2. **フレームレート戦略**：YOLO は毎フレーム実行、CLIP は「期限切れ」トラックのみ実行（新規は即、未確定は毎フレーム、
   確定済みは 30 フレームごとのローテーション再検証）——実測で静止フレーム 61 ms、新品出現 ~200 ms で確定。
3. **追跡バックエンド**：`SHELF_TRACKER=bytetrack.yaml`（ultralytics 内蔵、ゼロ追加依存）。`off` で従来の貪欲 IoU マッチングに回退可能。
4. **投票/ヒステリシス**：`CONFIRM_FRAMES=3` / `VOTE_MIN=3` / `MATCH_THRESHOLD_KEEP=0.55`——単一フレームの誤検出でラベルが書き換わらないことを保証。
5. **マルチスケールクエリ**：`MATCH_SCALES=1.0,1.2`（§7.2 README 参照）——「ぴったりクロップ vs 参照写真まるごと」のスケール不一致による検索スコアの押し下げを緩和。
