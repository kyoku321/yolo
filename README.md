# 🛒 スーパーの棚の商品認識 Demo
**SKU-110K デテクタのトレーニング方法**と**デバッグ方法**を順に説明する。

---

## 1. これは何？

棚の写真を入力すると、**各商品の位置ボックス＋商品名**（価格・バーコードなどのメタデータ付き）を出力する：

```
棚の写真 ──► YOLO が N 個の商品ボックスを検出 ──► ボックスごとにクロップ
                                          │
                              CLIP が各クロップを 512 次元ベクトルに変換
                                          │
                              「商品DB」内のベクトルとコサイン類似度を比較
                                          │
                                ヒット → 商品名を表示 / ミス → Unknown
```

これは業界の主流である**検索ベース（retrieval-based）の商品認識**方式
（SKU-110K / coarse-to-fine 論文の発想）：

- **検出**（YOLO）は「どこに商品があるか」だけ答え、**単一クラス**の問題；
- **認識**（CLIP + ベクトル検索）は「これはどの SKU か」答え、**検索**の問題。

**コアの利点**：新商品を扱う場合は、Web ページで参照写真 1–5 枚をアップロードして登録するだけでよく、
**どのモデルも再学習しなくていい**。これが検索ベース方式が小売業界で事実上の標準になった理由だ。

### 初心者が必ず押さえるべき 5 つの概念

| 概念 | 一言で説明 |
|------|-----------|
| **embedding（ベクトル）** | CLIP は任意の画像を 512 次元の数値ベクトルに変換する。見た目類似の画像はベクトルも近くなる。登録と認識は**同じ** CLIP モデルを使うので、ベクトルは同じ「空間」にあり比較できる。 |
| **コサイン類似度** | 2 つのベクトルの挟角を表す 0~1 の小数（本プロジェクトのベクトルは全て L2 正規化済みなので、内積 = コサイン）。`MATCH_THRESHOLD`（デフォルト 0.65）以上で「認識した」扱い、それ以外は Unknown と表示。 |
| **IoU** | 2 つのボックスの重複面積 ÷ 和集合面積。NMS はこれで重複除去し、リアルタイム追跡はこれで「このボックスは前フレームのどのターゲットと同一商品か」を判断する。 |
| **NMS（非最大値抑制）** | デテクタは同じ商品にほぼ重なる複数ボックスを出しがち。NMS は IoU 閾値に従って重複ボックスを潰す。 |
| **オープンボキャブラリ検出（YOLO-World）** | 学習なしにテキストプロンプト（"bottle", "package"…）で任意の物体を検出するデテクタ。ゼロ学習で使えるが、ボックスは緩めになりがち。 |

---

## 2. 全体アーキテクチャ

```
┌────────────────────────────────────────────────────────────────────────┐
│ ブラウザ（Gradio Web UI, http://127.0.0.1:7860）                       │
│                                                                        │
│   Tab 1 Register        Tab 2 Recognize   Tab 3 Live     Tab 4 Catalog │
│    参考画像登録          棚画像認識       カメラ実時間   カタログ管理  │
│   Gradio フォーム       Gradio フォーム   自作 HTML/JS   自作 HTML/JS  │
└───────┬───────────────┬────────────────┬──────────────┬────────────────┘
                │               │               │  JSON over HTTP               │  JSON over HTTP
                │               │               │  POST /live/api/frame               │  GET /catalog/api/list
                │               │               │  POST /live/api/reset               │  POST /catalog/api/delete
                │               │               │  GET  /live/api/grasp               │
                │               │               │               │  （ロボットアームの grasp 準備 API）
│                ▼               ▼               ▼               ▼       │
┌────────────────────────────────────────────────────────────────────────┐
│ app.py — Gradio Blocks + FastAPI ルーティング                          │
│ （重いモデルは遅延ロード、メインスレッドでウォームアップ）             │
└───────┬───────────────┬────────────────┬──────────────┬────────────────┘
                │               │               │               │
│                ▼               ▼               ▼               ▼       │
┌────────────────────────────────────────────────────────────────────────┐
│ shelf_demo/pipeline.py — ShelfPipeline                                 │
│ （エンドツーエンドのオーケストレーション、唯一のエントリ）             │
│ register_product()   recognize()   match_boxes()   delete_product()    │
│  ┌───────────────┬───────────────┬──────────────────────┬────────────┐  │
│  │ ▼             │ ▼             │ ▼                    │ ▼          │  │
│  │ detector.py   │ embedder.py   │ database.py          │ crop 管理  │  │
│  │ YOLO 検出     │ CLIP ベクトル │ SQLite メタデータ    │ data/crops/ │  │
│  │ 3 種バックエンド │ (512 次元)    │ + NumPy ベクトルDB   │            │  │
│                                                 data/catalog.sqlite         │
│                                                 data/vectors.npz             │
│ live.py — LiveRecognizer（ByteTrack 追跡 + マルチフレーム投票 + 定期再確認）│
│ live_web.py / catalog_web.py — 自作フロントエンドテンプレート（HTML/CSS/JS）│
│ catalog_api.py — Catalog ページの JSON handler                         │
│ draw.py — 枠線/サマリー描画    config.py — 全設定（env で上書き可）    │
└────────────────────────────────────────────────────────────────────────┘
```

**モジュールの責務を一言で**：

| モジュール | 責務 |
|------|------|
| `shelf_demo/config.py` | すべての調整可能パラメータ（パス/検出/CLIP/検索/デバイス）、全て環境変数で上書き可能 |
| `shelf_demo/detector.py` | `Detector` / `AutoDetector`：YOLO 推論をラップし、`Box(x1,y1,x2,y2,conf)` のリストを出力。バックエンド：`auto` / `custom` / `world` / `coco` |
| `shelf_demo/embedder.py` | `Embedder`：open_clip 画像エンコーダ、画像 → L2 正規化済み (N, 512) ベクトル |
| `shelf_demo/database.py` | `Catalog`：SQLite に SKU メタデータを保存 + `VectorStore`（NumPy による厳密なコサイン近傍探索）にベクトルを保存。`add / search / delete` |
| `shelf_demo/pipeline.py` | `ShelfPipeline`：上記 3 ブロックを繋ぐ。登録・認識・削除はいずれもここで入出する |
| `shelf_demo/live.py` | リアルタイム認識：`LiveRecognizer`（ByteTrack トラッカー + 時系列投票）+ セッション管理 + フレーム JSON の処理 |
| `shelf_demo/live_web.py` | Live ページのフロントエンド（canvas による枠描画、getUserMedia によるフレーム取得、パラメータスライダー） |
| `shelf_demo/catalog_api.py` / `catalog_web.py` | Catalog ページのバックエンド JSON ハンドラ / フロントエンドのテーブル |
| `shelf_demo/draw.py` | 認識結果の枠描画（緑=ヒット、赤=Unknown）+ テキストサマリー |
| `app.py` | Gradio UI の 4 タブ + FastAPI カスタムルートのマウント |
| `scripts/` | トレーニング（§8）、デバッグ（§9）、エンドツーエンドテスト（§13）のスクリプト |

---

## 3. ディレクトリ構造

```
yolo/
├── app.py                    # Web エントリ：Gradio 4 タブ + FastAPI ルート
├── requirements.txt
├── yolov8n.pt                # COCO 事前学習ウェイト（6 MB、coco バックエンド用）
├── yolov8s-worldv2.pt        # YOLO-World ウェイト（25 MB、world バックエンド用）
├── weights/
│   ├── sku110k_best.pt       # ★ 本プロジェクトが SKU-110K で学習した単一クラスデテクタ（存在すれば自動有効化）
│   └── clip/ViT-B-32.pt      # 事前ダウンロード済みの CLIP ウェイト（デフォルト CLIP は ViT-B-16 で初回実行時に自動ダウンロード）
├── shelf_demo/               # コアコード（§2 の責務表参照）
│   ├── config.py  detector.py  embedder.py  database.py
│   ├── pipeline.py  live.py  draw.py
│   ├── live_web.py  catalog_api.py  catalog_web.py
├── scripts/
│   ├── train_sku110k.py      # ★ SKU-110K 棚デテクタの学習（§8 参照）
│   ├── smoke_test.py         # GUI なしで実行するエンドツーエンド自己チェック
│   ├── debug_detect.py       # 「商品が検出されない」段階別診断
│   ├── compare_detectors.py  # custom vs YOLO-World のボックス品質比較図
│   ├── sweep_detect.py       # imgsz/conf/NMS-iou のパラメータスウィープ
│   ├── test_grasp.py         # grasp 準備インターフェース（ロボットアーム）の単体テスト、モデル不要で秒オーダー
│   ├── e2e_live_ui.py        # [dev] Live ページの Playwright E2E テスト
│   └── e2e_catalog_ui.py     # [dev] Catalog ページの Playwright E2E テスト
├── datasets/SKU-110K/        # 学習データ（train.txt 8219 / val 588 / test 2936、単一クラス object）
├── runs/detect/...           # 学習成果物（loss カーブ、best.pt など）
├── data/                     # 実行時データ（§7 参照）
└── docs/plans/               # 設計ドキュメント（リアルタイム動画注釈の進化記録）
```

---

## 4. クイックスタート

```bash
# 1. 環境（Python 3.12 推奨）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 注：初回実行時に ultralytics が YOLO-World が必要な openai CLIP テキストエンコーダを自動で補装する

# 2. 自己チェック（「登録→検索」の閉ループのみ検証。デテクタ/カメラ不要。初回は CLIP ウェイト ~600 MB をダウンロード）
python scripts/smoke_test.py

# 3. Web UI を起動
python app.py            # http://127.0.0.1:7860 を開く
```

起動時に YOLO + CLIP モデルを**メインスレッド**でウォームアップする（MPS/torch のネイティブ初期化は
Gradio のワーカースレッドに向いていない）。その後 UI はすぐ開き、認識も滞らない。

**使用フロー**：

1. **Tab 1 · Register**：商品の正面写真 1–5 枚 + 名称（必須）/価格/バーコード/カテゴリ をアップロード → 登録。
2. **Tab 2 · Recognize**：棚写真をアップロード → 左側に枠付きの結果画像（緑枠=認識、赤枠=Unknown）、右側に数量統計。imgsz / conf / マッチ閾値を調整可能。
3. **Tab 3 · Live**：Start をクリックしてカメラ権限を許可し、棚に向けて → リアルタイム枠描画 + 商品名が映像を追従。カメラ、検出解像度、conf、閾値を調整可能。Re-recognize で全量再認識を強制。
4. **Tab 4 · Catalog**：テーブルで全 SKU を閲覧（登録写真のサムネイル付き、クリックで拡大）。各行の Delete でその SKU を削除（ベクトル + メタデータ + 写真ファイルをまとめて削除）。

> ブラウザのカメラは「安全なコンテキスト」でのみ使える：`http://127.0.0.1:7860` はそのまま使える；
> LAN IP でのアクセスには HTTPS または `gradio launch(share=True)` が必要。---

## 5. 各機能の背後にあるアーキテクチャ（タブ別解説）

### 5.1 Tab 1 · Register：商品登録

**呼び出しチェーン**：

```
gr.Button("Register")
  └─► app.py do_register(files, name, barcode, price, category)
        └─► pipeline.register_product(images, ...)
              ├─► Embedder.embed(images)          # ① CLIP：N 枚の画像 → (N, 512) L2 正規化ベクトル
              ├─► Catalog.add_product(...)        # ② SQLite に 1 行挿入 + vectors.npz に N 個のベクトルを追加（同じ sku_id を付与）+ ディスクへ保存
              └─► _save_reference_crops(...)      # ③ 参照写真を data/crops/skuNNNN_<タイムスタンプ>_i.jpg に保存（Catalog ページ表示用）
```

**ポイント**：

- 1 つの SKU は **N 個のベクトル**を持つ（N = 参照写真の枚数、1–5 枚）。複数枚 =
  多角度のカバレッジで、検索時は「その SKU の最良ベクトル」が勝つ（5.2 参照）。
- 登録は**モデルの重みに一切触れない**——これが検索ベース方式の売り：新商品の一括登録は秒オーダーで完了。
- 登録写真をファイルとして保存するのは Catalog ページでの表示と人手による確認のため。検索自体はベクトルしか見ない。

### 5.2 Tab 2 · Recognize：棚認識（コアパイプライン）

**呼び出しチェーン**：

```
gr.Button("Recognize")
  └─► app.py do_recognize(image, conf, threshold, imgsz)
        └─► pipeline.recognize(shelf_image, ...)
              ├─► build_detector().detect(image, conf, imgsz)  # ① YOLO → [Box]（単一クラス、agnostic NMS）
              │     DETECTOR=auto の場合 = 自学習モデル + 単品クローズアップ用フォールバック（§7.1）
              └─► pipeline.match_boxes(image, boxes, thr)      # ② ボックスごとにマルチスケールクロップ → CLIP バッチエンコード → ベクトル検索
                        └─► rank_boxes(...)   # 各ボックスを MATCH_SCALES に従って複数枚クロップし、SKU ごとに最高スコアを採用（§7.2）
                              └─► Catalog.search(embeddings, topk=5)
                                    すなわち (n,512) @ (512,N) の 1 回の行列積 + SKU ごとに最良を採用
              └─► draw_recognitions(...) + summary(...)        # ③ 枠描画 + 統計テキスト
```

**ポイント**：

- **検出と認識は完全に分離**：デテクタは「枠」だけを担当し、SKU が何かは完全にベクトル検索が決める。
  そのためデテクタの差し替え / CLIP モデルの差し替え / 商品の増減は互いに影響しない。
- 1 ボックス → マルチスケールクロップ → 各クロップが 1 ベクトル → DB 内全ベクトルとコサインを計算し、
  **まず SKU ごとに集約して最高スコアを取り、次にスケール横断で最高スコアを取る**（1 つの SKU が参照画像
  5 枚あれば 5 個のベクトルがあり、高い方が勝つ）。top1 スコアが `MATCH_THRESHOLD` 以上でヒット、それ以外は
  `Unknown (score)`。
- `match_boxes` を `recognize` から切り出したのは、Live ページ（5.3）が**新規ターゲットのみ**に
  CLIP を走らせるため——写真ページとリアルタイムページで同一の認識ロジックを共用する。ランキングロジックは
  `rank_boxes()` に置き、`debug_detect.py` もこれを使うことで、デバッグツールと app のスコアが完全に一致することを保証する。
- `agnostic_nms=True`：YOLO-World では 1 つの物体が "package"/"box"/"pouch"
  といった複数プロンプトに重複ヒットすることがある。クラス単位の NMS では除去できず、クラス横断 NMS なら 1 ボックスに潰せる。

### 5.3 Tab 3 · Live：カメラリアルタイム注釈（性能設計のコア）

**なぜ毎フレーム全量実行できないのか？** M1 MPS 実測：YOLO @960 ≈ 45 ms/フレーム（~22 FPS）だが、
CLIP のクロップ 1 枚 ≈ 150–300 ms で、1 フレームに 30 個のボックスあれば 5–9 秒になる。そこでコア戦略は：

> **YOLO は毎フレーム実行（速い）、CLIP は「期限切れ」のトラックのみ実行（高コスト）。**
> ターゲットの同一性は ByteTrack が維持し、ラベルはマルチフレーム投票で確定させ、定期的に自動で再検証する。

**フロントエンド**（自作 HTML/JS、`live_web.py`）：

```
ブラウザの getUserMedia でカメラフレームを取得（≤1280px に縮小、JPEG quality 0.7）
  → 毎フレーム POST /live/api/frame {uuid, image(data-url), imgsz, conf, threshold}
     （同時刻に 1 件のみリクエストを進行中として許可。戻らなければそのラウンドをスキップして滞留防止。8s タイムアウトで中断）
  → サーバが JSON {boxes[{xyxy,label,match}], n_objects, n_matched, n_new, ms, fps} を返す
  → ブラウザが映像フレーム + ボックス/ラベルを <canvas> に描画。独立した 66 ms タイマーで再描画（~15 FPS）
```

オーバーレイのボックスは**ブラウザが描画**し、サーバが描画済み JPEG を返す方式ではない——ネットワーク負荷が極小
（座標とテキストのみ）であり、検出が少し遅れても映像は滑らかに保てる。

**バックエンド**（`live.py`）：

```
POST /live/api/frame  (asyncio.to_thread で 50–300ms の MPS 計算をイベントループ外へ)
  └─► handle_frame(payload, pipeline)
        ├─► SESSIONS.get(uuid) → LiveRecognizer     # ブラウザセッションごとに 1 個。uuid はフロントの crypto.randomUUID() が生成
        │     （120s 放置で自動回収。_map_lock がセッションテーブルを保護、model_lock がモデル呼び出しを直列化——
        │      MPS は並行推論を嫌うため、2 つのロックは絶対にネストせず、自己デッドロックを回避）
        └─► rec.process(image)
              1) YOLO 検出（毎フレーム、imgsz デフォルト 960。検出確信度は ByteTrack の
                 低確信度帯で実行し、弱い枠は関連付けには参加するが最終出力には出さない）。DETECTOR=auto の場合、
                 自学習モデルが密集棚に「見えない」時は自動で YOLO-World フォールバックへ切替え（§7.1）、かつ
                 そのフレームの ByteTrack 新規トラック閾値を YOLO_WORLD_CONF まで下げる。
                 さもなくばフォールバック枠のスコアが低すぎてトラックが作れない
              2) セッションごとに ByteTrack。Kalman による安定化 + 高/低確信度の 2 段階関連付けでフレーム横断の id を安定
                 （SHELF_TRACKER、デフォルト bytetrack.yaml。=off で従来の貪欲 IoU マッチングに回退）
              3) CLIP + 検索は「期限切れ」のトラックのみ実行：新規トラックは即座に、未確定トラックは毎フレーム、
                 確定済みトラックは 30 フレームごとにローテーションで再検証（1 フレームあたり最大 4 個、かつ 1 フレーム全体で
                 EMBED_MAX_PER_FRAME=8 個の枠まで CLIP に入れ、特定フレームでの計算リソースのスパイクを防止）
              4) 時系列投票：各トラックは直近 5 回の検索結果を保持し、票数と平均スコアの両方がパスした
                 （CONFIRM_FRAMES=3 / VOTE_MIN=3）時点で SKU を確定。各検索は
                 MATCH_SCALES に従ってマルチスケールクロップし最高スコアを採用（§7.2）。確定前は
                 Unknown（スキャン中）を表示し、grasp には参加しない。ヒステリシス閾値 MATCH_THRESHOLD_KEEP=0.55
                 で確定済みラベルを過敏にしすぎず、誤検出 1 回でラベルを書き換えられないようにする
              5) 現在の全ボックスの認識結果 + changed フラグを出力（変化がなければサマリーテキストを返さない）
```

**効果**：棚が静止している時、フレームレート ≈ 純検出の上限（静止フレーム ~61 ms/フレーム ≈ 16 FPS）。
カメラが新品に向くと、新規ターゲットはまず ByteTrack の 1 フレームゲートを通過し、その後数フレームにわたって各 1 回 CLIP を実行、
約 3 フレーム（~200 ms）でラベルを確定——その間は Unknown 表示で grasp 不可。そのためラベルはちらつかない。

**フロントエンドが Gradio 標準の `gr.Image(streaming=True)` を使わない理由**：Gradio 6 の
録画ボタンの表示は内部の `stream_state` が駆動しており、app レイヤではリセット不能——停止をクリックしてもボタンが
"Stop" のまま固まったまま、サーバ側のセッションも実際には終了しない（ソースレベルの診断で確認。詳細は
`docs/plans/2026-08-12-realtime-video-annotation-design.md` 参照）。自作フロントにしたことで
Start/Stop の状態は常に真実を反映する。フロントには **generation カウンタ**もある：停止後に到着した
進行中のレスポンスは破棄されるため、「停止済み」の状態を上書きされない。

**Re-recognize ボタン**：POST `/live/api/reset` でそのセッションのトラッカーと投票履歴をクリアし、
次のフレームで全ターゲットが確認フローを最初からやり直す。SKU が削除/再登録された後に手動で押す必要は
実際にはない：確定済みトラックは 30 フレームごとに自動で再検証され、自ら修正する。即座に更新したい時に押せばいい。

**レスポンスの `id` フィールド**：各ボックスは安定した `track_id` を持つ（セッション内で自動増加、トラック存続中は不変）。
棚に同じ商品の複数個体がある場合、フレーム横断で「どの物理ターゲットか」を区別するのはこれ——ロボットアームは必ずこれを使わねばならず、名前では区別できない。

#### ロボットアーム統合：`GET /live/api/grasp`

ロボットアームに特定の名称の商品を grasp させたい場合、サーバは「grasp 準備」インターフェースを提供し、タイミング判断ロジックを
トラッカー内部にカプセル化（`LiveRecognizer._update_grasp` が毎フレーム各名称の安定ウィンドウを維持）：

```
GET /live/api/grasp?uuid=<Live セッション id>&name=<商品名>
→ {ok, ready, reason, track_id, box[x1,y1,x2,y2], point[cx,cy],
   frames_stable, age_ms, img_w, img_h, available_names[]}
```

**`ready=true`（= 最良の grasp 瞬間）の条件**：その名称の**同じトラック**が連続
`GRASP_WINDOW=5` フレーム（~300 ms @16 FPS）出現し、かつ：

- 隣接 2 フレームのボックス IoU ≥ `GRASP_IOU=0.9`（位置が収束、YOLO 枠の揺れを除外）；
- その間に新規ターゲットが確定していない（新規トラックが投票完了したフレームのみで全ウィンドウをリセット；
  単一フレームの誤検出でできた未確定トラックは grasp に影響を与えない）；
- セッションが 1 秒以内（`GRASP_MAX_AGE_MS`）にフレームを処理している（停止したカメラを向いて grasp しないための防止策）。

返ってくる `box` はウィンドウ 5 フレームの**平均ボックス**（揺れ除去）、`point` はその中心点。

**ロボットアームコントローラ推奨フロー**：

1. Live ページでカメラを起動しておく（セッション id は
   `document.querySelector('[data-role="slv-root"]').dataset.slvSession` から読む）；
2. 100–200 ms ごとに `grasp?name=X` をポーリング（`available_names` で名称の綴りミスをチェック）；
3. `ready=true` になったら、さらに `/live/api/frame` に 1 フレーム同期送信し、そのレスポンスの枠を使う（最新、
   遅延 ≤1 フレーム）として grasp 動作を発火；
4. grasp 前にカメラと棚は固定されていること。`reason` フィールドがなぜまだ準備できないかを示す
   （未安定 / シーン変化あり / カメラ未起動）。

> ⚠️ `box`/`point` は**送信画像のピクセル上**の 2D 座標。ロボットアームはカメラ内パラメータ +
> 手眼キャリブレーション（hand-eye calibration）でこれらを 3D ポーズに変換する必要がある（深度カメラがある場合、ボックス内の中心点の深度を取る方が安定）。

### 5.4 Tab 4 · Catalog：商品 DB 管理

**呼び出しチェーン**：

```
フロント JS (catalog_web.py, gr.HTML で注入)
  ├─► GET  /catalog/api/list
  │     └─► catalog_api.list_catalog → SKU ごとに 1 行 + 写真 URL リスト（/catalog/photos/<ファイル名>）
  │        （写真ディレクトリ data/crops は Starlette StaticFiles で /catalog/photos にマウント）
  └─► POST /catalog/api/delete {sku_id}
        └─► catalog_api.delete_sku → pipeline.delete_product(sku_id)
              ├─► VectorStore.remove(sku_id)   # その SKU の全ベクトルを削除
              ├─► SQLite DELETE products       # メタデータ行を削除
              └─► data/crops/skuNNNN_*.jpg を削除  # 参照写真ファイルを削除
```

純 JSON 入出力のハンドラ（`catalog_api.py`）はどのフレームワークにも依存せず、単体テストが容易；
`app.py` がこれらを FastAPI ルートにラップする役割を負う。

### 5.5 app.py の「書かないと必ず踏む 2 つの坑」

1. **ルートは `_app=` で渡す**：`demo.launch()` は内部で Gradio の FastAPI
   app を再構築し、事前に `demo.app` にマウントしたカスタムルートは破棄される。そこで `build_app()`
   が新たに `gradio.routes.App()` を作り、全ルートをマウントしてから、`_app=fastapi_app` で起動する。
2. **重い処理は必ず `asyncio.to_thread`**：YOLO/CLIP は同期的ブロッキング呼び出し。スレッドに移さないと
   Gradio のイベントループを止め、UI 全体が無応答になる。

---

## 6. データフローと永続化

登録と認識は同一のベクトル空間を共有（同一の CLIP モデルでエンコード）——これが全チェーンが成立する根本条件：

```
  登録（一回限り）                                  認識（クエリ毎）
  ┌────────────────────────┐                        ┌──────────────────────────────────┐
  │ 参考写真 ×N            │                        │ 棚の写真                         │
  │ ▼                      │                        │ YOLO → Box ×N → crops ×N         │
  │ CLIP.encode_image      │                        │ ▼                                │
  │ ▼                      │                        │ CLIP.encode_image（同一モデル）  │  同じ
  │ (N, 512) 正規化ベクトル│                        │ ▼                                │  コサイン類似度 vs DB 内全ベクトル
  └───────────┬────────────┘ 同一ベクトル空間 ─────►│                                  │
              ▼                                     └────────────────┬─────────────────┘
                                                                     ▼
   ┌────────────────────────────────────────────────data/ ディレクトリ──────────────────────────────────────────────────┐
   │ vectors.npz     {vectors: (N,512) float32, ids: (N,) int64}                                                        │
   │                 ↑ N は参考写真の総数（SKU 数ではない）。ids は各ベクトルの所属 sku_id を記録                       │
   │ catalog.sqlite  products(sku_id, name, barcode, price,                                                             │
   │                 category, n_refs, created_at)                                                                      │
   │ crops/          sku0007_20260811113440_0.jpg …（表示用の写真）                                                     │
   └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- 各 add/delete の後、`vectors.npz` は全量書き換え（数千ベクトルの規模ならミリ秒オーダー）。
- CLIP モデルの差し替え = ベクトル空間が変わる → 旧ベクトルは全滅し、**全商品の再登録が必要**
  （`VectorStore._load` はファイルに保存された次元を正とし、次元不一致時のサイレントな誤マッチを防止）。

---

## 7. デテクタの選択（「商品が検出できるか」を直接決定する）

`DETECTOR` 環境変数でバックエンドを切替（`shelf_demo/detector.py` 内でディスパッチ）：

| `DETECTOR` | モデル | 原理 | ボックス品質 | いつ使うか |
|-----------|------|------|--------|--------|
| `auto`（**デフォルト**、custom ウェイトの存在が必要） | SKU-110K 自学習 YOLO **+** YOLO-World フォールバック | まず自学習モデルで「密集棚に見えるか」を判定し、見えない時にゼロショットモデルを実行 | 密集棚=★ 多くてぴったり；単品クローズアップ=★ 1 個の正確な枠 | **推奨**。棚と「手持ちの単品をカメラに向けた」両方で動きたい場合 |
| `custom` | 本プロジェクトが SKU-110K で学習した YOLOv8n（単一クラス "product"） | 教師ありファインチューニング | 密集棚 ★、**単品クローズアップ ✗** | 密集棚のみ撮影で最速を重視する場合 |
| `world`（custom ウェイトが無い時のデフォルト） | YOLO-World (`yolov8s-worldv2.pt`) | オープンボキャブラリ、テキストプロンプト駆動、**ゼロ学習** | 単品クローズアップ ★、密集棚はやや緩め | 学習ウェイトが無い時；または手持ちの単品のみ認識する場合 |
| `coco` | 通常 YOLOv8n (`yolov8n.pt`) | COCO 80 クラス事前学習 | スナック/パッケージには**ほぼ検出不能** | まず UI を動かすだけの場合 |

**デフォルトロジック**（`config.py`）：`weights/sku110k_best.pt` が存在 → 自動で `auto`；
無ければ `world` にフォールバック。

### 7.1 なぜ `auto` が必要か：2 つのバックエンドは真逆のシーンでそれぞれ失効する

SKU-110K 自学習モデルの学習事前知識は「**1 セルにびっしりと小さな商品が詰まっている**」。
**単一の**商品を OBSBOT カメラの前に突き出すと完全に out-of-distribution：

| シーン | `custom` 検出 | YOLO-World 検出 |
|------|--------------|----------------|
| 4K 密集棚 `data/shelf.jpg` | **77** 個のぴったり枠、映像面積の 35% をカバー | 9 個の緩い大枠（1 列まるごと枠に入れることも） |
| 単品クローズアップ（`data/crops/` の参照画像 17 枚） | **0/17 に有用な枠なし**：枠 0 個、または 10px の小枠を数十~数百個ハルシネーション（最大の枠も映像の 0.4% 以下） | **15/17** で高確信度の 1 個の大枠 |

つまり「SKU は登録済みなのに、カメラの前に持っても bounding box が現れない」のは閾値の問題ではない——**`conf` を
下げても自学習モデルは大量の小枠を吐くだけ**で、本物の枠はそもそも学習できていない。`auto` のやり方は
**出力形状で切替える**（`AutoDetector`、閾値は全て `config.py` にある）：

```
coverage = 自学習モデルの全ボックスの和集合 / 映像面積
max_area = 最大のボックス / 映像面積
trusted  = 枠が存在 and coverage >= AUTO_COVERAGE_MIN(0.15) and max_area >= AUTO_MIN_AREA(0.01)

trusted  → 自学習モデルの枠を使う（密集棚、変更前とバイト単位で一致）
trusted でない → YOLO-World を実行；「大きな物体」(max_area >= AUTO_SWITCH_AREA=0.05) を見つけた
            場合のみ採用、そうでなければ自学習モデルの結果を維持
```

閾値は実測に基づき、思いつきで決めたものではない：密集棚の coverage は 0.20–0.44、最大ボックス 2.0–3.5%；
単品クローズアップ 17 枚は coverage ≤0.16、最大ボックス <1%（大半 <0.4%）で、両端に十分な余裕がある。
`auto` では密集棚は 1 フレームたりともフォールバックをトリガーしない（`path=specialist`）、単品クローズアップのみがトリガーする
（`path=fallback`）。

**再現可能な証拠**：

```bash
python scripts/verify_detection.py     # クローズアップ 17 枚：custom 0/17 → auto 15/17 に有用な枠；
                                       # 密集棚：auto は custom とバイト単位で一致（35 ボックス）
python scripts/test_live_closeup.py    # 本物のチェーン（CLIP 検索含む）：
                                       # ALKALINE 参照画像 → 1 ボックス + ラベル ALKALINE(0.68)
```

**コスト**：フォールバックフレームで YOLO-World を 1 回余分に実行（@960 で約 185 ms、@640 で約 90 ms）。密集棚では
トリガーされないため、通常のフレームレートには影響しない。`app.py` は起動時にフォールバックモデルをウォームアップし、初回フレームの固まりを回避する。

#### 「このフレーム」が自学習モデルか YOLO-World かを見分ける方法

`auto` は**フレーム単位**で決まるため、次のどの方法で見ても「直近 1 フレーム」の結果になる。
内部状態は `detector.last_path`（`specialist` = 自学習モデル / `fallback` = YOLO-World）、
対外は統一して `detector.active_backend`（`custom` / `world`）として公開：

| 方法 | 見る場所 | 説明 |
|------|--------|------|
| **Live ページ（最も直感的）** | ステータスバー末尾の `· model SKU-110K` または `· model YOLO-World` | 毎フレーム更新；単品を持っている時は YOLO-World、棚を向いている時は SKU-110K と表示 |
| **起動バナー** | `python app.py` が出力する `検出器 Detector : auto — ...` | auto モードが有効であることを示すだけで、某フレームで誰が使われたかは示さない |
| **デバッグスクリプト** | `debug_detect.py` の `[STAGE 1] DETECTION: N boxes ... (path: fallback)` | 単一画像、最も詳細 |
| **バッチ比較** | `verify_detection.py` 出力の `path=` 列 | `custom`/`world` の単独実行結果と並べて見る |
| **コード内** | `pipeline.detector.active_backend` | ロボットアーム / 外部プログラムの統合はこちらを使う |
| **強制固定** | `DETECTOR=custom` または `DETECTOR=world` | auto の影響を排除して某バックエンドを再現したい時 |

```bash
# 最も手軽な判断：商品をカメラの前に突き出し、Live のステータスバーを見る
#   model YOLO-World と表示  -> 自学習モデルがシーンを認識できずフォールバックに入った（正常）
#   model SKU-110K と表示    -> 自学習モデルがこれを密集棚と判断
python app.py

# 単一画像の CLI 検証
python scripts/debug_detect.py data/crops/sku0026_20260901002504_0.jpg --imgsz 960
#   -> [STAGE 1] DETECTION: 1 boxes ... (path: fallback)
python scripts/debug_detect.py data/shelf.jpg --imgsz 960
#   -> [STAGE 1] DETECTION: 35 boxes ... (path: specialist)
```

**検証済みの比較**（`python scripts/compare_detectors.py data/shelf.jpg`、
出力は `data/compare_custom.jpg` / `data/compare_world.jpg`）：同じ 4K 棚画像で、
custom は単品にぴったりの **61** 個のボックスを検出（中央値 200×148 px）、YOLO-World は **21**
個の緩い大枠のみ（中央値 412×333 px、頻繁に 1 列まるごと枠に入れる）。

YOLO-World のプロンプトは `YOLO_WORLD_PROMPTS` でカスタマイズ可能（カンマ区切り）。デフォルトでは
小売/パッケージ語 20 個まで拡張済み（`...,battery,jar,tube,cylinder,container,cup,cup noodle,tin,object`）：
この数語の追加でフォールバックのヒット率が 13/17 から 16/17 に向上。フォールバックモデルは独自の確信度
`YOLO_WORLD_CONF=0.05` と NMS `YOLO_WORLD_IOU=0.5` を使う（自学習モデルの 0.2/0.3 と分離）。

**デフォルトパラメータがデテクタごとに変わる理由**（「商品が検出されない」の第 1 の坑）：

| パラメータ | `custom` | `world`/`coco` | 理由 |
|------|----------|----------------|------|
| `DETECT_CONF` | 0.2 | 0.05 | 学習済みモデルは確信度のキャリブレーションが良い；YOLO-World のゼロショットスコアは全体的に低く、閾値が高ければ即座に 0 枠 |
| `DETECT_IOU` | 0.3 | 0.5 | 単一クラス学習モデルは同じ商品に半セルずれた重複枠を重ねる（IoU≈0.4）。0.5 では除去できず、0.3 で除去でき、本当に隣接している商品を誤って合併することもない |
| `DETECT_IMGSZ` | 1280 | 1280 | 4K 棚写真を 640 に圧縮すると商品が小さすぎる；密集棚は 1920 推奨 |

**検証済みの比較**（`python scripts/compare_detectors.py data/shelf.jpg`、
出力は `data/compare_custom.jpg` / `data/compare_world.jpg`）：同じ 4K 棚画像で、
custom は単品にぴったりの **61** 個のボックスを検出（中央値 200×148 px）、YOLO-World は **21**
個の緩い大枠のみ（中央値 412×333 px、頻繁に 1 列まるごと枠に入れる）。

YOLO-World のプロンプトは `YOLO_WORLD_PROMPTS` でカスタマイズ可能（カンマ区切り）。デフォルトでは
小売/パッケージ語 20 個まで拡張済み（`...,battery,jar,tube,cylinder,container,cup,cup noodle,tin,object`）：
この数語の追加でフォールバックのヒット率が 13/17 から 16/17 に向上。フォールバックモデルは独自の確信度
`YOLO_WORLD_CONF=0.05` と NMS `YOLO_WORLD_IOU=0.5` を使う（自学習モデルの 0.2/0.3 と分離）。

### 7.2 付随修正：検索側のマルチスケールクロップ（枠は出た、あとは正しく認識できなければ）

検出の修正だけでは不十分：枠が出てもラベルが赤の `Unknown` なら、体感は同じく壊れている。
ALKALINE 参照画像の実測——検出を直して 1 個の枠が出たが、検索スコアは **0.592** しか無い（閾値 0.65）。
根本原因はランキングにはない（top-1 は ALKALINE、2 位が 0.360）で、**スケール不一致**：

```
商品DB に保存されているのは「登録時の写真まるごと」；
クエリ時に CLIP に渡しているのは「検出枠のぴったりクロップ」。
ぴったりクロップは参照写真の背景/比率の文脈を失う → 同一商品のコサインが押し下げられる。
```

`rank_boxes()`（`shelf_demo/pipeline.py`）は現在、各ボックスを `MATCH_SCALES`
（デフォルト `1.0,1.2`）で各 1 枚クロップし各 1 回エンコードして、**SKU ごとに最高スコアを採用**（test-time
augmentation で、真のヒットのスコアは上がる一方）：

| 指標（参照画像 17 枚、`scripts/eval_match.py`） | ぴったりクロップ 1.0 | +1.2 | マルチスケール最高値 |
|---|---|---|---|
| 真のヒットのコサイン平均 | 0.751 | 0.801 | **0.824** |
| 真のヒットのコサイン中央値 | 0.797 | 0.876 | **0.876** |
| `MATCH_THRESHOLD=0.65` を超えた枚数 | ~9/17 | 15/17 | **15/17** |

ALKALINE は 0.587 → **0.671**（閾値通過、ラベルが緑に）。コストは CLIP エンコード 1 回分；
計算リソースを節約したい場合は `MATCH_SCALES=1.0` で無効化可能。

> より徹底的な解決策は**登録時も物体クロップを使う**こと（DB に単品クロップを保存し、クエリと同一分布にする）だが、
> 既存 SKU の再登録が必要になるため、ここではデータを改めないマルチスケールクエリで同じアライメント効果を先に実現する。

---
## 8. 自分でデテクタを学習する：SKU-110K 完全ガイド

> 目標：**単一クラスで、密集棚にボックスが多くてぴったり**の商品デテクタを得て、
> ボックスが緩めのゼロショット YOLO-World を置き換える。

### 8.1 なぜ SKU-110K データセットか？

- **SKU-110K** は小売棚向けの密集物体検出データセット：棚画像約 1.17 万枚
  （このリポジトリの `datasets/SKU-110K/`：train 8219 / val 588 / test 2936）、
  **11 万+ のアノテーションボックス**、画像は 3024×3024。
- クラスは `object`（= 商品）のみ——**まさに我々が望んでいる形**：デテクタは
  「どこに商品があるか」だけを学習し、商品が何かは下流の検索パイプラインに委ねる。
- アノテーションは CSV（`image_name,x1,y1,x2,y2,class,image_width,image_height`）、
  Ultralytics は `data="SKU-110K.yaml"` で参照する。

### 8.2 学習コマンド

```bash
# 完全品質の学習（CUDA マシン推奨；M シリーズ Mac も mps で可、やや遅い）
python scripts/train_sku110k.py --model yolov8n.pt --epochs 50

# 高速 demo 級学習（このリポジトリの既存ウェイトはこれで学習：データ 15% サブセット、~1.2 時間 @ M1 MPS）
python scripts/train_sku110k.py --model yolov8n.pt --epochs 15 --fraction 0.15 --batch 8
```

スクリプトのパラメータ（`scripts/train_sku110k.py`）：

| パラメータ | デフォルト | 説明 |
|------|------|------|
| `--model` | `yolov8n.pt` | ベースウェイト（n が最速、s/m は精度高い）。初回は自動ダウンロード |
| `--epochs` | 15 | 学習ラウンド数。全データなら 50+ 推奨 |
| `--imgsz` | 640 | 学習解像度（推論時は `DETECT_IMGSZ=1280/1920` を使う） |
| `--batch` | 16 | GPU メモリ/MPS メモリが足りなければ下げる（M1 実測 8 が安定） |
| `--fraction` | 1.0 | 各ラウンドで使う訓練セットの割合；0.15 = 高速検証 |
| `--device` | auto | `cuda` / `mps` / `cpu` |
| `--patience` | 10 | 早期停止：10 ラウンド改善しなければ停止 |

**初回実行**時に Ultralytics が SKU-110K データセット（**約 13 GB**）を
`datasets/` に自動ダウンロードする——時間がかかるので辛抱強く待って；以降の再学習はローカルキャッシュをそのまま使う。

### 8.3 学習中に何が起こるか

成果物は全て `runs/detect/sku110k/` に入る：

```
runs/detect/sku110k/
├── results.csv        # ラウンドごとの指標（loss、P、R、mAP50、mAP50-95）
├── results.png        # loss/精度カーブ
├── BoxPR_curve.png    # 精度-適合率カーブ
├── confusion_matrix*.png
├── train_batch0.jpg   # 学習サンプレの可視化（データ拡張の効果を見る）
├── val_batch0_pred.jpg# 検証セット予測の可視化（目視でボックス品質を確認）
└── weights/
    ├── best.pt        # ★ 検証セットで最良のウェイト（使うのはこれ）
    └── last.pt        # 最終ラウンドのウェイト
```

**指標の見方**（初心者向け）：

- **P（precision 精度）**：検出した枠のうち何割が正しいか。0.85 = 100 個検出して 85 個正しい。
- **R（recall 適合率）**：実際の商品の中で何割が検出されたか。0.76 = 24% を見落としている。
- **mAP@50**：IoU 閾値 0.5 での平均精度。密集検出の「総合点」。>0.8 ならボックスが多くて商品に概ねぴったり。
- **mAP@50-95**：IoU 0.5~0.9 の全レンジの平均で、**ボックスのぴったり度**をより厳しく測る。
  このリポジトリの既存ウェイトは 0.473——まだ改善の余地（8.5 参照）。

### 8.4 このリポジトリで既に完了した 1 回の学習（再現可能）

| 項目 | 値 |
|----|----|
| ベースモデル | yolov8n.pt（事前学習初期化） |
| データ | SKU-110K、`fraction=0.15`（各ラウンドで訓練セットの 15%） |
| 超パラメータ | 15 epochs, imgsz 640, batch 8, patience 10 |
| ハードウェア | M1 MPS、約 1.2 時間 |
| 結果 | **P = 0.852 / R = 0.761 / mAP@50 = 0.818 / mAP@50-95 = 0.473** |
| ウェイト | `weights/sku110k_best.pt` にコピー済み、app が自動有効化 |

### 8.5 学習後のデプロイ + 品質向上のしかた

```bash
# 1) 最良のウェイトを標準位置へ（app がこれを見つけると auto バックエンドを自動有効化）
cp runs/detect/sku110k/weights/best.pt weights/sku110k_best.pt

# 2) 起動して目視で検証
python app.py

# 3) 定量的検証：YOLO-World / auto とボックス品質を比較
python scripts/compare_detectors.py data/shelf.jpg
python scripts/verify_detection.py
# 期待値：custom はボックス数が多く、ボックス中央値サイズが小さい（単品にぴったり）；
#         単品クローズアップは auto のフォールバック経路で有用な枠が出る
```

自動有効化したくない場合は明示的に指定もできる：

```bash
export DETECTOR=custom
export YOLO_WEIGHTS=runs/detect/sku110k/weights/best.pt
python app.py
```

**さらなる品質向上なら、費用対効果の順に**：

1. **全データ + ラウンド増**：`--fraction 1.0 --epochs 50`（最も直接的な効果、R と mAP が共に上昇）；
2. **より大きなモデル**：`--model yolov8s.pt`（m も）でボックスのぴったり度（mAP@50-95）を交換；
3. **CUDA マシン**：M1 MPS では batch 8 までしか開けないが、CUDA なら batch 16+ で学習が速くて安定；
4. **推論側**：4K 密集棚は `DETECT_IMGSZ` を 1920 まで上げる（フレームレートの低下に注意）。

> 注意：デテクタは「枠の品質」だけを決定し、**認識が正しくなるかは決定しない**——認識が正しいかは
> CLIP + 商品 DB が決める（§11 の本番運用提案参照）。

---

## 9. デバッグガイド：不具合時に最初に見る場所

### 9.1 「No products detected」——必ず検出の問題

「検出されない」と「認識されない」は別物。まず `debug_detect.py` で段階的に特定する：

```bash
python scripts/debug_detect.py data/shelf.jpg                # デフォルトデテクタ（auto）
python scripts/debug_detect.py data/shelf.jpg --conf 0.05    # 閾値を下げて多め検出
python scripts/debug_detect.py data/shelf.jpg --detector coco  # 比較（通常 0 枠で、クラス問題であることを示す）
python scripts/debug_detect.py data/shelf.jpg --imgsz 1920   # 4K 密集棚の解像度を上げる
```

このスクリプトがやること：① 検出した枠の数**と `specialist` 経路か `fallback` 経路かを出力**
（0 個 → デテクタの問題、指示に従って imgsz/conf を調整するか custom を学習）；
② **すべての**生検出枠を `data/debug_boxes.jpg` に描画（デテクタが「見ている」ものを直接確認）；
③ 商品 DB が空でなければ、app と完全同一の `rank_boxes()`（マルチスケールクロップ込み）で各枠の
最良マッチ SKU + 類似度を出力し、`MATCH_THRESHOLD` をどこに調整すべきかも教えてくれる（検出は正常なのに全
Unknown → 検索/閾値/参照写真品質の問題で、検出の問題ではない）。

#### 「手持ちの単品をカメラに向けても枠が出ない」

最も誤診されやすいケース。症状と原因（§7.1 参照）：

| 現象 | 意味 | 対応 |
|------|------|------|
| 枠 0 個 | 自学習 SKU-110K モデルにとって「大きな単品 1 個」は out-of-distribution | `DETECTOR=auto`（デフォルト）を維持；YOLO-World フォールバックへの自動切替は既に有効 |
| 10px の小枠が数十~数百個、最大の枠が映像の <1% | 同上。**conf を下げたことで強いて出たハルシネーション枠**で、本当の検出ではない | `conf` を片端から下げるな；`DETECTOR` が `custom` に設定されていないか確認 |
| 大きな枠が 1 個あるがラベルは赤い Unknown、スコア 0.55–0.65 | 検出は既に正しい。「ぴったりクロップ vs 参照写真まるごと」のスケール不一致 | デフォルトの `MATCH_SCALES=1.0,1.2` を維持（§7.2） |

このチェーンをワンクリックで再現（実際に CLIP 検索を実行する。`data/catalog.sqlite` にその SKU が登録済みであること）：

```bash
python scripts/test_live_closeup.py                    # デフォルトは ALKALINE 参照画像
python scripts/test_live_closeup.py data/crops/xxx.jpg # 自分のものに差し替え
python scripts/verify_detection.py                     # バッチ：custom vs world vs auto
python scripts/eval_match.py                           # 検索スケール/閾値の評価
```

### 9.2 「1 つの商品に複数の枠が出る」

学習済み単一クラスモデルは同じ商品に半セルずれた重複枠（IoU≈0.4）を出す。デフォルト
`DETECT_IOU=0.3` で処理済み；それでも残る場合は `sweep_detect.py` で
imgsz × conf × NMS-iou の組み合わせをスキャンし、棚領域 3 つの拡大比較図を `data/sweep/` に出力：

```bash
python scripts/sweep_detect.py data/shelf.jpg
```

### 9.3 その他の常用コマンド

```bash
python scripts/smoke_test.py                 # 検索閉ループの自己チェック（赤/青の偽 SKU 2 つ）
python scripts/compare_detectors.py data/shelf.jpg   # 2 つのデテクタのボックス品質比較
```

---

## 10. 設定項目（全て環境変数、`shelf_demo/config.py`）

| 変数 | デフォルト | 説明 |
|------|------|------|
| `DETECTOR` | `weights/sku110k_best.pt` 存在時 `auto`、それ以外 `world` | `auto` / `custom` / `world` / `coco`（§7） |
| `DETECT_IMGSZ` | `1280` | 検出推論の解像度；4K 密集棚は `1920` |
| `DETECT_CONF` | `world` で 0.05、それ以外 0.2 | 検出確信度閾値（§7 の表で差分の理由を説明）。`auto` 下ではこれは**自学習モデル**の閾値 |
| `DETECT_IOU` | `world` で 0.5、それ以外 0.3 | NMS の重複除去閾値 |
| `DETECT_MAX_DET` | `1000` | 画像 1 枚あたりの最大ボックス数（棚は非常に密集） |
| `DETECT_CLASS_AGNOSTIC` | `1` | クラス横断 NMS（YOLO-World の複数プロンプトの重複枠に必須） |
| `YOLO_WEIGHTS` | `weights/sku110k_best.pt`（存在時）、それ以外 `yolov8n.pt` | `coco`/`custom` のウェイトパス |
| `YOLO_WORLD_WEIGHTS` | `yolov8s-worldv2.pt` | YOLO-World ウェイトパス（`auto` のフォールバックモデル） |
| `YOLO_WORLD_PROMPTS` | 小売/パッケージ語 20 個（`product,package,...,object`） | オープンボキャブラリプロンプト（カンマ区切り） |
| `YOLO_WORLD_CONF` | `0.05` | フォールバックモデル固有の確信度（`DETECT_CONF` と分離） |
| `YOLO_WORLD_IOU` | `0.5` | フォールバックモデル固有の NMS IoU |
| `AUTO_COVERAGE_MIN` | `0.15` | `auto`：自学習モデルのボックス和集合が映像の ≥ この割合をカバーして初めて「密集棚」とみなす（§7.1） |
| `AUTO_MIN_AREA` | `0.01` | `auto`：自学習モデルの最大ボックスが ≥ この割合で初めて信頼に足す |
| `AUTO_SWITCH_AREA` | `0.05` | `auto`：フォールバックモデルが ≥ この割合の大きな物体を見つけて初めて採用される |
| `CLIP_MODEL` / `CLIP_PRETRAINED` | `ViT-B-16` / `laion2b_s34b_b88k` | embedding モデル。**モデル変更時は全商品の再登録が必要**（ベクトル空間が変わる）。より強ければ `ViT-B-16-SigLIP` + `webli` を試せる |
| `EMBED_DIM` | `512` | ベクトル次元（実際にロードした CLIP モデルに合わせて自動補正） |
| `MATCH_THRESHOLD` | `0.65` | コサイン類似度がこの値未満 → Unknown（ViT-B-16 でキャリブレーション済み） |
| `SEARCH_TOPK` | `5` | ボックスごとに top-k の候補 SKU を返す |
| `MATCH_SCALES` | `1.0,1.2` | クエリクロップのスケール（カンマ区切り）、SKU ごとに最高スコアを採用；`1.0` = マルチスケール無効（§7.2） |
| `EMBED_BATCH_SIZE` | `32` | CLIP 1 フォワードあたりの画像数（4K 棚の数百クロップをチャンク分割してメモリ制限） |
| `SHELF_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `SHELF_DATA_DIR` | `./data` | ベクトル DB / SQLite / 写真のルートディレクトリ |

Live ページの追跡/投票の env スイッチ（`shelf_demo/config.py`）：`SHELF_TRACKER`
（デフォルト `bytetrack.yaml`；`botsort.yaml` で他の ultralytics トラッカーに切替；
`off` で従来の貪欲 IoU マッチングに回退）、`CONFIRM_FRAMES=3` /
`VOTE_WINDOW=5` / `VOTE_MIN=3`（SKU 投票の閾値）、
`MATCH_THRESHOLD_KEEP=0.55`（確定済みラベルのヒステリシス保持閾値）、
`REEMBED_INTERVAL=30` / `REEMBED_MAX_PER_FRAME=4`（定期再検証の頻度）、
`EMBED_MAX_PER_FRAME=8`（1 フレームあたり CLIP に入る最大**枠**数；未確定トラックは毎フレーム
再エンコードが必要で、この上限が無ければ「棚の商品が全部 DB に無い」状態でフレームレートが <1 FPS まで落ちる）。
フォールバック経路の枠スコアは自学習モデルより全体的に低く、`LiveRecognizer` はそのフレームで
ByteTrack の新規トラック閾値を自動的に `YOLO_WORLD_CONF` まで下げる。さもなくば枠は低確信度枠として扱われ、永遠に
トラックが作れない——フォールバックが無駄になることになる。
その他の組み込み定数（`shelf_demo/live.py`、env を通さない）：トラック喪失の許容 `MAX_MISSES=4` フレーム、
セッション放置回収 `SESSION_TTL_S=120` 秒、フロントの imgsz 選択範囲 480–1536；
grasp 準備 `GRASP_WINDOW=5` フレーム / `GRASP_IOU=0.9` / `GRASP_MAX_AGE_MS=1000` ms
（意味は §5.3 ロボットアーム統合を参照）。

---

## 11. 重要な設計判断（なぜこう書いたか）

### 11.1 ベクトル検索は FAISS でなく NumPy なぜか？

macOS では `faiss-cpu` と `torch` がそれぞれ OpenMP ランタイム（libomp）を 1 つずつバンドルしており、
同一プロセスで同時にロードすると `OMP: Error #15` が報告され、Gradio のマルチスレッド配信下では**断続的にセグフォ**。
`KMP_DUPLICATE_LIB_OK=TRUE` はエラーの抑制だけで、公式は「クラッシュするか静かに誤った計算をする可能性がある」と明言しており、信頼できない。

Demo 規模（数千個の 512 次元ベクトル）の近傍探索は 1 回の行列積で足り、NumPy がミリ秒未満で完了し、
ネイティブ依存の競合はゼロ。`VectorStore` の API は意図的に
標準ベクトル DB の形態 `add / search / delete` にしている——規模が上がる（10 万+ SKU）時に FAISS / Milvus / Qdrant に
差し替えるだけでよく、上位コードは無変更。

### 11.2 Live ページと Catalog ページのフロントエンドが自作である理由

- **Live**：Gradio 6 の `gr.Image(streaming=True)` は内部の状態マシンが制御不能
  （停止ボタンがリセットされず、セッションが終了しない——ソースレベルで app レイヤでは修正不能と確認）。自作にすると：
  フロントの canvas が独立して 15 FPS で再描画（検出が遅くても映像が滞らない）、generation カウンタで
  停止後の遅れて着いたレスポンスを破棄、カメラのホットスワップでストリームが切れない。
- **Catalog**：Gradio ネイティブテーブルは「写真サムネイル + 行内削除ボタン」の操作を収容できないため、
  `gr.HTML` で純 JS テーブルを注入し、データは 2 つの JSON ルートを介する。
- 両者共通の坑：Gradio の `gr.HTML` テンプレート文字列に**`${` が出てはならない**
  （テンプレート補完スロットとみなされてしまう）ため、JS は一貫して文字列連結 / DOM API を使う。

### 11.3 Live セッションが 2 つのロックを使う理由

`_map_lock` はセッション dict 自体を保護；`model_lock` はフレーム処理全体（トラック更新 + MPS 推論
実行）の間に保持——MPS は並行推論を受け付けない。2 つのロックは**絶対にネストしない**。さもなくば
`handle_reset` 内で自己デッドロック（初版で踏んだ。faulthandler で特定。設計ドキュメント参照）。

---

## 12. Demo から本番へ：まだ必要なこと

この Demo は**ゼロショット CLIP** を使う。チェーンは完結しているが、細粒度な SKU 区別には限界がある。本番システムの発展方向：

1. **embedding モデルのファインチューニング**：ゼロショット CLIP は小売データで top-1 が ~40% しかなく、
   小売画像でファインチューニングすると ~89–92% に到達可能。**SigLIP / DINOv2** を backbone に差し替えも可。
2. **細粒度の曖昧性除去**：同型でスペック（容量など）が異なるもの（500ml vs 750ml）が最も壊れやすく、
   本番では**第二段階の reranker / キーポイント（keypoint）マッチ / OCR（パッケージ文字の読み取り）**を追加するのが常。
3. **ベクトル DB 昇格**：10 万+ SKU なら FAISS / Milvus / Qdrant に ANN を導入
   （`VectorStore` インターフェースは既にアライメント済み、そのまま差し替え可）。
4. **アノテーションの自動生成**：Grounding DINO + Autodistill で棚画像をゼロショットアノテーションし、
   蒸留して小さな YOLO を学習——人手を節約。
5. **VLM フォールバック**：検索確信度の低いクロップを生成型 VLM（Qwen-VL / GPT-4o）に渡して二次確認。

> 生成型 VLM は現在、棚認識の本番運用の主流は**ではない**（コスト高、遅延大、SKU レベルでハルシネーション）、
> 主に「データ生成」と低確信度のフォールバックに使われる。

---

## 12.5 Eye-in-hand ロボットアームの grasp（RGB-D 拡張、進行中）

目標形態：ロボットドッグがロボットアームを背負って棚の前で商品の grasp/陳列を行い、深度カメラ（RealSense）は手首に装着。
設計ドキュメント：**`docs/plans/2026-08-27-eye-in-hand-design.md`**。
認識レイヤ（YOLO+CLIP、2D 安定ウィンドウ）は完全に変更なし。新規に幾何レイヤを追加：

| モジュール | 責務 |
|------|------|
| `shelf_demo/transforms.py` | SE3 数学：変換チェーン `P_base = T_base_ee·T_ee_cam·P_cam` |
| `shelf_demo/camera.py` | RealSense のフレーム取得（深度を RGB にアライン、メートル変換、内パラメータ読取） |
| `shelf_demo/pose3d.py` | 2D 枠 + 深度 → カメラ座標系の 3D ターゲット点（中央値フィルタ） |
| `shelf_demo/calibration.py` | 手眼キャリブレーション：純 numpy の Park 解法（cv2≥5 で calibrateHandEye が削除されたため） |
| `shelf_demo/rgbd_live.py` | `RGBDGraspSession.grasp3d(name)` → メートル座標 |
| `shelf_demo/robot.py` | `ArmBase` インターフェース + `MockArm` |

```bash
python scripts/test_transforms.py            # SE3 数学の自己チェック（秒オーダー、ハード不要）
python scripts/test_rgbd_grasp.py            # 全チェーン合成データテスト（同上、秒オーダー）
python scripts/calibrate_handeye.py simulate # 手眼キャリブレーションの自己チェック（ハード不要）
python scripts/rs_live.py --target "コカ・コーラ"     # 実機ループ：ターゲット 3D をリアルタイム出力
# アーム到着後：collect（10+ 組の取得）→ solve → data/handeye.json、
# rgbd_live の point_base_m は即座に使えるようになる
```

---

## 13. 開発とテスト

```bash
# grasp 準備インターフェース（ロボットアーム）の単体テスト：偽 pipeline で LiveRecognizer を駆動、モデル不要、秒オーダー
python scripts/test_grasp.py

# 検出：自学習 / YOLO-World / auto の 3 バックエンドを「密集棚 + 単品クローズアップ 17 枚」で比較
python scripts/verify_detection.py
# 検索：マルチスケールクロップの効果と閾値の評価
python scripts/eval_match.py
# 単品クローズアップの全チェーン（検出→追跡→CLIP→ラベル）、catalog にその SKU が登録済みであること
python scripts/test_live_closeup.py

# 開発依存（app 実行には不要）
pip install playwright && playwright install chromium-headless-shell

# Live ページ E2E：Playwright の偽カメラで、開始/停止/再認識/フレームストリームの継続性/カメラホットスワップを検証
python scripts/e2e_live_ui.py

# Catalog ページ E2E：テスト SKU を実際に登録 → テーブル描画/サムネイル/削除（写真ファイルの掃除込み）を検証
python scripts/e2e_catalog_ui.py
```

設計の進化記録は `docs/plans/2026-08-12-realtime-video-annotation-design.md`
（実測タイムテーブル、Gradio ストリーミング問題のソースレベル診断を含む）を参照。
