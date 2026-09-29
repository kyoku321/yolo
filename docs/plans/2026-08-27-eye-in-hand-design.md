# 手首マウント RGB-D（eye-in-hand）+ ロボットアーム — 設計書

**日付**: 2026-08-27
**状態**: 完了（M1–M5a 実施済み、アーム実機待機）
**対象**: `/Users/kyoku/Documents/IC/yolo`（supermarket-shelf-product-recognition）

---

## 1. 背景と目標

### 1.1 システム構成

- **ロボットドッグ**（本体、移動）
- **ロボットアーム**（6 軸、手首に RealSense 深度カメラ装着）
- **RealSense D435i**（RGB + 深度、手首マウント = **eye-in-hand**）
- **既存認識スタック**（`shelf_demo/`：YOLO + CLIP ベクトル検索、ライブ動画注釈）

ドッグは既に **OBSBOT Meles カメラ + 深度**で動作中（棚撮影用、eye-to-hand）。今回の拡張はアームの
手首カメラ（eye-in-hand）で、アームの「掴む / 陳列する」動作を可能にする。

### 1.2 目標

1. RealSense から **RGB + 深度（メートル単位）+ 内パラメータ** を読み取る
2. 2D 検出枠 + 深度 → **カメラ座標系の 3D 座標**へ（`x, y, z` メートル）
3. **手眼キャリブレーション**（手首フレーム ↔ カメラフレームの `T_ee_cam`）を解く
4. `T_base_ee(t)`（aBot アーム API から読み取り）と合成 → **ベース座標系の 3D ターゲット点**
5. 既存の `grasp?name=` 2D 安定ウィンドウをそのまま 3D 版に昇格
6. **ゼロ新規ネイティブ依存**（opencv 5 の `calibrateHandEye` 削除に備え、手眼キャリブレーションは純 NumPy 実装）

### 1.3 非目標（今回のスコープ外）

- 軌道計画 / 逆運動学 / 衝突回避（aBot アーム SDK の役割）
- grasp 姿勢（quaternion）の推定——まずは**ターゲット点（point-to-point）**のみ；grasp 姿勢は
  商品カテゴリの固定グリッパオフセットで近似（§8）
- 複数物体の同時 grasp、棚の再陳列戦略

---

## 2. 座標系と幾何

### 2.1 フレーム定義

| フレーム | 原点 | +X | +Y | +Z | 由来 |
|------|------|----|----|----|------|
| `world`（ベース） | アームベース | 床面に沿う右方向 | 床面に沿う前方 | 上向き | aBot アーム API |
| `ee`（手首） | 手首ジョイント | アーム定義 | アーム定義 | グリッパの伸び方向（下向き） | aBot アーム API |
| `cam` | カメラ光軸 | OpenCV 慣例 | OpenCV 慣例 | 光軸（前方向） | RealSense / OpenCV |
| `board` | 標定板中心 | 標定板定義 | 標定板定義 | 法線方向 | ArUco |

### 2.2 合成チェーン

```
P_base = T_base_ee(t) · T_ee_cam · P_cam
```

- `P_cam`：対象点がカメラ座標系にある 3D 点（メートル）
- `T_ee_cam`：**手眼キャリブレーションの出力**（静的、一度解いたら JSON に永続化）
- `T_base_ee(t)`：アームの現在姿勢（aBot API の `get_ee_pose()` で t ごとに取得）

### 2.3 手眼キャリブレーション（eye-in-hand の Park 法）

標定板を固定し、アームを N 姿勢（10+）動かす。各姿勢 i で：

- `T_base_board,i`：4 枚の ArUco マーカーから最小二乗で解く
- `T_ee_cam,i`（既知・固定）
- `T_base_ee,i`：aBot API から読み取り

Park 方程式：`T_base_ee,i · T_ee_cam,i · T_ee_cam,1⁻¹ · T_base_ee,1⁻¹ = T_base_board,i · T_board,1⁻¹ · T_base_board,1`

左辺（LHS）を右辺（RHS）に等置して対数取ると線形システムになり、**純 NumPy** で解ける
（opencv 5 が `calibrateHandEye` を削除したので自前実装——scipy すら使わない）。

**出力**：`T_ee_cam`（4×4）、再投影誤差、`data/handeye.json` に保存。

---

## 3. アーキテクチャ

### 3.1 モジュール

```
shelf_demo/
├── transforms.py      # ★ 新規: SE3 数学 (NumPy only): compose/invert/log/exp/Park 法
├── camera.py          # ★ 新規: RealSense 取得 (RGB+深度アライン, メートル変換, 内パラメータ)
├── pose3d.py          # ★ 新規: 2D 枠 + 深度 → 3D (カメラ座標), メディアンフィルタ
├── calibration.py     # ★ 新規: 手眼キャリブレーション (ArUco 板 → T_ee_cam, Park 法純 NumPy)
├── rgbd_live.py       # ★ 新規: RGBDGraspSession — 既存の LiveRecognizer を包み、
│                      #   grasp?name= の 3D 版 (点 + 安定ウィンドウ, メートル単位)
├── robot.py           # ★ 新規: ArmBase 抽象 + MockArm (テスト/シミュレーション用)
├── (既存: pipeline.py / live.py / detector.py / embedder.py / database.py … 変更なし)
```

### 3.2 3D grasp のデータフロー

```
ブラウザ/スクリプト:  GET /live/api/grasp3d?name=Coke   (既存の 2D /live/api/grasp と並行)
        │
        ▼
RGBDGraspSession.grasp3d(name)
        │
        ├── 1) camera.py:       RGB + 深度フレームを取得 (メートル単位、深度→RGB アライン)
        ├── 2) 既存 LiveRecognizer.process(rgb)
        │      → YOLO 検出 → ByteTrack → CLIP 投票 → 2D ラベル + track_id
        ├── 3) 対象 track の box を取得
        ├── 4) pose3d.box_to_3d(box, depth_m):
        │      box 領域の深度ピクセルをサンプリング → メディアン深度 z
        │      P_cam = ((u-cx)/fx·z, (v-cy)/fy·z, z)   # 中央値で外れ値を抑制
        ├── 5) 安定ウィンドウ (既存の 2D IoU ウィンドウを拡張):
        │      直近 GRASP_WINDOW フレームの P_cam を保持、IoU + 深度変化の両方が安定でなければ None
        ├── 6) T_ee_cam · P_cam → P_ee (手首座標)
        └── 7) 任意 (アーム接続時): T_base_ee(t) · P_ee → P_base (ベース座標)
        → JSON: {ready, point_cam_m, point_ee_m, point_base_m, box, depth_m, ...}
```

---

## 4. 実装マイルストーン

| # | マイルストーン | 内容 | 検証 |
|---|------|------|------|
| M1 | `transforms.py` + 単体テスト | SE3 数学、Park 法、合成 | 合成データで手眼キャリブレーションが既知値に復元される |
| M2 | `camera.py` + 取得スクリプト | RealSense RGB/深度/内パラメータ | `scripts/rs_grasp3d_demo.py`：カメラ → 枠 → 3D 点の表示 |
| M3 | `pose3d.py` | 2D→3D、メディアン深度、外れ値除去 | 既知距離の物体で z の誤差 < 2% |
| M4 | `calibration.py` + 標定スクリプト | ArUco 板 → N 姿勢 → `T_ee_cam` | `scripts/calibrate_handeye.py`、再投影誤差報告 |
| M5 | `rgbd_live.py` + `/grasp3d` API | 2D 安定ウィンドウを 3D に昇格 | `scripts/test_grasp3d.py`：MockArm で E2E |
| M6 | アーム統合（**要実機**） | `ArmBase` の本物の実装、`T_base_ee` 読み取り | 実機で grasp |

---

## 5. キー設計判断

1. **ゼロ新規ネイティブ依存**：`transforms` / `pose3d` / `calibration` は純 NumPy。
   RealSense は `pyrealsense2`（既存 `scripts/rs_live.py` が既に使用）。
   OpenCV は ArUco 検出にのみ使用（opencv 5 で `calibrateHandEye` が削除されているため、
   手眼キャリブレーション本体は自前 NumPy 実装——scipy すら使わない）。
2. **既存認識レイヤに変更を加えない**：`LiveRecognizer` は RGB フレームの 2D 認識のみを提供；
   3D 化は外側のラッパー（`RGBDGraspSession`）で行う。
3. **深度の外れ値除去**：枠領域の深度ピクセルの**中央値**を採用（エッジ / 穴で平均は壊れる）。
   有効深度ピクセルが 30% 未満ならそのフレームは有効と見なさない。
4. **安定ウィンドウを 3D に昇格**：2D IoU の安定に加えて、**深度変化の上限**を追加
   （カメラが動いていてもターゲットが静止していることを保証）。
5. **手眼キャリブレーションの成果物を永続化**：`data/handeye.json`（T_ee_cam + 誤差 + タイムスタンプ）、
   起動時に自動ロード；アーム / カメラの再装着で再度実行。

---

## 6. 設定（環境変数、`shelf_demo/config.py` に追加）

| 変数 | デフォルト | 説明 |
|------|------|------|
| `RSD2_DEPTH_UNITS` | `1000` | 深度スケーリング（mm→m） |
| `GRASP3D_WINDOW` | `5` | 3D 安定ウィンドウのフレーム数 |
| `GRASP3D_MAX_DEPTH_DELTA` | `0.02` | 深度変化の上限（メートル） |
| `GRASP3D_MIN_VALID_PIXELS` | `0.3` | 枠領域の有効深度ピクセルの最小割合 |
| `HANDEYE_JSON` | `data/handeye.json` | 手眼キャリブレーション成果物パス |
| `ARUCO_DICT` | `4x4_50` | ArUco 辞書 |
| `ARUCO_MARKERS` | `0,1,2,3` | 標定板上の 4 枚のマーカー id |
| `ARUCO_MARKER_LEN_MM` | `40` | マーカー寸法（メートル変換に使用） |

---

## 7. テスト計画

| レベル | テスト | 説明 |
|------|------|------|
| 単体 | `test_transforms.py` | 合成 / 逆変換 / log-exp / Park 法（既知 T_ee_cam の復元） |
| 単体 | `test_pose3d.py` | 合成深度マップで 2D→3D（既知点の復元）、外れ値ロバストネス |
| 単体 | `test_handeye.py` | 合成データ（既知 T_ee_cam で N 姿勢生成）→ 手眼キャリブレーション実行 → 誤差検証 |
| 統合 | `test_grasp3d.py` | `MockArm` + 合成カメラで `/grasp3d` の E2E（モデル不要、秒オーダー） |
| E2E | `calibrate_handeye.py` | 実機 ArUco 板（または `--simulate` で合成データ） |
| 実機 | `rs_grasp3d_demo.py` | RealSense + 認識 + 3D 点表示 |

---

## 8. grasp 姿勢（次回）

点ターゲットが安定したら、grasp 姿勢の追加：

1. **カテゴリオフセット方式**（今回の拡張で実装可能）：`GRASP_OFFSETS = {"bottle": (0, 0, -0.05), ...}`
   を商品カテゴリに紐付け、グリッパのアプローチ方向は常に `-Z_ee`（手首下向き）
2. **法線推定方式**：深度クラウドの PCA で grasp 面法線を推定（後回し）

---

## 9. リスクと対策

| リスク | 対策 |
|------|------|
| 深度ノイズ（光沢パッケージ / 透明ボトル） | メディアン深度 + 有効ピクセル率フィルタ；悪ければ枠の中央 50% のみサンプリング |
| 手眼キャリブレーション精度不足 | 10+ 姿勢、姿勢間の回転差 30°+ を保証；再投影誤差を報告し閾値超過で警告 |
| アーム API の遅延（`T_base_ee` が古い） | `T_base_ee` と深度フレームのタイムスタンプを並べて、50 ms を超えれば警告 |
| RealSense のフレームドロップ | 取得タイムアウト + 直近フレームの再利用（2D トラッカーが既にギャップ耐性あり） |

---

## 10. 実装記録

- **M1–M5a 完了**（2026-08-27）：`transforms.py`（SE3 + Park）、`camera.py`、`pose3d.py`、
  `calibration.py`、`rgbd_live.py`、`robot.py`（`ArmBase` + `MockArm`）、
  `scripts/test_transforms.py` / `test_pose3d.py` / `test_handeye.py` /
  `test_rgbd_grasp.py` / `calibrate_handeye.py`（`--simulate` 付き）/ `rs_grasp3d_demo.py`。
  全ユニットテスト + 合成 E2E がグリーン（モデル / ハードウェア不要）。
- **待ち**：M4 実機標定（ArUco 板 + 10+ 姿勢）と M6 アーム実機（aBot SDK の `ArmBase` 実装）。
