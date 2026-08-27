# Eye-in-hand 抓取设计（2026-08-27）

## 需求

最终形态：**机器狗驮着机械臂在超市货架前理货**。第一阶段先做"**取下商品**"，
理货放回（需要对排面空间的力与空间感知）作为第二阶段。

感知侧沿用现有 demo：YOLO 检测（哪里有商品）+ CLIP 检索（是哪个 SKU）。
本设计只解决 **eye-in-hand 几何与抓取链路**——深度相机装在机械臂腕部，
相机跟着手动，狗怎么动都不影响抓取闭环。

## 为什么是 eye-in-hand

| 方案 | 相机装哪 | 问题 |
|------|---------|------|
| eye-to-hand | 狗头/货架固定 | 狗每挪一次要重标外参；手腕遮挡不可见；近景无法精对准 |
| **eye-in-hand** | 机械臂腕部 | 标定一次固定不变；可主动凑近看；相机→目标关系经臂正解闭环 |

代价：必须做**手眼标定**，求出 `T_ee_cam`（相机系→末端法兰系的固定变换）。
标定板动或相机松动都要重标（拆装夹具后立即重标）。

## 坐标系约定

```
        P_base = T_base_ee(t) · T_ee_cam · P_cam

world       世界系（狗建图/导航用，本阶段基本不直接用）
base        机械臂基座系（SDK 以它为准）
ee          末端法兰系        T_base_ee 由臂正解（SDK）每个采样时刻给出
cam         相机光学系        T_ee_cam 由手眼标定得到，固定
board       标定板系          仅标定时使用，固定于桌上
```

- 单位统一**米**；RealSense 的 uint16 深度乘出厂 depth scale 转 float32 米。
- 每次拍摄记录**当时的** `T_base_ee`，与该帧深度配对使用（臂动中拍照必须
  注意时戳，MVP 阶段伺服时臂静止拍照，规避同步问题）。
- 抓取期间狗从站立转为锁死姿态，臂基座可视为不动 → `P_base` 恒定，
  狗本体与 SLAM 不进抓取闭环。

## 模块划分（与现有代码的关系）

现有识别层（`detector / embedder / database / pipeline / live`）**一行不改**，
2D 抓取稳定窗口（`GRASP_WINDOW/GRASP_IOU`）也保留——伺服收敛判定直接复用。
新增的是其下/其旁的几何层：

```
┌─ eye-in-hand 几何层（新增，全部 numpy / 无副作用） ─────────────┐
│ transforms.py   SE3 数学：rotvec↔R、compose、invert、点变换     │
│ pose3d.py       2D 框 + 深度图 + K → 相机系 3D 目标点           │
│ camera.py       RealSense 采帧（pyrealsense2 懒加载）           │
│ calibration.py  手眼标定样本管理 + calibrateHandEye 解算        │
│ rgbd_live.py    RGB-D 抓取会话：LiveRecognizer + latest Frame   │
│                 + arm FK → /grasp 3D 版输出                     │
│ robot.py        ArmBase 接口 + MockArm（真机 SDK 到了再实现）   │
└──────────────────────────────────────────────────────────────┘
         ▲ 复用                            ▲ 复用
   live.LiveRecognizer(2D 稳定窗口)   pipeline.ShelfPipeline(检测+识别)
```

| 文件 | 职责 |
|------|------|
| `shelf_demo/transforms.py` | SE3：`rotvec_to_R/R_to_rotvec`、`compose/invert`、`transform_points`。纯 numpy，可单测 |
| `shelf_demo/camera.py` | `Frame(rgb,u8 / depth_m,f32 / K / ts)` + `RealSenseCamera`（对齐 depth→color、读内参、depth scale 转米） |
| `shelf_demo/pose3d.py` | `deproject`、框中心 ROI 中值深度 → `target_point()`。无效深度返回 None |
| `shelf_demo/calibration.py` | `HandEyeCalibrator.add_sample(T_base_ee, T_cam_board)`、`solve()`；迁到一半发现 **cv2 ≥ 5.0 删掉了 `calibrateHandEye`**（只留下 CALIB_HAND_EYE_* 常量），故改为纯 numpy 的 Park 式解（四元数零空间最小二乘），cv2<5 环境可用 `method="cv2:Tsai"` 交叉验证 |
| `shelf_demo/rgbd_live.py` | `GraspSession.process(frame, fk)`；`grasp3d(name)` 输出 `point_cam_m` / `point_base_m` / 深度质量统计 |
| `shelf_demo/robot.py` | `ArmBase.fk()` 接口 + `MockArm`（测试/仿真链路占位） |
| `scripts/rs_live.py` | 真机 headless 环：相机→识别→`grasp3d` 循环打印目标位（伺服 PC 端入口） |
| `scripts/calibrate_handeye.py` | ChArUco 采集 + 解算；`--simulate` 无硬件自检标定数学 |
| `scripts/test_transforms.py` / `test_rgbd_grasp.py` | 纯合成数据单测（无模型、无相机、秒级跑完） |

## 数据流（一次抓取）

```
RealSense 对齐帧 ──► Frame(rgb, depth_m, K, ts)
   rgb ──► GraspSession.process(): YOLO 检测 + IoU 跟踪 + CLIP 认新目标（现成）
   grasp3d("可乐"):
     2D 稳定窗口(GRASP_WINDOW 帧 IoU≥0.9) 通过? ──否──► reason
        │是
        ▼
     稳定框中心 ROI 中值深度 ──► deproject ──► P_cam（稳健，带 valid_px 计数）
        ▼  乘该帧 T_ee_cam、T_base_ee
     P_base ──► 臂控（预抓取位 → 直线插补 → 夹取 → 验证）
```

ROI 取框中心 `TARGET3D_ROI_SHRINK`(0.4) 收缩区域，过滤 0 / 超量程像素，
中值抗边缘飞点；`valid_px` 太低（< 8）直接报不可抓而不是给野值。

## 标定流程（设备到齐后 15 分钟）

1. `scripts/collect fk-server`（或用户自己的臂侧脚本）实时把 `T_base_ee`
   写入 `data/fk.json`；`calibrate_handeye.py collect` 每按一次回车采一对
   (ChArUco 板位姿, 该时刻 FK)，要求 ≥10 个姿态且各姿态旋转差 >15°；
2. `calibrate_handeye.py solve` 解 `T_ee_cam`，打印 AX=XB 残差，写
   `data/handeye.json`；残差大则续采（`--append`）；
3. `calibrate_handeye.py --simulate` 在无任何硬件时用合成数据验证解算
   闭环（已含在自检里）。

## 伺服策略（两段式，本期只做接口预留，控制循环留到真机）

1. **Look-then-move**：中距离拍一帧 → `grasp3d` → 臂走到目标前方
   ~12 cm 的预抓取位（P_base 换算）；
2. **近景闭环**：凑近后重拍重定位、小步修正（深度 20~40 cm 内
   RealSense 精度最好），对准后直线插补到位 → 夹取 → 抬起验证
   （力/电流或视觉复查）。

狗侧只负责：导航到货架（自带 SLAM）→ 站立锁死 → 触发抓取状态机，
全程不进入视觉闭环。货架对接定位可用 AprilTag 辅助（后续）。

## 风险/已知取舍

- **算力位置未定**：机载 Jetson 还是 WiFi 回传。本期接口与位置无关；
  回传方案要评估货架区 WiFi 时延。
- **特写注册**：eye-in-hand 会拍近距离大角度图，入库参考图需多角度，
  否则 CLIP 相似度掉线（识别层无需改，只补注册习惯）。
- **时戳同步**：臂动中拍照会有 FK-深度错位；MVP 静止拍照规避，
  后续上硬件时间戳对齐。
- **装夹刚性**：相机夹具松动 = 标定作废，夹具上留定位销。

## 落地顺序

1. ✅ transforms / pose3d / camera / calibration / rgbd_live + 合成数据单测（本期）
2. ⬜ RealSense 实采：手持当"假 eye-in-hand"，验证目标点稳定性（有相机就能测）
3. ⬜ 臂到货：跑标定脚本 → `rs_live.py` 出 `point_base_m`
4. ⬜ 伺服控制循环 + 状态机（真机调速度与安全）
5. ⬜ 狗到货：导航对接 + 任务编排；理货"放回"第二阶段
