"""Hand-eye calibration (eye-in-hand): solve T_ee_cam.

Classical AX = XB: the calibration board rests on the table (fixed in the
arm's base frame), so collecting (T_base_ee_i, T_cam_board_i) pairs while
moving the arm through diverse poses lets you solve the constant
end-effector->camera transform T_ee_cam.

The solver is implemented in numpy (quaternion-averaging, Park-style):
opencv >= 5.0 REMOVED cv2.calibrateHandEye while keeping its constants, so
depending on it is not portable. If the function exists (cv2 < 5), pass
method="cv2:Tsai" (or Park/Horaud/Andreff/Daniilidis) to cross-check.

No hardware is needed to validate the math:
scripts/calibrate_handeye.py --simulate generates synthetic sample pairs and
checks the recovered transform against ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .transforms import R_to_rotvec, compose, invert, rotvec_to_R

CV2_METHODS = ("Tsai", "Park", "Horaud", "Andreff", "Daniilidis")


# ---- quaternion helpers (unit quat [w, x, y, z]) ------------------------------

def _R_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m21 - m12) / s,
                         (m02 - m20) / s, (m10 - m01) / s])
    if m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        return np.array([(m21 - m12) / s, 0.25 * s,
                         (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        return np.array([(m02 - m20) / s, (m01 + m10) / s,
                         0.25 * s, (m12 + m21) / s])
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2
    return np.array([(m10 - m01) / s, (m02 + m20) / s,
                     (m12 + m21) / s, 0.25 * s])


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_L(q: np.ndarray) -> np.ndarray:
    """Left-multiplication matrix: L(q1) @ q2 == q1 ⊗ q2."""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, -z, y],
                     [y, z, w, -x],
                     [z, -y, x, w]])


def _quat_R(q: np.ndarray) -> np.ndarray:
    """Right-multiplication matrix: R(q2) @ q1 == q1 ⊗ q2."""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, z, -y],
                     [y, -z, w, x],
                     [z, y, -x, w]])


# ---- the numpy solver ----------------------------------------------------------

def solve_handeye(samples: list["HandEyeSample"]) -> tuple[np.ndarray, float]:
    """Solve A X = X B over all motion pairs -> (T_ee_cam, residual).

    For each pair (i, j) of poses:
        A = inv(T_base_ee_i) @ T_base_ee_j   (arm motion, hand frame)
        B = T_cam_board_i @ inv(T_cam_board_j) (board motion, camera frame)
    Rotation: quat eigen-solution of L(qA) q = R(qB) q (nullspace of the
    stacked differences). Translation: stacked (R_A - I) t = R_x t_B - t_A.
    """
    if len(samples) < 3:
        raise ValueError(f"need >= 3 samples, have {len(samples)}")

    motions: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            A = invert(samples[i].T_base_ee) @ samples[j].T_base_ee
            B = samples[i].T_cam_board @ invert(samples[j].T_cam_board)
            if np.linalg.norm(R_to_rotvec(A[:3, :3])) > 1e-4:  # moved enough
                motions.append((A, B))
    if not motions:
        raise ValueError("samples never moved - not enough pose diversity")

    C = np.vstack([_quat_L(_R_to_quat(A[:3, :3])) - _quat_R(_R_to_quat(B[:3, :3]))
                   for A, B in motions])
    q = np.linalg.svd(C)[2][-1]          # smallest right singular vector
    q = q if q[0] >= 0 else -q
    R_x = _quat_to_R(q)

    rows, rhs = [], []
    for A, B in motions:
        rows.append(A[:3, :3] - np.eye(3))
        rhs.append(R_x @ B[:3, 3] - A[:3, 3])
    t_x, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs),
                              rcond=None)
    X = compose(R_x, t_x)
    return X, _axxb_residual(samples, X)


def _axxb_residual(samples: list["HandEyeSample"], X: np.ndarray) -> float:
    """RMS of ||rot(LHS)-rot(RHS)||(rad) + ||t_LHS - t_RHS||(m) per motion."""
    errs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            A = invert(samples[i].T_base_ee) @ samples[j].T_base_ee
            B = samples[i].T_cam_board @ invert(samples[j].T_cam_board)
            lhs, rhs = A @ X, X @ B
            d_r = np.linalg.norm(R_to_rotvec(lhs[:3, :3] @ rhs[:3, :3].T))
            d_t = float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]))
            errs.append(d_r * d_r + d_t * d_t)
    return float(np.sqrt(np.mean(errs))) if errs else 0.0


# ---- sample container / persistence --------------------------------------------

@dataclass
class HandEyeSample:
    T_base_ee: np.ndarray    # arm forward kinematics at capture time (4x4)
    T_cam_board: np.ndarray  # board pose seen by the camera at that time (4x4)


@dataclass
class HandEyeCalibrator:
    samples: list[HandEyeSample] = field(default_factory=list)

    def add_sample(self, T_base_ee, T_cam_board) -> int:
        self.samples.append(HandEyeSample(
            np.asarray(T_base_ee, dtype=np.float64),
            np.asarray(T_cam_board, dtype=np.float64)))
        return len(self.samples)

    def solve(self, method: str = "park") -> tuple[np.ndarray, float]:
        """method="park" (bundled numpy solver) or "cv2:<Tsai|Park|...>"."""
        if method.startswith("cv2:"):
            return self._solve_cv2(method[4:])
        if method != "park":
            raise ValueError("method must be 'park' or 'cv2:<NAME>'")
        return solve_handeye(self.samples)

    def _solve_cv2(self, name: str) -> tuple[np.ndarray, float]:
        import cv2
        fn_name = f"CALIB_HAND_EYE_{name.upper()}"
        if not hasattr(cv2, "calibrateHandEye") or not hasattr(cv2, fn_name):
            raise RuntimeError(
                f"this opencv ({cv2.__version__}) has no calibrateHandEye "
                f"(removed in cv2>=5) - use method='park' instead")
        R_g2b = np.stack([s.T_base_ee[:3, :3] for s in self.samples])
        t_g2b = np.stack([s.T_base_ee[:3, 3] for s in self.samples])
        R_t2c = np.stack([s.T_cam_board[:3, :3] for s in self.samples])
        t_t2c = np.stack([s.T_cam_board[:3, 3] for s in self.samples])
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                    method=getattr(cv2, fn_name))
        X = compose(R, t.reshape(3))
        return X, _axxb_residual(self.samples, X)

    def save_samples(self, path) -> None:
        data = [{"T_base_ee": s.T_base_ee.tolist(),
                 "T_cam_board": s.T_cam_board.tolist()} for s in self.samples]
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def load_samples(cls, path) -> "HandEyeCalibrator":
        cal = cls()
        for d in json.loads(Path(path).read_text()):
            cal.add_sample(d["T_base_ee"], d["T_cam_board"])
        return cal


# ---- the solved result ----------------------------------------------------

def save_handeye(path, T_ee_cam: np.ndarray, meta: dict | None = None) -> None:
    """Persist the solved hand-eye transform as the runtime config file."""
    payload = {
        "rotation": R_to_rotvec(np.asarray(T_ee_cam)[:3, :3]).tolist(),
        "translation": np.asarray(T_ee_cam)[:3, 3].tolist(),
        "meta": meta or {},
    }
    Path(path).write_text(json.dumps(payload, indent=1))


def load_handeye(path=None) -> np.ndarray:
    """Load T_ee_cam (4x4). Raises when calibration was never run."""
    path = Path(path) if path else config.HANDEYE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/calibrate_handeye.py first")
    d = json.loads(path.read_text())
    return compose(np.asarray(d["rotation"]), np.asarray(d["translation"]))


def fk_json_to_T(data: dict) -> np.ndarray:
    """Accept {"rotation": rotvec|3x3, "translation": [x,y,z]} -> 4x4."""
    rot = np.asarray(data["rotation"], dtype=np.float64)
    R = rotvec_to_R(rot) if rot.reshape(-1).shape == (3,) else rot.reshape(3, 3)
    return compose(R, np.asarray(data["translation"], dtype=np.float64))
