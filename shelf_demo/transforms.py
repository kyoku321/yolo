"""SE(3) rigid-transform math for the eye-in-hand stack (numpy only).

Conventions (see docs/plans/2026-08-27-eye-in-hand-design.md):
  - poses are 4x4 homogeneous matrices named T_<from>_<to> in docstrings by
    their *indices*, i.e. T_a_b maps points expressed in frame b into
    frame a:  P_a = T_a_b @ P_b.
  - the transform chain for grasping is
        P_base = T_base_ee @ T_ee_cam @ P_cam
    where T_base_ee comes from the arm SDK (forward kinematics) and
    T_ee_cam from hand-eye calibration (calibration.py).
  - all lengths in metres, rotations as rotvecs (axis*angle, OpenCV style).

Everything here is pure numpy and unit-tested by scripts/test_transforms.py.
"""
from __future__ import annotations

import numpy as np


def rotvec_to_R(rotvec) -> np.ndarray:
    """Rotation vector (axis * angle) -> 3x3 rotation matrix (Rodrigues)."""
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def R_to_rotvec(R) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (inverse of rotvec_to_R)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # 180-degree rotation: axis from symmetric part of R + I
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]
        axis = axis / np.linalg.norm(axis)
        return theta * axis
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    return (theta / (2.0 * np.sin(theta))) * axis


def compose(rot, t) -> np.ndarray:
    """Build T from a rotation (3x3 R or length-3 rotvec) and translation."""
    rot = np.asarray(rot, dtype=np.float64)
    R = rotvec_to_R(rot) if rot.shape == (3,) else rot.reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (cheaper and exact vs inv())."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points) -> np.ndarray:
    """Apply T (4x4) to one 3D point (returns (3,)) or an (N,3) array."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts2 = pts.reshape(1, 3) if single else pts
    R, t = T[:3, :3], T[:3, 3]
    out = pts2 @ R.T + t
    return out[0] if single else out


def target_in_base(T_base_ee, T_ee_cam, p_cam) -> np.ndarray:
    """The full eye-in-hand chain: camera-frame point -> arm-base frame."""
    T = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
        T_ee_cam, dtype=np.float64)
    return transform_points(T, np.asarray(p_cam, dtype=np.float64))


def hstack(rot, t) -> np.ndarray:
    """Alias for compose(): from a JSON pair {"rotation", "translation"}."""
    return compose(rot, t)
