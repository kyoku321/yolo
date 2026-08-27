"""Robot (arm) abstraction for the eye-in-hand stack.

The perception pipeline only needs ONE thing from the arm at this stage:
forward kinematics, i.e. T_base_ee at the moment a camera frame was taken.
Motion commands (pre-grasp pose, linear approach, grip) are added when the
real arm SDK lands — yours should implement this same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ArmBase(ABC):
    """Minimal interface between perception and any 6/7-DoF arm SDK."""

    @abstractmethod
    def fk(self) -> np.ndarray:
        """Forward kinematics NOW: 4x4 T_base_ee (metres, arm base frame)."""


class MockArm(ArmBase):
    """In-code stand-in: a fixed pose that tests can move deterministically.

    Records fk() calls so tests can assert the pipeline actually queries the
    arm once per accepted frame (no hidden caching).
    """

    def __init__(self, T_base_ee=None) -> None:
        self.T = np.eye(4) if T_base_ee is None else np.asarray(T_base_ee,
                                                                dtype=float)
        self.fk_calls = 0

    def set_fk(self, T_base_ee) -> None:
        self.T = np.asarray(T_base_ee, dtype=float)

    def fk(self) -> np.ndarray:
        self.fk_calls += 1
        return self.T.copy()
