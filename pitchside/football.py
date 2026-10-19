"""Pitch geometry and football models shared by the simulator and the engine.

Coordinate convention: metres, pitch 105 x 68, origin at the home team's
bottom-left corner. Home always attacks towards x = 105 (coordinates are
normalised per half, as most tracking providers deliver them).
"""
from __future__ import annotations

import math

import numpy as np

L, W = 105.0, 68.0
GOAL_HALF = 3.66
HZ = 10
DT = 1.0 / HZ
SIDES = ("home", "away")


def side_index(side: str) -> int:
    return 0 if side == "home" else 1


def attacking_goal(team: int) -> np.ndarray:
    return np.array([L if team == 0 else 0.0, W / 2])


def progress(team: int, x: float) -> float:
    """0 at a team's own goal line, 1 at the goal line it attacks."""
    return x / L if team == 0 else 1.0 - x / L


def shot_geometry(x: float, y: float, team: int) -> tuple[float, float]:
    gx = L if team == 0 else 0.0
    dx, dy = abs(gx - x), abs(W / 2 - y)
    dist = math.hypot(dx, dy)
    angle = math.atan2(2 * GOAL_HALF * dx, dx * dx + dy * dy - GOAL_HALF**2)
    if angle < 0:
        angle += math.pi
    return dist, angle


def xg(x: float, y: float, team: int, under_pressure: bool = False, header: bool = False) -> float:
    """Location-based expected-goals model (distance + visible goal angle)."""
    dist, angle = shot_geometry(x, y, team)
    z = -0.6 - 0.13 * dist + 1.2 * angle
    p = 1.0 / (1.0 + math.exp(-z))
    if under_pressure:
        p *= 0.72
    if header:
        p *= 0.6
    return float(min(max(p, 0.005), 0.95))


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from p to segment ab, and the projection fraction along ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a)), 0.0
    f = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + f * ab))), f


def match_seconds(period: int, t_in_period: float) -> float:
    return t_in_period + (0.0 if period == 1 else 45 * 60.0)


def clock(period: int, t_in_period: float) -> str:
    s = match_seconds(period, t_in_period)
    return f"{int(s // 60):02d}:{int(s % 60):02d}"
