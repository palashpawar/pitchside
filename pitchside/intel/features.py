"""Frame-level features for pass difficulty (xPass) — shared by training and live scoring."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..football import L, W, point_segment_distance, progress

FEATURES = ["length", "lane_opponents", "passer_pressed", "receiver_marked", "bypassed", "into_box", "forward_m"]
MODEL_PATH = Path(__file__).with_name("xpass_model.json")


def pass_features(start: np.ndarray, end: np.ndarray, team: int, xy: np.ndarray) -> dict:
    """Geometry of a pass at the moment it is played. ``xy`` is one tracking frame (23, 2)."""
    opp = xy[(1 - team) * 11:(1 - team) * 11 + 11]
    length = float(np.linalg.norm(end - start))
    lane = 0
    for p in opp:
        d, f = point_segment_distance(p, start, end)
        if d < 2.5 and 0.1 < f < 0.95:
            lane += 1
    press_d = float(np.min(np.linalg.norm(opp - start, axis=1)))
    mark_d = float(np.min(np.linalg.norm(opp - end, axis=1)))
    u0, u1 = progress(team, start[0]), progress(team, end[0])
    bypassed = 0
    if u1 > u0:
        bypassed = int(sum(u0 < progress(team, p[0]) < u1 for p in opp[1:]))
    into_box = u1 > 0.843 and abs(end[1] - W / 2) < 20.16
    return {
        "length": length,
        "lane_opponents": lane,
        "passer_pressed": press_d,
        "receiver_marked": mark_d,
        "bypassed": bypassed,
        "into_box": bool(into_box),
        "forward_m": (u1 - u0) * L,
    }


def feature_vector(f: dict) -> np.ndarray:
    return np.array([
        1.0,
        max(0.0, f["length"] - 10) / 10,
        f["lane_opponents"],
        1.0 if f["passer_pressed"] < 2 else 0.5 if f["passer_pressed"] < 4 else 0.0,
        1.0 if f["receiver_marked"] < 1.5 else 0.5 if f["receiver_marked"] < 3 else 0.0,
        f["bypassed"] / 3,
        1.0 if f["into_box"] else 0.0,
        max(0.0, f["forward_m"]) / 10,
    ])


class XPassModel:
    """Logistic regression: probability a pass is completed given its geometry."""

    def __init__(self, coef: np.ndarray | None = None, meta: dict | None = None):
        if coef is None and MODEL_PATH.exists():
            doc = json.loads(MODEL_PATH.read_text())
            coef, meta = np.array(doc["coef"]), doc.get("meta", {})
        if coef is None:  # hand-set prior, used until a model is trained
            coef = np.array([3.2, -0.45, -0.35, -0.6, -0.6, -0.1, -1.0, 0.0])
        self.coef = coef
        self.meta = meta or {}

    def prob(self, f: dict) -> float:
        z = float(feature_vector(f) @ self.coef)
        return 1.0 / (1.0 + math.exp(-z))

    def difficulty(self, f: dict) -> int:
        """0-100 rating: 100 x the probability the pass fails."""
        return int(round(100 * (1 - self.prob(f))))

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
        """Newton-Raphson (IRLS) logistic regression."""
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (p - y) + l2 * w
            H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return w
