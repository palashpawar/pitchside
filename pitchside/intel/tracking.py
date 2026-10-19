"""Metrics computed straight from tracking frames: speed, distance, ball speed, auto-eventing."""
from __future__ import annotations

import numpy as np

from ..football import DT

SPRINT_MS = 7.0          # 25.2 km/h: the standard sprint threshold
HIGH_INTENSITY_MS = 5.5  # 19.8 km/h
TOP_SPEED_ALERT_MS = 30 / 3.6
TELEPORT_MS = 13.0       # faster than any human: a discontinuity, not movement


class TrackingMetrics:
    """Running physical metrics per player, plus instantaneous ball speed."""

    def __init__(self, n: int = 22):
        self.n = n
        self.distance = np.zeros(n)
        self.hi_distance = np.zeros(n)
        self.top_speed = np.zeros(n)
        self.sprints = np.zeros(n, dtype=int)
        self.speed = np.zeros(n)
        self.in_sprint = np.zeros(n, dtype=bool)
        self.ball_speed = 0.0
        self.match_top = 0.0
        self._prev: np.ndarray | None = None
        self._prev_period: int | None = None
        self._prev_t = 0.0
        self._last_alert = np.full(n, -1e9)

    def update(self, t: float, period: int, xy: np.ndarray) -> list[dict]:
        """Ingest one frame; return threshold triggers (speed alerts, distance milestones)."""
        dt = t - self._prev_t
        if self._prev is None or period != self._prev_period or not 0 < dt < 1.0:
            self._prev, self._prev_period, self._prev_t = xy.copy(), period, t
            return []
        step = np.linalg.norm(xy[:self.n] - self._prev[:self.n], axis=1)
        v = step / dt
        valid = v < TELEPORT_MS
        v = np.where(valid, v, 0.0)
        self.speed = 0.6 * self.speed + 0.4 * v
        km_before = (self.distance // 1000).astype(int)
        self.distance += np.where(valid, step, 0.0)
        self.hi_distance += np.where(valid & (self.speed > HIGH_INTENSITY_MS), step, 0.0)
        starting = (self.speed > SPRINT_MS) & ~self.in_sprint
        self.sprints += starting
        self.in_sprint = np.where(self.in_sprint, self.speed > SPRINT_MS - 0.5, self.speed > SPRINT_MS)
        bv = float(np.linalg.norm(xy[22] - self._prev[22]) / dt)
        self.ball_speed = bv if bv < 45 else 0.0
        triggers = []
        for i in np.flatnonzero(self.speed > self.top_speed):
            self.top_speed[i] = self.speed[i]
            if self.speed[i] >= TOP_SPEED_ALERT_MS and t - self._last_alert[i] > 120:
                self._last_alert[i] = t
                record = self.speed[i] > self.match_top
                triggers.append({"type": "speed_alert", "index": int(i), "speed_kmh": round(float(self.speed[i]) * 3.6, 1),
                                 "match_record": bool(record)})
            self.match_top = max(self.match_top, float(self.speed[i]))
        for i in np.flatnonzero((self.distance // 1000).astype(int) > km_before):
            triggers.append({"type": "distance_milestone", "index": int(i), "km": int(self.distance[i] // 1000)})
        self._prev, self._prev_period, self._prev_t = xy.copy(), period, t
        return triggers


class AutoEventer:
    """Re-derives possession and passes from raw tracking, without the provider's event labels.

    A player "controls" the ball when it stays within 1 m of them for 3 consecutive
    frames (0.3 s): a ball in flight at pass speed covers >1 m per frame, so passes
    never register as touches. Control moving between team-mates is a pass;
    moving to an opponent is a turnover.
    """

    CONTROL_M = 1.0
    CONTROL_FRAMES = 3

    def __init__(self, team_of: np.ndarray):
        self.team_of = team_of
        self.controller: int | None = None
        self._cand: int | None = None
        self._count = 0
        self.possession_changes: list[dict] = []
        self.passes: list[dict] = []
        self._control_since = 0.0
        self._control_pos: np.ndarray | None = None

    def update(self, t: float, xy: np.ndarray) -> dict | None:
        prev, self._prev = getattr(self, "_prev", None), xy.copy()
        dt, self._prev_t = t - getattr(self, "_prev_t", t - DT), t
        if not 0 < dt < 1.0:
            prev = None
        d = raw = np.linalg.norm(xy[:22] - xy[22], axis=1)
        if prev is not None:
            # Controlling means moving with the ball, not just standing near it.
            rel = np.linalg.norm((xy[:22] - prev[:22]) - (xy[22] - prev[22]), axis=1) / dt
            d = np.where(rel < 2.5, d, np.inf)
        j = int(np.argmin(d))
        if self.controller is not None and j != self.controller and d[self.controller] < 1.5 \
                and d[j] > d[self.controller] - 0.3:
            j = self.controller  # hysteresis: a challenger must be clearly closer to take control
        if d[j] < self.CONTROL_M:
            self._count = self._count + 1 if self._cand == j else 1
            self._cand = j
        else:
            self._cand, self._count = None, 0
        need = self.CONTROL_FRAMES
        if (self.controller is not None and self._cand is not None
                and self.team_of[self._cand] != self.team_of[self.controller] and raw[self.controller] < 2.0):
            need = 6  # a challenge on a carrier only counts once the defender clearly has the ball
        if self._count != need or j == self.controller:
            return None
        prev, self.controller = self.controller, j
        out = None
        if prev is not None:
            same = self.team_of[prev] == self.team_of[j]
            out = {"t": round(t, 1), "type": "auto_pass" if same else "auto_turnover", "from": prev, "to": j,
                   "start": self._control_pos.tolist() if self._control_pos is not None else None,
                   "end": xy[j].tolist()}
            (self.passes if same else self.possession_changes).append(out)
        elif not self.possession_changes:
            self.possession_changes.append({"t": round(t, 1), "type": "auto_turnover", "from": None, "to": j})
        self._control_pos = xy[j].copy()
        return out


def possession_agreement(auto_changes: list[dict], truth: list[dict], team_of: np.ndarray,
                         tolerance: float = 2.5) -> dict:
    """Precision/recall of auto-detected possession changes against ground truth."""
    true_t = [(p["t"], p["team"]) for p in truth[1:]]
    auto_t = [(c["t"], "home" if team_of[c["to"]] == 0 else "away") for c in auto_changes if c["from"] is not None]
    used: set[int] = set()
    hits = 0
    for t, team in auto_t:
        for k, (tt, tteam) in enumerate(true_t):
            if k not in used and tteam == team and abs(tt - t) <= tolerance:
                used.add(k)
                hits += 1
                break
    precision = hits / len(auto_t) if auto_t else 0.0
    recall = hits / len(true_t) if true_t else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "auto": len(auto_t), "truth": len(true_t)}
