"""Agent-based synthetic football match simulator.

Produces two synchronised streams, like a real data provider would:

* tracking frames: 22 players + ball at 10 Hz
* labelled events: passes, shots, tackles, interceptions, pressures,
  fouls, cards, set pieces, goals, period markers

Players move under speed and acceleration limits towards targets derived
from their formation slot, the ball position, and team tactics (press
intensity, tempo, directness). Tactics shift with game state, so the
explainability layer has real tactical changes to detect. Those ground-truth
shifts are kept in ``truth`` and never streamed, so detection can be scored.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..football import (DT, GOAL_HALF, HZ, L, SIDES, W, attacking_goal, clock,
                        match_seconds, point_segment_distance, progress, xg)
from .teams import COMPETITION, FORMATIONS, HARBOURSIDE, KINGSMOOR, REFEREE, VENUE, Team

ACCEL = 4.0
ATTACKING_POSITIONS = {"ST", "RW", "LW", "AM", "RAM", "LAM", "CM"}


@dataclass
class MatchData:
    meta: dict
    events: list[dict]
    t: np.ndarray       # (N,) seconds since kick-off, continuous across halves
    period: np.ndarray  # (N,)
    xy: np.ndarray      # (N, 23, 2) players 0-21 (home then away), ball at 22
    truth: dict

    def save(self, stem: str | Path) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix(".npz"), t=self.t, period=self.period, xy=self.xy)
        stem.with_suffix(".json").write_text(
            json.dumps({"meta": self.meta, "events": self.events, "truth": self.truth}, ensure_ascii=False))

    @classmethod
    def load(cls, stem: str | Path) -> "MatchData":
        stem = Path(stem)
        arr = np.load(stem.with_suffix(".npz"))
        doc = json.loads(stem.with_suffix(".json").read_text())
        return cls(doc["meta"], doc["events"], arr["t"], arr["period"], arr["xy"], doc["truth"])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


class MatchSimulator:
    def __init__(self, home: Team = KINGSMOOR, away: Team = HARBOURSIDE, seed: int = 7,
                 half_length: float = 45 * 60.0):
        self.teams = (home, away)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.half_length = half_length
        self.players = [p for t in self.teams for p in t.players]
        self.slots = [FORMATIONS[t.formation] for t in self.teams]
        self.team_of = np.array([0] * 11 + [1] * 11)
        a = lambda k: np.array([getattr(p, k) for p in self.players])
        self.pace, self.passing, self.finishing = a("pace"), a("passing"), a("finishing")
        self.tackling, self.dribbling = a("tackling"), a("dribbling")
        self.vmax = 7.4 + 2.2 * self.pace  # 7.4 to 9.6 m/s
        self.pos = np.zeros((22, 2))
        self.vel = np.zeros((22, 2))
        self.jitter = np.zeros((22, 2))
        self.base_tactics = [dict(home.tactics), dict(away.tactics)]
        self.tactics = [dict(home.tactics), dict(away.tactics)]
        self.score = [0, 0]
        self.events: list[dict] = []
        self.t = 0.0
        self.tp = 0.0
        self.period = 1
        self.ball = np.array([L / 2, W / 2])
        self.ball_ref = self.ball.copy()  # smoothed ball position the team shape reacts to
        self.mode = "dead"
        self.carrier: int | None = None
        self.poss = 0
        self.flight: dict | None = None
        self.loose: dict | None = None
        self.dead: dict | None = None
        self.restart_kind: str | None = None
        self.hold_until = 0.0
        self.carry_target = self.ball.copy()
        self.pressured: set[int] = set()
        self.tackle_cool: dict[int, float] = {}
        self.booked: set[int] = set()
        self.last_pass: tuple[int, int, float] | None = None
        self.truth = {"possession": [], "tactics": [], "shot_xg": {}}
        # Random "purple patches": spells of extra intensity that create momentum swings.
        self.spells = []
        for team in (0, 1):
            for _ in range(2):
                start = float(self.rng.uniform(8, 82))
                self.spells.append((team, start, start + float(self.rng.uniform(6, 11))))

    # ---------- coordinates ----------
    def _xy(self, team: int, u: float, v: float) -> np.ndarray:
        return np.array([u * L, v * W]) if team == 0 else np.array([(1 - u) * L, (1 - v) * W])

    def _uv(self, team: int, p: np.ndarray) -> tuple[float, float]:
        return (p[0] / L, p[1] / W) if team == 0 else (1 - p[0] / L, 1 - p[1] / W)

    def _range(self, team: int) -> range:
        return range(team * 11, team * 11 + 11)

    def _pos_name(self, i: int) -> str:
        return self.players[i].position

    # ---------- events ----------
    def _emit(self, kind: str, team: int | None = None, player: int | None = None,
              at: np.ndarray | None = None, **kw) -> int:
        ev = {"id": len(self.events) + 1, "t": round(self.t, 2), "period": self.period,
              "clock": clock(self.period, self.tp), "type": kind}
        if team is not None:
            ev["team"] = SIDES[team]
        if player is not None:
            ev["player_id"] = self.players[player].id
        if at is not None:
            ev["x"], ev["y"] = round(float(at[0]), 2), round(float(at[1]), 2)
        for k, v in kw.items():
            if isinstance(v, (float, np.floating)):
                v = round(float(v), 3)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            ev[k] = v
        self.events.append(ev)
        return ev["id"]

    # ---------- main loop ----------
    def run(self) -> MatchData:
        frames, ts, periods = [], [], []
        stoppage = [float(self.rng.uniform(60, 200)), float(self.rng.uniform(150, 330))]
        for period in (1, 2):
            self.period, self.tp = period, 0.0
            self._kickoff(team=0 if period == 1 else 1, teleport=True)
            self._emit("period_start")
            end = self.half_length + stoppage[period - 1]
            while self.tp < end or self.mode in ("flight", "loose"):
                self._step()
                frames.append(np.vstack([self.pos, self.ball[None]]).astype(np.float32))
                ts.append(self.t)
                periods.append(period)
                self.t += DT
                self.tp += DT
            self._emit("period_end", score=list(self.score))
        self._emit("full_time", score=list(self.score))
        return MatchData(self._meta(), self.events, np.array(ts, dtype=np.float32),
                         np.array(periods, dtype=np.int8), np.stack(frames), self.truth)

    def _meta(self) -> dict:
        return {
            "match_id": f"SYN-{self.seed:04d}",
            "synthetic": True,
            "competition": COMPETITION,
            "venue": VENUE,
            "referee": REFEREE,
            "seed": self.seed,
            "hz": HZ,
            "pitch": {"length": L, "width": W},
            "coordinate_note": "metres; home attacks towards x=105 in both halves",
            "weather": ["clear, 14°C", "light rain, 9°C", "overcast, 11°C"][self.seed % 3],
            "teams": {
                t.side: {
                    "name": t.name, "short": t.short, "color": t.color, "alt_color": t.alt_color,
                    "formation": t.formation,
                    "players": [{"id": p.id, "name": p.name, "short": p.short, "number": p.number,
                                 "position": p.position, "index": i + (0 if t.side == "home" else 11)}
                                for i, p in enumerate(t.players)],
                } for t in self.teams
            },
        }

    def _step(self) -> None:
        if int(round(self.tp * HZ)) % (60 * HZ) == 0:
            self._update_tactics()
        self.jitter = self.jitter * 0.99 + self.rng.normal(0, 0.2, (22, 2))
        self.ball_ref += (self.ball - self.ball_ref) * min(1.0, DT / 1.2)
        targets, caps = self._targets()
        self._move(targets, caps)
        if self.mode == "held":
            self._held()
        elif self.mode == "flight":
            self._flight_tick()
        elif self.mode == "loose":
            self._loose_tick()
        else:
            self._dead_tick()

    # ---------- tactics ----------
    def _minute(self) -> float:
        return match_seconds(self.period, self.tp) / 60.0

    def _update_tactics(self) -> None:
        m = self._minute()
        for team in (0, 1):
            tac = dict(self.base_tactics[team])
            diff = self.score[team] - self.score[1 - team]
            if diff < 0 and m >= 55:
                tac["press"] += 0.30
                tac["directness"] += 0.20
                tac["tempo"] += 0.15
            elif diff > 0 and m >= 70:
                tac["press"] -= 0.22
                tac["tempo"] -= 0.15
                tac["directness"] += 0.10
            for s_team, a, b in self.spells:
                if s_team == team and a <= m < b:
                    tac["press"] += 0.25
                    tac["tempo"] += 0.20
            tac = {k: round(float(np.clip(v, 0.05, 0.98)), 2) for k, v in tac.items()}
            if tac != self.tactics[team]:
                self.tactics[team] = tac
                self.truth["tactics"].append({"t": round(self.t, 1), "minute": round(m, 1),
                                              "team": SIDES[team], **tac})

    # ---------- movement ----------
    def _shape(self, team: int) -> np.ndarray:
        bu, bv = self._uv(team, self.ball_ref)
        press = self.tactics[team]["press"]
        if self.mode == "dead":
            has = self.dead["team"] == team
        else:
            has = self.poss == team
        kickoff = self.mode == "dead" and self.dead["kind"] == "kickoff"
        if has:
            uc, ds, ls, sh = 0.27 + 0.46 * bu, 1.25, 1.12, 0.12
        else:
            uc, ds, ls, sh = 0.29 + 0.36 * bu + 0.14 * (press - 0.5), 0.85, 0.78, 0.32
        if kickoff:
            uc, ds, ls, sh = 0.34, 1.0, 1.0, 0.0
        opp = 1 - team
        line = max(progress(team, self.pos[k][0]) for k in range(opp * 11 + 1, opp * 11 + 11))
        out = np.zeros((11, 2))
        for k, (du, v) in enumerate(self.slots[team]):
            if k == 0:
                u, vv = 0.025 + 0.08 * bu, 0.5 + (bv - 0.5) * 0.3
            else:
                u = uc + du * ds
                vv = 0.5 + (v - 0.5) * ls + (bv - 0.5) * sh
                if kickoff:
                    u = min(u, 0.48)
                elif has:
                    u = min(u, line + 0.01)
            out[k] = self._xy(team, float(np.clip(u, 0.01, 0.97)), float(np.clip(vv, 0.03, 0.97)))
        return out

    def _targets(self) -> tuple[np.ndarray, np.ndarray]:
        fatigue = 1.0 - 0.07 * min(self.t / 5600.0, 1.0)
        vmax = self.vmax * fatigue
        tg = np.vstack([self._shape(0), self._shape(1)]) + self.jitter
        dist = np.linalg.norm(tg - self.pos, axis=1)
        cap = np.where(dist > 12, 0.68 * vmax, np.where(dist > 5, 0.36 * vmax, 0.26 * vmax))
        if self.mode != "dead":
            self._mark(tg)

        if self.mode == "held":
            c = self.carrier
            tg[c], cap[c] = self.carry_target, 0.45 * vmax[c]
            self._assign_pressers(tg, cap, vmax)
        elif self.mode == "flight":
            f = self.flight
            if f["receiver"] is not None:
                r = f["receiver"]
                tg[r], cap[r] = f["end"], 0.95 * vmax[r]
            if f["kind"] == "pass" and f["result"] == "complete":
                opp = 1 - self.team_of[f["receiver"]]
                ids = list(self._range(opp))[1:]
                k = min(ids, key=lambda j: np.linalg.norm(self.pos[j] - f["end"]))
                tg[k] = f["end"] + _unit(attacking_goal(1 - opp) - f["end"]) * 1.5
                cap[k] = 0.85 * vmax[k]
        elif self.mode == "loose":
            for team in (0, 1):
                ids = list(self._range(team))
                k = min(ids, key=lambda j: np.linalg.norm(self.pos[j] - self.ball))
                tg[k], cap[k] = self.ball, 0.95 * vmax[k]
        else:
            p = self.dead["player"]
            tg[p], cap[p] = self.dead["spot"], 0.75 * vmax[p]
        cap[0] = min(cap[0], 0.6 * vmax[0])
        cap[11] = min(cap[11], 0.6 * vmax[11])
        return tg, cap

    def _mark(self, tg: np.ndarray) -> None:
        """Out of possession, outfield players tuck in goal-side of the nearest attacker in their zone."""
        att_team, d_team = self.poss, 1 - self.poss
        attackers = list(self._range(att_team))[1:]
        own_goal = attacking_goal(att_team)
        for j in list(self._range(d_team))[1:]:
            spot = tg[j].copy()
            d = np.linalg.norm(self.pos[attackers] - spot, axis=1)
            k = int(np.argmin(d))
            if d[k] > 14:
                continue
            a = self.pos[attackers[k]]
            goal_side = a + _unit(own_goal - a) * 1.6
            new = spot + 0.6 * (goal_side - spot)
            floor = progress(d_team, spot[0]) - 0.05
            if progress(d_team, new[0]) < floor:
                new[0] = floor * L if d_team == 0 else (1 - floor) * L
            tg[j] = new

    def _assign_pressers(self, tg: np.ndarray, cap: np.ndarray, vmax: np.ndarray) -> None:
        c = self.carrier
        d_team = 1 - self.team_of[c]
        press = self.tactics[d_team]["press"]
        ball_depth = progress(d_team, self.ball[0])  # how far up the pitch from the defenders' goal
        active = ball_depth > 0.55 - 0.65 * press
        n = 1 + int(press > 0.5) + int(press > 0.8) if active else 1
        ids = sorted(list(self._range(d_team))[1:], key=lambda j: np.linalg.norm(self.pos[j] - self.ball))
        own_goal = attacking_goal(1 - d_team)
        for rank, j in enumerate(ids[:n]):
            d = np.linalg.norm(self.pos[j] - self.ball)
            if not active and d > 9:
                break
            offset = _unit(own_goal - self.ball) * (1.5 + 1.5 * rank)
            tg[j] = self.ball + offset
            cap[j] = (0.90 if active else 0.65) * vmax[j]

    def _move(self, tg: np.ndarray, cap: np.ndarray) -> None:
        d = tg - self.pos
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        desired = d / dist * np.minimum(cap[:, None], dist * 1.1)
        dv = desired - self.vel
        mag = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
        self.vel += dv * np.minimum(1.0, ACCEL * DT / mag)
        self.pos += self.vel * DT
        self.pos[:, 0] = np.clip(self.pos[:, 0], -2, L + 2)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -2, W + 2)

    # ---------- possession ----------
    def _hold_time(self, team: int, pressured: bool) -> float:
        mean = 2.8 - 1.5 * self.tactics[team]["tempo"]
        if pressured:
            mean *= 0.6
        return max(0.3, float(self.rng.gamma(2.0, mean / 2.0)))

    def _nearest_opp(self, team: int, p: np.ndarray, skip_gk: bool = False) -> tuple[int, float]:
        ids = list(self._range(1 - team))[1 if skip_gk else 0:]
        d = np.linalg.norm(self.pos[ids] - p, axis=1)
        k = int(np.argmin(d))
        return ids[k], float(d[k])

    def _new_carry_target(self, c: int) -> None:
        team = self.team_of[c]
        fwd = _unit(attacking_goal(team) - self.pos[c])
        if progress(team, self.pos[c][0]) < 0.66:
            fwd = np.array([1.0 if team == 0 else -1.0, 0.0])
        lateral = np.array([-fwd[1], fwd[0]]) * self.rng.normal(0, 0.5)
        _, d = self._nearest_opp(team, self.pos[c] + fwd * 4)
        if d < 3:
            lateral *= 3
        tgt = self.pos[c] + _unit(fwd + lateral) * float(self.rng.uniform(7, 13))
        self.carry_target = np.array([np.clip(tgt[0], 1, L - 1), np.clip(tgt[1], 1, W - 1)])

    def _give(self, j: int, hold: float | None = None) -> None:
        team = int(self.team_of[j])
        if team != self.poss or not self.truth["possession"]:
            self.truth["possession"].append({"t": round(self.t, 1), "team": SIDES[team]})
        self.poss, self.carrier, self.mode = team, j, "held"
        self.flight = self.loose = None
        self.pressured = set()
        _, d = self._nearest_opp(team, self.pos[j])
        self.hold_until = self.t + (hold if hold is not None else self._hold_time(team, d < 3))
        self._new_carry_target(j)

    def _held(self) -> None:
        c = self.carrier
        team, opp = int(self.team_of[c]), 1 - int(self.team_of[c])
        self.ball = self.pos[c] + _unit(self.vel[c]) * 0.5
        if np.linalg.norm(self.carry_target - self.pos[c]) < 1.5:
            self._new_carry_target(c)
        for k in self._range(opp):
            d = float(np.linalg.norm(self.pos[k] - self.pos[c]))
            closing = float(self.vel[k] @ _unit(self.pos[c] - self.pos[k])) > 2.5
            if d < 2.5 and closing and k not in self.pressured and k != opp * 11:
                self.pressured.add(k)
                self._emit("pressure", opp, k, self.pos[k], on_player_id=self.players[c].id)
            if d < 1.8 and k != opp * 11 and self.tackle_cool.get(k, 0) < self.t and self.restart_kind is None:
                if self.rng.random() < (0.06 + 0.12 * self.tactics[opp]["press"]) * DT:
                    self.tackle_cool[k] = self.t + 2.5
                    if self._tackle(k, c):
                        return
        if self.t >= self.hold_until:
            self._decide(c)

    def _tackle(self, k: int, c: int) -> bool:
        p_win = float(np.clip(0.5 + 0.35 * (self.tackling[k] - self.dribbling[c]), 0.2, 0.8))
        team_k = int(self.team_of[k])
        if self.rng.random() < p_win:
            self._emit("tackle", team_k, k, self.pos[k], outcome="won", on_player_id=self.players[c].id)
            self._give(k)
            return True
        if self.rng.random() < 0.30:
            card = k not in self.booked and self.rng.random() < 0.16
            self._emit("foul", team_k, k, self.pos[k], on_player_id=self.players[c].id)
            if card:
                self.booked.add(k)
                self._emit("card", team_k, k, self.pos[k], card="yellow")
            self._set_dead("free_kick", int(self.team_of[c]), c, self.ball.copy(), 6.0)
            return True
        self._emit("tackle", team_k, k, self.pos[k], outcome="lost", on_player_id=self.players[c].id)
        return False

    def _decide(self, c: int) -> None:
        team = int(self.team_of[c])
        kind, self.restart_kind = self.restart_kind, None
        if kind == "kickoff":
            return self._pass(c, mode="kickoff")
        if kind == "throw_in":
            return self._pass(c, mode="short")
        if kind == "corner":
            return self._pass(c, mode="cross")
        if kind == "goal_kick" or c in (0, 11):
            return self._pass(c, mode="keeper")
        p = self.pos[c]
        dist = float(np.linalg.norm(attacking_goal(team) - p))
        if dist < 33:
            _, d_opp = self._nearest_opp(team, p, skip_gk=True)
            chance = xg(p[0], p[1], team, under_pressure=d_opp < 1.5)
            p_shot = min(0.25, chance * 0.42 + (0.045 if 16 < dist < 28 else 0.0))
            if self._pos_name(c) in ATTACKING_POSITIONS:
                p_shot *= 1.2
            if self.rng.random() < p_shot:
                return self._shoot(c, chance, header=kind == "header")
        if self.rng.random() < 0.10:
            self._new_carry_target(c)
            self.hold_until = self.t + self._hold_time(team, False)
            return
        self._pass(c)

    # ---------- passing ----------
    def _pass_success(self, c: int, end: np.ndarray, team: int, length: float) -> float:
        start = self.pos[c]
        lane = 0
        for k in self._range(1 - team):
            d, f = point_segment_distance(self.pos[k], start, end)
            if d < 2.0 and 0.15 < f < 0.95:
                lane += 1
        _, press_d = self._nearest_opp(team, start)
        _, mark_d = self._nearest_opp(team, end)
        p = 0.99 - 0.007 * max(0.0, length - 15) - 0.04 * lane
        p -= 0.08 if press_d < 2 else 0.03 if press_d < 4 else 0.0
        p -= 0.08 if mark_d < 1.5 else 0.03 if mark_d < 3 else 0.0
        p += 0.10 * (self.passing[c] - 0.65)
        u_end, v_end = self._uv(team, end)
        if u_end > 0.84 and abs(v_end - 0.5) < 0.3:
            p -= 0.22  # the penalty area is crowded: real box-entry passes complete ~60%
        return float(np.clip(p, 0.2, 0.985))

    def _pass(self, c: int, mode: str = "open") -> None:
        team = int(self.team_of[c])
        tac = self.tactics[team]
        start = self.pos[c].copy()
        u_c = progress(team, start[0])
        cands, scores = [], []
        for j in self._range(team):
            if j == c:
                continue
            lead = self.pos[j] + self.vel[j] * 0.7
            lead = np.array([np.clip(lead[0], 0.5, L - 0.5), np.clip(lead[1], 0.5, W - 0.5)])
            d = float(np.linalg.norm(lead - start))
            max_len = {"short": 25, "kickoff": 22}.get(mode, 62)
            if d < 4 or d > max_len:
                continue
            gain = (progress(team, lead[0]) - u_c) * L
            _, open_d = self._nearest_opp(team, lead)
            if mode == "cross":
                u_j, v_j = self._uv(team, lead)
                if u_j < 0.83 or abs(v_j - 0.5) > 0.22:
                    continue
                s = 2.0 * self.finishing[j]
            elif mode == "kickoff":
                s = -0.15 * gain
            else:
                s = (0.07 * gain * (0.6 + tac["directness"]) + 0.30 * min(open_d, 7.0)
                     - 0.035 * max(0.0, d - 18) * (1.1 - tac["directness"]))
                if mode == "keeper":
                    s += 0.05 * d * tac["directness"]
            if j % 11 == 0:
                s -= 2.5
            cands.append((j, lead, d))
            scores.append(s)
        if not cands:
            self._new_carry_target(c)
            self.hold_until = self.t + 0.6
            return
        w = np.exp((np.array(scores) - max(scores)) / 0.9)
        j, end, d = cands[int(self.rng.choice(len(cands), p=w / w.sum()))]
        p_ok = self._pass_success(c, end, team, d)
        if mode == "cross":
            p_ok *= 0.55
        speed = float(np.clip(9 + 0.32 * d + self.rng.normal(0, 1.2), 8, 30))
        result, receiver = "complete", j
        if self.rng.random() >= p_ok:
            if (d > 28 and self.rng.random() < 0.4) or mode == "cross" and self.rng.random() < 0.3:
                result, receiver = "out", None
                end = self._out_point(start, end)
            elif self.rng.random() < 0.6:
                result, receiver = "loose", None
                end = end + self.rng.normal(0, 2.5, 2)
                end = np.array([np.clip(end[0], 1, L - 1), np.clip(end[1], 1, W - 1)])
            else:
                result = "intercept"
                best = None
                for k in self._range(1 - team):
                    dk, f = point_segment_distance(self.pos[k], start, end)
                    if f >= 0.3 and (best is None or dk < best[0]):
                        best = (dk, f, k)
                if best is None:
                    receiver, _ = self._nearest_opp(team, end)
                else:
                    _, f, receiver = best
                    end = start + (end - start) * f
        length = float(np.linalg.norm(end - start))
        self.flight = {"kind": "pass", "start": start, "end": end, "t0": self.t,
                       "t1": self.t + max(length, 1.0) / speed, "receiver": receiver,
                       "result": result, "passer": c, "mode": mode}
        self.mode, self.carrier = "flight", None
        outcome = {"complete": "complete", "intercept": "incomplete", "loose": "incomplete", "out": "out"}[result]
        self._emit("pass", team, c, start, end_x=round(float(end[0]), 2), end_y=round(float(end[1]), 2),
                   outcome=outcome, pass_type=mode,
                   recipient_id=self.players[j].id if result == "complete" else None)

    def _out_point(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        d = _unit(end - start)
        ts = []
        if d[0] > 1e-6: ts.append((L - start[0]) / d[0])
        if d[0] < -1e-6: ts.append(-start[0] / d[0])
        if d[1] > 1e-6: ts.append((W - start[1]) / d[1])
        if d[1] < -1e-6: ts.append(-start[1] / d[1])
        t = min(ts) + 0.5
        p = start + d * t
        return np.array([np.clip(p[0], -0.5, L + 0.5), np.clip(p[1], -0.5, W + 0.5)])

    # ---------- shooting ----------
    def _shoot(self, c: int, chance: float, header: bool = False) -> None:
        team, opp = int(self.team_of[c]), 1 - int(self.team_of[c])
        start = self.pos[c].copy()
        goal = attacking_goal(team)
        fin = self.finishing[c]
        speed = float(np.clip(self.rng.normal(22 + 9 * fin - (8 if header else 0), 3), 11, 36))
        blockers = []
        for k in list(self._range(opp))[1:]:
            dk, f = point_segment_distance(self.pos[k], start, goal)
            if dk < 1.6 and f > 0.05:
                blockers.append((f, k))
        gk = opp * 11
        p_goal = float(np.clip(chance * (0.85 + 0.3 * (fin - 0.5)), 0, 0.9))
        r = self.rng.random()
        if r < p_goal:
            result, end, receiver = "goal", goal + np.array([0, self.rng.uniform(-3.2, 3.2)]), None
        elif blockers and self.rng.random() < min(0.6, 0.3 * len(blockers)):
            f, k = min(blockers)
            result, end, receiver = "blocked", start + (goal - start) * f, k
        elif self.rng.random() < 0.47:
            result, end, receiver = "saved", self.pos[gk].copy(), gk
        else:
            side = 1 if self.rng.random() < 0.5 else -1
            result, end, receiver = "off_target", goal + np.array([0, side * self.rng.uniform(4, 10)]), None
        self.truth["shot_xg"][str(len(self.events) + 1)] = round(chance, 4)
        dist = float(np.linalg.norm(end - start))
        self.flight = {"kind": "shot", "start": start, "end": end, "t0": self.t,
                       "t1": self.t + max(dist, 1.0) / speed, "receiver": receiver,
                       "result": result, "passer": c, "mode": "shot"}
        self.mode, self.carrier = "flight", None
        self._emit("shot", team, c, start, end_x=round(float(end[0]), 2), end_y=round(float(end[1]), 2),
                   outcome=result, body_part="head" if header else "foot")

    # ---------- ball in flight / loose / dead ----------
    def _flight_tick(self) -> None:
        f = self.flight
        a = min(1.0, (self.t - f["t0"]) / (f["t1"] - f["t0"]))
        self.ball = f["start"] + (f["end"] - f["start"]) * a
        if a < 1.0:
            return
        if f["kind"] == "pass":
            if f["result"] == "out":
                return self._ball_out(f)
            r = f["receiver"]
            if f["result"] == "loose":
                self.mode, self.flight = "loose", None
                self.loose = {"vel": _unit(f["end"] - f["start"]) * 1.5, "since": self.t, "intended": None,
                              "passer": f["passer"]}
                return
            if f["result"] == "intercept":
                self._emit("interception", int(self.team_of[r]), r, self.ball)
                return self._give(r)
            if np.linalg.norm(self.pos[r] - self.ball) > 1.6:
                self.mode, self.flight = "loose", None
                self.loose = {"vel": _unit(f["end"] - f["start"]) * 2.5, "since": self.t, "intended": r,
                              "passer": f["passer"]}
                return
            self.last_pass = (f["passer"], r, self.t)
            self._give(r)
            if f["mode"] == "cross":
                self.hold_until = self.t + 0.3
                self.restart_kind = "header"
            return
        self._shot_result(f)

    def _shot_result(self, f: dict) -> None:
        c = f["passer"]
        team, opp = int(self.team_of[c]), 1 - int(self.team_of[c])
        res = f["result"]
        if res == "goal":
            self.score[team] += 1
            assist = None
            if self.last_pass and self.last_pass[1] == c and self.t - self.last_pass[2] < 15:
                assist = self.players[self.last_pass[0]].id
            self._emit("goal", team, c, self.ball, assist_id=assist, score=list(self.score))
            self._kickoff(opp, teleport=False)
        elif res == "saved":
            gk = opp * 11
            self._emit("save", opp, gk, self.ball)
            if self.rng.random() < 0.28:
                self._set_dead("corner", team, None, self._corner_spot(team), 14.0)
            else:
                self._give(gk, hold=float(self.rng.uniform(3, 6)))
        elif res == "blocked":
            k = f["receiver"]
            self._emit("block", opp, k, self.ball)
            r = self.rng.random()
            if r < 0.3:
                self._set_dead("corner", team, None, self._corner_spot(team), 14.0)
            elif r < 0.75:
                self._give(k)
            else:
                self.mode, self.flight = "loose", None
                self.loose = {"vel": self.rng.normal(0, 3, 2), "since": self.t, "intended": None, "passer": c}
        else:
            spot = np.array([5.5 if opp == 0 else L - 5.5, W / 2 + self.rng.uniform(-9, 9)])
            self._set_dead("goal_kick", opp, opp * 11, spot, 9.0)
        self.last_pass = None if res != "goal" else self.last_pass

    def _corner_spot(self, team: int) -> np.ndarray:
        x = L - 0.5 if team == 0 else 0.5
        return np.array([x, 0.5 if self.rng.random() < 0.5 else W - 0.5])

    def _ball_out(self, f: dict) -> None:
        p = f["end"]
        passer_team = int(self.team_of[f["passer"]])
        other = 1 - passer_team
        if 0 < p[0] < L:
            spot = np.array([p[0], 0.0 if p[1] < W / 2 else W])
            self._emit("out", passer_team, f["passer"], spot, restart="throw_in")
            self._set_dead("throw_in", other, None, spot, float(self.rng.uniform(5, 9)))
            return
        end_team = 0 if p[0] <= 0 else 1  # whose goal line
        if end_team == passer_team:
            spot = self._corner_spot(other)
            self._emit("out", passer_team, f["passer"], spot, restart="corner")
            self._set_dead("corner", other, None, spot, 14.0)
        else:
            spot = np.array([5.5 if end_team == 0 else L - 5.5, W / 2])
            self._emit("out", passer_team, f["passer"], spot, restart="goal_kick")
            self._set_dead("goal_kick", end_team, end_team * 11, spot, 9.0)

    def _loose_tick(self) -> None:
        lo = self.loose
        self.ball = self.ball + lo["vel"] * DT
        lo["vel"] *= 0.94
        self.ball = np.array([np.clip(self.ball[0], 0.5, L - 0.5), np.clip(self.ball[1], 0.5, W - 0.5)])
        d = np.linalg.norm(self.pos - self.ball, axis=1)
        j = int(np.argmin(d))
        if d[j] < 1.1 or self.t - lo["since"] > 6:
            team = int(self.team_of[j])
            passer_team = int(self.team_of[lo["passer"]])
            if team != passer_team:
                self._emit("ball_recovery", team, j, self.ball)
            elif lo["intended"] == j:
                self.last_pass = (lo["passer"], j, self.t)
            self._give(j)

    def _set_dead(self, kind: str, team: int, player: int | None, spot: np.ndarray, wait: float) -> None:
        if player is None:
            ids = list(self._range(team))[1:]
            player = min(ids, key=lambda j: np.linalg.norm(self.pos[j] - spot))
        self.mode, self.carrier, self.flight, self.loose = "dead", None, None, None
        self.ball = spot.copy()
        self.dead = {"kind": kind, "team": team, "player": player, "spot": spot.copy(), "until": self.t + wait}

    def _kickoff(self, team: int, teleport: bool) -> None:
        kicker = next(i for i in self._range(team) if self.players[i].position == "ST")
        self._set_dead("kickoff", team, kicker, np.array([L / 2, W / 2]), 0.5 if teleport else 38.0)
        if teleport:
            self.pos = np.vstack([self._shape(0), self._shape(1)])
            self.vel[:] = 0
            self.pos[kicker] = self.ball + np.array([-0.6 if team == 0 else 0.6, 0])

    def _dead_tick(self) -> None:
        dd = self.dead
        p = dd["player"]
        if self.t < dd["until"]:
            return
        if np.linalg.norm(self.pos[p] - dd["spot"]) > 1.5:
            if self.t < dd["until"] + 8:
                return
            self.pos[p] = dd["spot"].copy()
        kind = dd["kind"]
        self._emit(kind, dd["team"], p, dd["spot"])
        self.restart_kind = kind if kind != "free_kick" else None
        self.carrier = p
        self.ball = dd["spot"].copy()
        self._give(p, hold=0.5)
        self.restart_kind = kind if kind != "free_kick" else None


def simulate(seed: int = 7, half_length: float = 45 * 60.0) -> MatchData:
    return MatchSimulator(seed=seed, half_length=half_length).run()
