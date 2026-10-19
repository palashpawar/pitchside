"""Live match engine: ingest -> interpret -> detect what matters.

Consumes the event and tracking streams in time order. Maintains match state
(team and player stats, xG, xPass, physical metrics), computes the rolling
explainability indices, and raises:

* fast-lane overlay cues: deterministic, low latency (speed alerts, pass and shot badges)
* moments: things worth explaining, each with a ``facts`` dict that is the only
  source of numbers and names the agents may use (the Verifier enforces this)
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..football import L, W, progress, side_index, xg
from .features import XPassModel, pass_features
from .tracking import AutoEventer, TrackingMetrics

WINDOW = 300.0          # rolling window for indices (s of match time)
INDEX_EVERY = 15.0
BASELINE_PATH = Path(__file__).with_name("league_baselines.json")
ON_BALL = {"pass", "shot", "interception", "ball_recovery", "save", "kickoff", "throw_in",
           "goal_kick", "corner", "free_kick"}


@dataclass
class Moment:
    id: str
    kind: str
    t: float
    clock: str
    minute: int
    team: str | None
    player_ids: list[str]
    importance: float
    facts: dict
    evidence_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _new_team_stats() -> dict:
    return {k: 0 for k in ("passes", "completed", "shots", "on_target", "goals", "pressures", "tackles_won",
                           "interceptions", "recoveries", "fouls", "corners", "final_third_entries",
                           "box_entries", "progressive_passes", "line_breaking_passes")} | \
        {"xg": 0.0, "xpass": 0.0, "possession_s": 0.0}


def _new_player_stats() -> dict:
    return {k: 0 for k in ("touches", "passes", "completed", "key_passes", "shots", "goals", "assists",
                           "tackles_won", "interceptions", "recoveries", "pressures", "progressive_passes",
                           "line_breaking_passes", "fouls", "yellow_cards")} | \
        {"xg": 0.0, "xpass": 0.0, "difficulty_sum": 0.0, "hardest_pass": 0}


class Baselines:
    """League percentile tables for window metrics, built by scripts/build_baselines.py."""

    def __init__(self, path: Path = BASELINE_PATH):
        self.q = json.loads(path.read_text()) if path.exists() else {}

    def pct(self, metric: str, value: float, default_scale: float = 1.0) -> float:
        qs = self.q.get(metric)
        if not qs:
            return float(np.clip(50 + 50 * np.tanh(value / default_scale - 1), 0, 100))
        return float(np.interp(value, qs, np.linspace(0, 100, len(qs))))


class MatchEngine:
    def __init__(self, meta: dict, xpass: XPassModel | None = None, baselines: Baselines | None = None):
        self.meta = meta
        self.xpass = xpass or XPassModel()
        self.base = baselines or Baselines()
        self.roster: dict[str, dict] = {}
        self.index_of: dict[str, int] = {}
        for side in ("home", "away"):
            for p in meta["teams"][side]["players"]:
                self.roster[p["id"]] = {**p, "team": side}
                self.index_of[p["id"]] = p["index"]
        self.id_at = {v: k for k, v in self.index_of.items()}
        self.team_of = np.array([0] * 11 + [1] * 11)
        self.tracking = TrackingMetrics()
        self.auto = AutoEventer(self.team_of)
        self.team = {"home": _new_team_stats(), "away": _new_team_stats()}
        self.players = {pid: _new_player_stats() for pid in self.roster}
        self.score = [0, 0]
        self.t = 0.0
        self.period = 1
        self.clock = "00:00"
        self.events: dict[int, dict] = {}
        self.window: deque[dict] = deque()
        self.frame: np.ndarray | None = None
        self.pending: list[dict] = []
        self.poss: dict | None = None
        self.possessions: list[dict] = []
        self.indices: dict = {}
        self.index_history: list[dict] = []
        self.next_index_t = INDEX_EVERY
        self.moments: list[Moment] = []
        self._last_kind: dict[str, float] = {}
        self._moment_seq = 0
        self._cue_seq = 0
        self._milestones: set[tuple[str, str]] = set()

    # ---------- helpers ----------
    def name(self, pid: str | None) -> str | None:
        return self.roster[pid]["name"] if pid else None

    def team_name(self, side: str) -> str:
        return self.meta["teams"][side]["name"]

    def _score_str(self) -> str:
        return f"{self.score[0]}-{self.score[1]}"

    def _minute(self) -> int:
        return int(self.clock.split(":")[0]) + 1

    def _cue(self, kind: str, duration: float, payload: dict, anchor: dict | None = None,
             priority: int = 2, evidence: list[int] | None = None) -> dict:
        self._cue_seq += 1
        return {"id": f"c{self._cue_seq}", "lane": "fast", "type": kind, "t_in": round(self.t, 1),
                "t_out": round(self.t + duration, 1), "clock": self.clock, "anchor": anchor or {"screen": "auto"},
                "priority": priority, "payload": payload, "evidence_ids": evidence or []}

    def _moment(self, kind: str, team: str | None, players: list[str], importance: float, facts: dict,
                evidence: list[int], throttle: float = 0.0, key: str | None = None) -> Moment | None:
        key = key or kind
        if throttle and self.t - self._last_kind.get(key, -1e9) < throttle:
            return None
        self._last_kind[key] = self.t
        self._moment_seq += 1
        base = {"minute": self._minute(), "score": self._score_str(),
                "home_team": self.team_name("home"), "away_team": self.team_name("away")}
        if team:
            base["team"] = self.team_name(team)
        m = Moment(f"m{self._moment_seq}", kind, round(self.t, 1), self.clock, self._minute(), team,
                   [p for p in players if p], importance, base | facts, evidence)
        self.moments.append(m)
        return m

    # ---------- frames ----------
    def on_frame(self, t: float, period: int, xy: np.ndarray) -> tuple[list[dict], list[Moment]]:
        self.t, self.period, self.frame = t, period, xy
        cues: list[dict] = []
        moments: list[Moment] = []
        for trig in self.tracking.update(t, period, xy):
            pid = self.id_at[trig["index"]]
            if trig["type"] == "speed_alert":
                cues.append(self._cue("speed_alert", 4.0, {"player_id": pid, "name": self.name(pid),
                                                           "speed_kmh": trig["speed_kmh"],
                                                           "match_record": trig["match_record"]},
                                      anchor={"player_id": pid}, priority=2))
                if trig["match_record"] and trig["speed_kmh"] >= 31:
                    m = self._moment("speed_record", self.roster[pid]["team"], [pid], 0.45,
                                     {"player": self.name(pid), "top_speed_kmh": trig["speed_kmh"]}, [],
                                     throttle=600)
                    if m:
                        moments.append(m)
            else:
                cues.append(self._cue("distance_milestone", 4.0, {"player_id": pid, "name": self.name(pid),
                                                                  "km": trig["km"]},
                                      anchor={"player_id": pid}, priority=1))
        self.auto.update(t, xy)
        for p in list(self.pending):
            if t <= p["t0"]:
                continue
            p["max"] = max(p["max"], self.tracking.ball_speed)
            p["frames"] += 1
            if p["frames"] >= p["need"]:
                self.pending.remove(p)
                c, m = self._finish_ball_measure(p)
                cues += c
                moments += m
        if t >= self.next_index_t:
            self.next_index_t = t + INDEX_EVERY
            moments += self._update_indices()
        return cues, moments

    # ---------- events ----------
    def on_event(self, ev: dict) -> tuple[list[dict], list[Moment]]:
        ev = dict(ev)  # annotate a copy; the feed's events stay as delivered
        self.events[ev["id"]] = ev
        self.t = max(self.t, ev["t"])
        self.clock, self.period = ev["clock"], ev["period"]
        side = ev.get("team")
        pid = ev.get("player_id")
        cues: list[dict] = []
        moments: list[Moment] = []
        kind = ev["type"]
        if side and kind in ON_BALL:
            self._possession(side, ev)
        ts = self.team.get(side) if side else None
        ps = self.players.get(pid) if pid else None
        if ps is not None and kind in ON_BALL | {"tackle"}:
            ps["touches"] += 1

        if kind == "pass":
            self._on_pass(ev, ts, ps)
        elif kind == "shot":
            ti = side_index(side)
            under = False
            if self.frame is not None:
                opp = self.frame[(1 - ti) * 11 + 1:(1 - ti) * 11 + 11]
                under = float(np.min(np.linalg.norm(opp - np.array([ev["x"], ev["y"]]), axis=1))) < 1.5
            ev["xg"] = round(xg(ev["x"], ev["y"], ti, under, ev.get("body_part") == "head"), 3)
            ts["shots"] += 1
            ts["xg"] += ev["xg"]
            ps["shots"] += 1
            ps["xg"] += ev["xg"]
            if ev["outcome"] in ("goal", "saved"):
                ts["on_target"] += 1
            if self.poss:
                self.poss["xg"] += ev["xg"]
            self._key_pass(ev)
            self.pending.append({"id": ev["id"], "kind": "shot", "t0": ev["t"], "max": 0.0, "frames": 0, "need": 5})
        elif kind == "goal":
            self.score = list(ev["score"])
            ts["goals"] += 1
            ps["goals"] += 1
            if ev.get("assist_id"):
                self.players[ev["assist_id"]]["assists"] += 1
            moments.append(self._goal_moment(ev))
        elif kind == "pressure":
            ts["pressures"] += 1
            ps["pressures"] += 1
        elif kind == "tackle" and ev["outcome"] == "won":
            ts["tackles_won"] += 1
            ps["tackles_won"] += 1
        elif kind == "interception":
            ts["interceptions"] += 1
            ps["interceptions"] += 1
        elif kind == "ball_recovery":
            ts["recoveries"] += 1
            ps["recoveries"] += 1
        elif kind == "foul":
            ts["fouls"] += 1
            ps["fouls"] += 1
        elif kind == "corner":
            ts["corners"] += 1
        elif kind == "card":
            ps["yellow_cards"] += 1
            cues.append(self._cue("card", 6.0, {"player_id": pid, "name": self.name(pid), "card": ev["card"],
                                                "team": side}, anchor={"player_id": pid}, priority=3,
                                  evidence=[ev["id"]]))
        elif kind in ("period_end", "full_time"):
            self._close_possession(ev["t"], "period_end")
        if kind not in ("period_start", "period_end", "full_time"):
            self.window.append(ev)
        while self.window and self.window[0]["t"] < self.t - WINDOW:
            self.window.popleft()
        return cues, moments

    def _on_pass(self, ev: dict, ts: dict, ps: dict) -> None:
        ti = side_index(ev["team"])
        start, end = np.array([ev["x"], ev["y"]]), np.array([ev["end_x"], ev["end_y"]])
        feats = pass_features(start, end, ti, self.frame) if self.frame is not None else None
        ok = ev["outcome"] == "complete"
        ts["passes"] += 1
        ps["passes"] += 1
        if feats:
            xp = self.xpass.prob(feats)
            diff = self.xpass.difficulty(feats)
            ev.update({"length_m": round(feats["length"], 1), "difficulty": diff, "xpass": round(xp, 3),
                       "bypassed": feats["bypassed"], "forward_m": round(float(feats["forward_m"]), 1),
                       "lane_opponents": feats["lane_opponents"]})
            ts["xpass"] += xp
            ps["xpass"] += xp
            ps["difficulty_sum"] += diff
        if ok:
            ts["completed"] += 1
            ps["completed"] += 1
            if self.poss:
                self.poss["passes"] += 1
            u0, u1 = progress(ti, ev["x"]), progress(ti, ev["end_x"])
            if u0 < 2 / 3 <= u1:
                ts["final_third_entries"] += 1
            if u1 > 0.843 and abs(ev["end_y"] - W / 2) < 20.16:
                ts["box_entries"] += 1
            if feats and feats["forward_m"] >= 10 and u1 > 0.5:
                ts["progressive_passes"] += 1
                ps["progressive_passes"] += 1
            if feats and feats["bypassed"] >= 3:
                ts["line_breaking_passes"] += 1
                ps["line_breaking_passes"] += 1
            if feats:
                ps["hardest_pass"] = max(ps["hardest_pass"], ev["difficulty"])
        self.pending.append({"id": ev["id"], "kind": "pass", "t0": ev["t"], "max": 0.0, "frames": 0, "need": 6})
        for n in (25, 50, 75):
            if ps["completed"] == n and ok:
                self._queue_milestone(ev["player_id"], f"{n}_completed_passes", ev)

    def _queue_milestone(self, pid: str, what: str, ev: dict) -> None:
        if (pid, what) in self._milestones:
            return
        self._milestones.add((pid, what))
        ps = self.players[pid]
        acc = ps["completed"] / max(ps["passes"], 1)
        if what.endswith("completed_passes") and acc < 0.85:
            return
        self._moment("milestone", self.roster[pid]["team"], [pid], 0.35,
                     {"player": self.name(pid), "milestone": what.replace("_", " "),
                      "passes_completed": ps["completed"], "passes_attempted": ps["passes"],
                      "pass_accuracy_pct": round(100 * acc)}, [ev["id"]], throttle=240, key="milestone")

    def _key_pass(self, shot: dict) -> None:
        ids = sorted(self.events)
        for eid in reversed(ids[-12:]):
            e = self.events[eid]
            if e["type"] == "pass" and e.get("team") == shot["team"] and shot["t"] - e["t"] < 6:
                if e["outcome"] == "complete" and e.get("recipient_id") == shot["player_id"]:
                    self.players[e["player_id"]]["key_passes"] += 1
                    shot["key_pass_id"] = e["id"]
                return

    # ---------- ball-speed measurement and fast cues ----------
    def _finish_ball_measure(self, p: dict) -> tuple[list[dict], list[Moment]]:
        ev = self.events[p["id"]]
        kmh = round(p["max"] * 3.6, 1)
        ev["ball_speed_kmh"] = kmh
        cues, moments = [], []
        pid, side = ev["player_id"], ev["team"]
        if p["kind"] == "shot":
            cues.append(self._cue("shot_badge", 6.0, {"player_id": pid, "name": self.name(pid), "xg": ev["xg"],
                                                      "shot_speed_kmh": kmh, "outcome": ev["outcome"],
                                                      "team": side},
                                  anchor={"player_id": pid}, priority=3, evidence=[ev["id"]]))
            moments += self._shot_moments(ev)
        elif ev["outcome"] == "complete" and (ev.get("difficulty", 0) >= 50
                                              or (ev.get("bypassed", 0) >= 5 and ev.get("difficulty", 0) >= 25)):
            rid = ev.get("recipient_id")
            cues.append(self._cue("pass_badge", 5.0, {"player_id": pid, "name": self.name(pid),
                                                      "recipient": self.name(rid), "distance_m": ev.get("length_m"),
                                                      "difficulty": ev.get("difficulty"),
                                                      "bypassed": ev.get("bypassed"), "ball_speed_kmh": kmh,
                                                      "team": side},
                                  anchor={"player_id": rid or pid}, priority=2, evidence=[ev["id"]]))
            if (ev.get("bypassed", 0) >= 4 and ev.get("forward_m", 0) >= 20 and ev.get("difficulty", 0) >= 30
                    and self.roster[pid]["position"] != "GK"):
                m = self._moment("line_breaking_pass", side, [pid, rid], 0.5,
                                 {"passer": self.name(pid), "receiver": self.name(rid),
                                  "opponents_bypassed": ev["bypassed"], "pass_distance_m": ev["length_m"],
                                  "difficulty": ev["difficulty"], "completion_probability_pct": round(100 * ev["xpass"]),
                                  "ball_speed_kmh": kmh}, [ev["id"]], throttle=480)
                if m:
                    moments.append(m)
        return cues, moments

    def _shot_moments(self, shot: dict) -> list[Moment]:
        side, pid = shot["team"], shot["player_id"]
        ps = self.players[pid]
        facts = {"shooter": self.name(pid), "xg": shot["xg"],
                 "distance_m": round(float(np.hypot((L if side == "home" else 0) - shot["x"], W / 2 - shot["y"])), 1),
                 "team_shots": self.team[side]["shots"], "team_xg": round(self.team[side]["xg"], 2)}
        if shot["ball_speed_kmh"] >= 40:  # short flights (blocks, point-blank saves) under-measure speed
            facts["shot_speed_kmh"] = shot["ball_speed_kmh"]
        ev_ids = [shot["id"]]
        if shot.get("key_pass_id"):
            kp = self.events[shot["key_pass_id"]]
            facts["key_pass_by"] = self.name(kp["player_id"])
            ev_ids.insert(0, kp["id"])
        if shot["outcome"] == "goal":
            return []  # the goal moment is raised by the goal event, once the score has changed
        if shot["xg"] >= 0.25:
            kind = "big_save" if shot["outcome"] == "saved" else "big_chance"
            if kind == "big_save":
                gk = next(p for p in self.roster.values()
                          if p["team"] != side and p["position"] == "GK")
                facts["goalkeeper"] = gk["name"]
            facts["outcome"] = shot["outcome"].replace("_", " ")
            m = self._moment(kind, side, [pid], 0.7, facts, ev_ids)
            return [m] if m else []
        return []

    def _goal_moment(self, g: dict) -> Moment:
        side, pid = g["team"], g["player_id"]
        shot = next((e for e in reversed(list(self.events.values()))
                     if e["type"] == "shot" and e["player_id"] == pid), None)
        ps = self.players[pid]
        facts = {"scorer": self.name(pid), "team_shots": self.team[side]["shots"],
                 "team_xg": round(self.team[side]["xg"], 2)}
        ev_ids = [g["id"]]
        if shot:
            facts |= {"xg": shot["xg"], "distance_m": round(float(np.hypot((L if side == "home" else 0) - shot["x"], W / 2 - shot["y"])), 1)}
            if shot.get("ball_speed_kmh", 0) >= 40:
                facts["shot_speed_kmh"] = shot["ball_speed_kmh"]
            ev_ids.append(shot["id"])
        ti = side_index(side)
        mine, theirs = self.score[ti], self.score[1 - ti]
        state = ("opening goal" if mine + theirs == 1 else "equaliser" if mine == theirs
                 else "go-ahead goal" if mine == theirs + 1
                 else "pulls one back" if mine < theirs else "extends the lead")
        poss = self.poss or {}
        facts |= {"game_state": state, "player_goals": ps["goals"], "move_passes": poss.get("passes", 0),
                  "move_duration_s": round(self.t - poss.get("t0", self.t)),
                  "move_players": len(poss.get("players", [])),
                  "move_started": poss.get("start_desc", "open play")}
        if g.get("assist_id"):
            facts["assist"] = self.name(g["assist_id"])
        ev_ids += poss.get("ids", [])[-6:]
        return self._moment("goal", side, [pid, g.get("assist_id")], 1.0, facts, sorted(set(ev_ids)))

    # ---------- possessions ----------
    def _possession(self, side: str, ev: dict) -> None:
        if self.poss and self.poss["team"] == side:
            self.poss["ids"].append(ev["id"])
            if ev.get("player_id"):
                self.poss["players"].add(ev["player_id"])
            return
        self._close_possession(ev["t"], ev["type"])
        ti = side_index(side)
        third = progress(ti, ev.get("x", L / 2))
        zone = "own third" if third < 1 / 3 else "middle third" if third < 2 / 3 else "final third"
        how = {"interception": "an interception", "tackle": "a tackle", "ball_recovery": "a loose-ball recovery",
               "throw_in": "a throw-in", "goal_kick": "a goal kick", "corner": "a corner", "free_kick": "a free kick",
               "kickoff": "the kick-off", "save": "a save"}.get(ev["type"], "open play")
        prev = self.events.get(ev["id"] - 1)
        if ev["type"] == "pass" and prev and prev["type"] == "tackle" and prev.get("team") == side:
            how = "a tackle"
        self.poss = {"team": side, "t0": ev["t"], "passes": 0, "xg": 0.0, "ids": [ev["id"]],
                     "players": {ev["player_id"]} if ev.get("player_id") else set(),
                     "start_desc": f"{how} in the {zone}"}

    def _close_possession(self, t: float, ended_by: str) -> None:
        if not self.poss:
            return
        p = self.poss
        p["t1"], p["ended_by"] = t, ended_by
        self.team[p["team"]]["possession_s"] += t - p["t0"]
        self.possessions.append({k: (sorted(v) if isinstance(v, set) else v) for k, v in p.items()})
        if p["passes"] >= 14 and ended_by != "period_end":
            self._moment("long_spell", p["team"], [], 0.4,
                         {"passes_in_spell": p["passes"], "spell_duration_s": round(t - p["t0"]),
                          "players_involved": len(p["players"])}, p["ids"][-8:], throttle=420)
        self.poss = None

    # ---------- rolling indices ----------
    def window_metrics(self) -> dict:
        evs = list(self.window)
        mins = max(1.0, min(WINDOW, self.t) / 60)
        out: dict = {"minutes": round(mins, 1)}
        seq = [e["team"] for e in evs if e["type"] in ON_BALL and e.get("team")]
        changes = sum(a != b for a, b in zip(seq, seq[1:]))
        out["turnovers_pm"] = changes / mins
        n_passes = sum(1 for e in evs if e["type"] == "pass" and e["outcome"] == "complete")
        out["passes_per_poss"] = n_passes / max(changes + 1, 1)
        for side in ("home", "away"):
            ti = side_index(side)
            other = "away" if side == "home" else "home"
            mine = [e for e in evs if e.get("team") == side]
            passes = [e for e in mine if e["type"] == "pass"]
            press = [e for e in mine if e["type"] == "pressure"]
            def_actions = [e for e in mine if e["type"] in ("tackle", "interception", "foul")
                           and progress(ti, e["x"]) > 0.4]
            opp_build = [e for e in evs if e.get("team") == other and e["type"] == "pass"
                         and progress(1 - ti, e["x"]) < 0.6]
            seq_t = [(e["t"], e["team"]) for e in evs if e["type"] in ON_BALL and e.get("team")]
            poss_s = sum(b[0] - a[0] for a, b in zip(seq_t, seq_t[1:]) if a[1] == side)
            total_s = max(1.0, sum(b[0] - a[0] for a, b in zip(seq_t, seq_t[1:])))
            fwd = sum(max(0.0, e.get("forward_m", 0)) for e in passes)
            length = sum(e.get("length_m", 0) for e in passes) or 1.0
            shots = [e for e in mine if e["type"] == "shot"]
            entries = sum(1 for e in passes if e["outcome"] == "complete"
                          and progress(ti, e["x"]) < 2 / 3 <= progress(ti, e["end_x"]))
            out[side] = {
                "possession_share": poss_s / total_s,
                "pressures_pm": len(press) / mins,
                "press_height_m": (float(np.mean([progress(ti, e["x"]) for e in press])) * L) if press else 0.0,
                "ppda": len(opp_build) / max(len(def_actions), 1),
                "tempo": len(passes) / max(poss_s / 60, 0.5),
                "directness": fwd / length,
                "pass_accuracy": (sum(e["outcome"] == "complete" for e in passes) / len(passes)) if passes else 0.0,
                "xg": sum(e.get("xg", 0) for e in shots),
                "shots": len(shots),
                "final_third_entries": entries,
            }
        return out

    def _update_indices(self) -> list[Moment]:
        if self.t < 120:
            return []
        w = self.window_metrics()
        b = self.base
        chaos = 0.5 * b.pct("turnovers_pm", w["turnovers_pm"], 3) + 0.5 * (100 - b.pct("passes_per_poss", w["passes_per_poss"], 3))
        idx = {"t": round(self.t, 1), "clock": self.clock, "chaos": round(chaos), "raw": w}
        threat = {}
        for side in ("home", "away"):
            s = w[side]
            press = (b.pct("pressures_pm", s["pressures_pm"], 3) + b.pct("press_height_m", s["press_height_m"], 50)
                     + (100 - b.pct("ppda", s["ppda"], 8))) / 3
            idx[side] = {"press": round(press), "tempo": round(b.pct("tempo", s["tempo"], 12)),
                         "directness": round(b.pct("directness", s["directness"], 0.4)),
                         "control": round(100 * s["possession_share"])}
            threat[side] = s["xg"] + 0.03 * s["final_third_entries"]
        idx["momentum"] = round(100 * (threat["home"] - threat["away"]) / (threat["home"] + threat["away"] + 0.15))
        self.indices = idx
        self.index_history.append(idx)
        return self._index_moments(idx)

    def _ago(self, seconds: float) -> dict | None:
        target = self.t - seconds
        for h in reversed(self.index_history):
            if h["t"] <= target:
                return h
        return None

    def _index_moments(self, now: dict) -> list[Moment]:
        out: list[Moment] = []
        before = self._ago(300)
        if before is None:
            return out
        for side in ("home", "away"):
            d_press = now[side]["press"] - before[side]["press"]
            if d_press >= 28 and now[side]["press"] >= 72:
                r, r0 = now["raw"][side], before["raw"][side]
                m = self._moment("press_shift", side, [], 0.75, {
                    "press_index_now": now[side]["press"], "press_index_before": before[side]["press"],
                    "pressures_per_min_now": round(r["pressures_pm"], 1),
                    "pressures_per_min_before": round(r0["pressures_pm"], 1),
                    "press_height_m_now": round(r["press_height_m"]), "press_height_m_before": round(r0["press_height_m"]),
                    "ppda_now": round(r["ppda"], 1), "ppda_before": round(r0["ppda"], 1),
                    "window_min": 5}, self._window_ids("pressure", side), throttle=600, key=f"press_shift_{side}")
                if m:
                    out.append(m)
            d_tempo = now[side]["tempo"] - before[side]["tempo"]
            if abs(d_tempo) >= 55:
                r, r0 = now["raw"][side], before["raw"][side]
                m = self._moment("tempo_shift", side, [], 0.5, {
                    "direction": "up" if d_tempo > 0 else "down",
                    "tempo_index_now": now[side]["tempo"], "tempo_index_before": before[side]["tempo"],
                    "passes_per_possession_min_now": round(r["tempo"], 1),
                    "passes_per_possession_min_before": round(r0["tempo"], 1),
                    "directness_pct_now": round(100 * r["directness"]),
                    "window_min": 5}, self._window_ids("pass", side), throttle=900, key=f"tempo_{side}")
                if m:
                    out.append(m)
        prev5 = self._ago(300)
        if prev5 and now["chaos"] >= 80 and prev5["chaos"] < 60:
            r, r0 = now["raw"], prev5["raw"]
            m = self._moment("chaos_spell", None, [], 0.55, {
                "chaos_index_now": now["chaos"], "chaos_index_before": prev5["chaos"],
                "turnovers_per_min_now": round(r["turnovers_pm"], 1),
                "turnovers_per_min_before": round(r0["turnovers_pm"], 1),
                "passes_per_possession_now": round(r["passes_per_poss"], 1),
                "window_min": 5}, self._window_ids("interception", None) + self._window_ids("tackle", None),
                throttle=600)
            if m:
                out.append(m)
        if prev5 and now["chaos"] <= 25 and prev5["chaos"] >= 50:
            leader = max(("home", "away"), key=lambda s: now[s]["control"])
            if now[leader]["control"] >= 60:
                m = self._moment("control_regained", leader, [], 0.5, {
                    "chaos_index_now": now["chaos"], "chaos_index_before": prev5["chaos"],
                    "possession_share_pct": now[leader]["control"],
                    "passes_per_possession_now": round(now["raw"]["passes_per_poss"], 1),
                    "pass_accuracy_pct": round(100 * now["raw"][leader]["pass_accuracy"]), "window_min": 5},
                    self._window_ids("pass", leader), throttle=600)
                if m:
                    out.append(m)
        older = self._ago(600)
        if older and abs(now["momentum"]) >= 45 and np.sign(now["momentum"]) != np.sign(older["momentum"]) \
                and abs(older["momentum"]) >= 30:
            side = "home" if now["momentum"] > 0 else "away"
            r = now["raw"][side]
            m = self._moment("momentum_swing", side, [], 0.65, {
                "momentum_now": abs(now["momentum"]), "momentum_10_min_ago_other_way": abs(older["momentum"]),
                "xg_last_5_min": round(r["xg"], 2), "shots_last_5_min": r["shots"],
                "final_third_entries_last_5_min": r["final_third_entries"], "window_min": 5},
                self._window_ids("shot", side), throttle=600)
            if m:
                out.append(m)
        return out

    def _window_ids(self, kind: str, side: str | None, limit: int = 12) -> list[int]:
        ids = [e["id"] for e in self.window if e["type"] == kind and (side is None or e.get("team") == side)]
        return ids[-limit:]

    # ---------- snapshots ----------
    def player_rating(self, pid: str) -> float:
        p = self.players[pid]
        r = (6.0 + 0.9 * p["goals"] + 0.6 * p["assists"] + 0.5 * p["xg"] + 0.2 * p["key_passes"]
             + 0.03 * (p["completed"] - p["xpass"]) + 0.05 * p["tackles_won"] + 0.05 * p["interceptions"]
             + 0.015 * p["progressive_passes"] + 0.03 * p["line_breaking_passes"] - 0.3 * p["yellow_cards"])
        return round(float(6.0 + 3.8 * np.tanh((r - 6.0) / 3.5)), 1)  # squashed onto a familiar 2-10 scale

    def player_snapshot(self, pid: str) -> dict:
        p = self.players[pid]
        i = self.index_of[pid]
        tr = self.tracking
        return {"id": pid, "name": self.name(pid), "team": self.roster[pid]["team"],
                "number": self.roster[pid]["number"], "position": self.roster[pid]["position"],
                **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()},
                "pass_accuracy_pct": round(100 * p["completed"] / p["passes"]) if p["passes"] else None,
                "avg_difficulty": round(p["difficulty_sum"] / p["passes"]) if p["passes"] else None,
                "passes_over_expected": round(p["completed"] - p["xpass"], 1),
                "distance_km": round(float(tr.distance[i]) / 1000, 2),
                "high_intensity_km": round(float(tr.hi_distance[i]) / 1000, 2),
                "top_speed_kmh": round(float(tr.top_speed[i]) * 3.6, 1),
                "sprints": int(tr.sprints[i]), "speed_kmh": round(float(tr.speed[i]) * 3.6, 1),
                "rating": self.player_rating(pid)}

    def snapshot(self) -> dict:
        teams = {}
        total = sum(self.team[s]["possession_s"] for s in ("home", "away")) or 1.0
        for side in ("home", "away"):
            s = self.team[side]
            teams[side] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()} | {
                "possession_pct": round(100 * s["possession_s"] / total),
                "pass_accuracy_pct": round(100 * s["completed"] / s["passes"]) if s["passes"] else None,
                "distance_km": round(float(self.tracking.distance[side_index(side) * 11:side_index(side) * 11 + 11].sum()) / 1000, 1),
            }
        return {"t": round(self.t, 1), "clock": self.clock, "period": self.period, "score": list(self.score),
                "teams": teams, "indices": {k: v for k, v in self.indices.items() if k != "raw"}}
