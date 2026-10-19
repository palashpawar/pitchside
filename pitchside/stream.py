"""Replay a stored or freshly simulated match as one time-ordered stream of events and frames."""
from __future__ import annotations

from typing import Iterator

from .sim.simulator import MatchData

STREAM_EVENTS_EXCLUDE = {"period_start"}


def iter_match(match: MatchData, start_index: int = 0) -> Iterator[tuple[str, object]]:
    """Yield ("event", dict) and ("frame", index) in match-time order.

    Events stamped at time t are released just before the frame recorded at t,
    which mirrors a live feed where the event message and the tracking packet
    for the same instant arrive together.
    """
    events = sorted(match.events, key=lambda e: (e["t"], e["id"]))
    k = 0
    t0 = float(match.t[start_index]) if start_index else -1.0
    while k < len(events) and events[k]["t"] < t0:
        k += 1
    for i in range(start_index, len(match.t)):
        t = float(match.t[i])
        while k < len(events) and events[k]["t"] <= t + 1e-6:
            yield "event", events[k]
            k += 1
        yield "frame", i
    while k < len(events):
        yield "event", events[k]
        k += 1


def run_engine(match: MatchData, engine) -> tuple[list[dict], list]:
    """Run a whole match through an engine offline; returns (cues, moments)."""
    cues, moments = [], []
    for kind, item in iter_match(match):
        if kind == "event":
            c, m = engine.on_event(item)
        else:
            c, m = engine.on_frame(float(match.t[item]), int(match.period[item]), match.xy[item])
        cues += c
        moments += m
    return cues, moments
