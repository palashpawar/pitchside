"""Deterministic grounding checks used by the Verifier.

Every number in generated text must trace to a value in the moment's facts,
and every player or club named must appear in those facts. This is what makes
the insights explainable: each sentence is backed by evidence the UI can show.
"""
from __future__ import annotations

import re

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")  # also catches ordinals like "27th"


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def allowed_numbers(facts: dict) -> set[float]:
    out: set[float] = set()
    for v in _walk(facts):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            f = float(v)
            out |= {f, float(round(f)), round(f, 1), round(f, 2), float(int(f))}
            if 0 < abs(f) < 1:  # proportions may be quoted as percentages
                out |= {float(round(100 * f)), round(100 * f, 1)}
        elif isinstance(v, str):
            for n in NUM_RE.findall(v):
                out.add(float(n.replace(",", ".")))
    return out


def fact_strings(facts: dict) -> str:
    return " ".join(str(v) for v in _walk(facts) if isinstance(v, str)).lower()


def check(text: str, facts: dict, roster_names: list[str], team_names: list[str]) -> list[str]:
    """Return a list of grounding problems (empty means the text is grounded)."""
    issues = []
    allowed = allowed_numbers(facts)
    for raw in NUM_RE.findall(text):
        n = float(raw.replace(",", "."))
        if not any(abs(n - a) < 0.051 for a in allowed):
            issues.append(f"number {raw} is not in the facts")
    known = fact_strings(facts)
    low = text.lower()
    for name in roster_names:
        surname = name.split()[-1].lower()
        mentioned = name.lower() in low or re.search(rf"\b{re.escape(surname)}\b", low)
        if mentioned and surname not in known:
            issues.append(f"{name} is not part of this moment")
    for team in team_names:
        if team.lower() in low and team.lower() not in known:
            issues.append(f"{team} is not part of this moment")
    return issues
