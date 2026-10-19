"""Fictional clubs, squads and formations for the synthetic league.

Every club, player, stadium and referee here is invented. Nothing is derived
from real Premier League data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    number: int
    position: str
    pace: float
    passing: float
    finishing: float
    tackling: float
    dribbling: float

    @property
    def short(self) -> str:
        return self.name.split()[-1]


@dataclass(frozen=True)
class Team:
    side: str  # "home" | "away"
    name: str
    short: str
    color: str
    alt_color: str
    formation: str
    players: tuple[Player, ...]
    tactics: dict = field(default_factory=dict)


# Formation slots in team-normalised coordinates, GK first.
# du: depth offset from the team's block centre (fraction of pitch length,
#     positive = towards the opponent goal); v: lateral position 0..1.
FORMATIONS: dict[str, list[tuple[float, float]]] = {
    "4-3-3": [
        (0.0, 0.5),
        (-0.17, 0.12), (-0.21, 0.37), (-0.21, 0.63), (-0.17, 0.88),
        (-0.02, 0.30), (-0.07, 0.50), (-0.02, 0.70),
        (0.17, 0.15), (0.20, 0.50), (0.17, 0.85),
    ],
    "4-2-3-1": [
        (0.0, 0.5),
        (-0.17, 0.12), (-0.21, 0.37), (-0.21, 0.63), (-0.17, 0.88),
        (-0.08, 0.38), (-0.08, 0.62),
        (0.08, 0.18), (0.09, 0.50), (0.08, 0.82),
        (0.21, 0.50),
    ],
}


def _p(side: str, n: int, name: str, pos: str, pace, passing, fin, tack, drib) -> Player:
    return Player(f"{side[0].upper()}{n:02d}", name, n, pos, pace, passing, fin, tack, drib)


KINGSMOOR = Team(
    side="home",
    name="Kingsmoor FC",
    short="KMR",
    color="#d64545",
    alt_color="#ffffff",
    formation="4-3-3",
    players=(
        _p("home", 1, "Aaron Lindqvist", "GK", 0.30, 0.55, 0.05, 0.20, 0.20),
        _p("home", 2, "Kofi Adjei", "RB", 0.82, 0.66, 0.30, 0.70, 0.62),
        _p("home", 5, "Marcus Hale", "CB", 0.58, 0.62, 0.25, 0.86, 0.40),
        _p("home", 6, "Jonas Weber", "CB", 0.55, 0.70, 0.22, 0.82, 0.38),
        _p("home", 3, "Theo Marchetti", "LB", 0.80, 0.68, 0.28, 0.68, 0.60),
        _p("home", 8, "Rafael Duarte", "CM", 0.66, 0.84, 0.55, 0.62, 0.74),
        _p("home", 4, "Elliot Crane", "DM", 0.58, 0.80, 0.35, 0.80, 0.55),
        _p("home", 10, "Yusuf Demir", "CM", 0.70, 0.88, 0.66, 0.50, 0.82),
        _p("home", 7, "Sami Haddad", "RW", 0.93, 0.72, 0.70, 0.30, 0.86),
        _p("home", 9, "Nico Vargas", "ST", 0.78, 0.66, 0.88, 0.32, 0.72),
        _p("home", 11, "Leo Okonkwo", "LW", 0.90, 0.70, 0.72, 0.30, 0.84),
    ),
    tactics={"press": 0.58, "tempo": 0.62, "directness": 0.32},
)

HARBOURSIDE = Team(
    side="away",
    name="Harbourside Albion",
    short="HBA",
    color="#2f6fdb",
    alt_color="#f2c14e",
    formation="4-2-3-1",
    players=(
        _p("away", 1, "Mateo Rinaldi", "GK", 0.32, 0.58, 0.05, 0.20, 0.20),
        _p("away", 2, "Owen Pryce", "RB", 0.78, 0.64, 0.28, 0.72, 0.58),
        _p("away", 4, "Dario Kovač", "CB", 0.60, 0.60, 0.30, 0.88, 0.36),
        _p("away", 5, "Ibrahim Sow", "CB", 0.66, 0.64, 0.26, 0.84, 0.42),
        _p("away", 3, "Felix Bauer", "LB", 0.76, 0.66, 0.26, 0.70, 0.56),
        _p("away", 6, "Hiro Tanaka", "DM", 0.62, 0.82, 0.40, 0.78, 0.60),
        _p("away", 8, "Andre Costa", "DM", 0.64, 0.76, 0.45, 0.74, 0.58),
        _p("away", 7, "Jamal Brooks", "RAM", 0.91, 0.70, 0.68, 0.34, 0.84),
        _p("away", 10, "Emil Sørensen", "AM", 0.72, 0.90, 0.72, 0.42, 0.86),
        _p("away", 11, "Kai Moreno", "LAM", 0.88, 0.70, 0.66, 0.32, 0.82),
        _p("away", 9, "Viktor Hallberg", "ST", 0.74, 0.62, 0.90, 0.36, 0.66),
    ),
    tactics={"press": 0.40, "tempo": 0.45, "directness": 0.55},
)

VENUE = "Kingsmoor Park"
COMPETITION = "Synthetic Premier Division (fictional)"
REFEREE = "R. Ashdown"
