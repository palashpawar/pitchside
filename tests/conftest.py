import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchside.intel.engine import MatchEngine  # noqa: E402
from pitchside.sim.simulator import simulate  # noqa: E402
from pitchside.stream import run_engine  # noqa: E402


@pytest.fixture(scope="session")
def match():
    return simulate(seed=3, half_length=900.0)


@pytest.fixture(scope="session")
def engine_run(match):
    eng = MatchEngine(match.meta)
    cues, moments = run_engine(match, eng)
    return eng, cues, moments
