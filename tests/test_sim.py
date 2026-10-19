import numpy as np

from pitchside.football import L, W
from pitchside.sim.simulator import simulate


def test_same_seed_same_match(match):
    again = simulate(seed=3, half_length=900.0)
    assert again.events == match.events
    assert np.array_equal(again.xy, match.xy)


def test_tracking_shape_and_bounds(match):
    assert match.xy.shape[1:] == (23, 2)
    assert len(match.t) == len(match.xy) == len(match.period)
    assert match.xy[:, :, 0].min() > -3 and match.xy[:, :, 0].max() < L + 3
    assert match.xy[:, :, 1].min() > -3 and match.xy[:, :, 1].max() < W + 3


def test_player_speeds_are_human(match):
    same_half = np.diff(match.period) == 0
    v = np.linalg.norm(np.diff(match.xy[:, :22], axis=0), axis=2)[same_half] * 10
    assert np.percentile(v, 99.9) < 10.5  # nobody outruns ~38 km/h


def test_event_stream_is_ordered_and_complete(match):
    ts = [e["t"] for e in match.events]
    assert ts == sorted(ts)
    kinds = {e["type"] for e in match.events}
    assert {"pass", "pressure", "shot", "tackle", "period_start", "full_time"} <= kinds
    passes = [e for e in match.events if e["type"] == "pass"]
    rate = sum(e["outcome"] == "complete" for e in passes) / len(passes)
    assert 0.7 < rate < 0.9
    for e in passes:
        assert {"x", "y", "end_x", "end_y", "outcome", "player_id", "team"} <= e.keys()


def test_meta_is_fictional_and_flagged(match):
    assert match.meta["synthetic"] is True
    assert "fictional" in match.meta["competition"]
