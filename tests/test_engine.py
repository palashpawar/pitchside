import numpy as np

from pitchside.intel.features import XPassModel


def test_stats_reconcile_with_events(match, engine_run):
    eng, _, _ = engine_run
    for side in ("home", "away"):
        passes = [e for e in match.events if e["type"] == "pass" and e["team"] == side]
        assert eng.team[side]["passes"] == len(passes)
        assert eng.team[side]["completed"] == sum(e["outcome"] == "complete" for e in passes)
        goals = sum(1 for e in match.events if e["type"] == "goal" and e["team"] == side)
        assert eng.team[side]["goals"] == goals
    assert eng.score == match.events[-1]["score"]


def test_pass_quality_annotations(engine_run):
    eng, _, _ = engine_run
    passes = [e for e in eng.events.values() if e["type"] == "pass" and "difficulty" in e]
    assert passes
    for e in passes:
        assert 0 <= e["difficulty"] <= 100 and 0 < e["xpass"] < 1
    # harder passes should fail more often
    hard = [e for e in passes if e["difficulty"] >= 40]
    easy = [e for e in passes if e["difficulty"] < 15]
    rate = lambda xs: sum(e["outcome"] == "complete" for e in xs) / len(xs)
    assert rate(hard) < rate(easy)


def test_ball_speed_measured_for_shots(engine_run):
    eng, cues, _ = engine_run
    shots = [e for e in eng.events.values() if e["type"] == "shot"]
    assert all("ball_speed_kmh" in s and "xg" in s for s in shots)
    assert any(c["type"] == "shot_badge" for c in cues)


def test_moments_are_grounded_structures(engine_run):
    _, _, moments = engine_run
    assert moments
    for m in moments:
        assert m.facts["minute"] >= 1 and "score" in m.facts
        assert 0 < m.importance <= 1
        assert all(isinstance(i, int) for i in m.evidence_ids)


def test_physical_metrics(engine_run):
    eng, _, _ = engine_run
    km = eng.tracking.distance / 1000
    assert km[1:11].min() > 1.0  # 30 minutes of football
    assert eng.tracking.top_speed.max() * 3.6 > 25


def test_snapshot_shape(engine_run):
    eng, _, _ = engine_run
    s = eng.snapshot()
    assert set(s["teams"]) == {"home", "away"}
    assert sum(s["teams"][k]["possession_pct"] for k in ("home", "away")) in (99, 100, 101)
    p = eng.player_snapshot("H09")
    assert 2 <= p["rating"] <= 10


def test_xpass_fit_recovers_signal():
    rng = np.random.default_rng(0)
    X = np.c_[np.ones(4000), rng.normal(size=4000)]
    y = (rng.random(4000) < 1 / (1 + np.exp(-(1.0 - 2.0 * X[:, 1])))).astype(float)
    w = XPassModel.fit(X, y)
    assert abs(w[0] - 1.0) < 0.2 and abs(w[1] + 2.0) < 0.2


def test_tracking_handles_lower_frame_rates(match):
    from pitchside.intel.tracking import TrackingMetrics
    full, half = TrackingMetrics(), TrackingMetrics()
    for i in range(len(match.t)):
        args = (float(match.t[i]), int(match.period[i]), match.xy[i])
        full.update(*args)
        if i % 2 == 0:
            half.update(*args)
    ratio = half.distance[1:11].sum() / full.distance[1:11].sum()
    assert 0.9 < ratio <= 1.0  # coarser sampling cuts corners slightly, never inflates
