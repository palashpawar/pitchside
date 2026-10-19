import asyncio

from pitchside.agents import pipeline as P

PROFILES = [{"mode": "casual", "club": "home", "language": "es", "player_id": "H09"},
            {"mode": "analyst", "language": "en"}]


def run(coro):
    return asyncio.run(coro)


def test_offline_pipeline_verifies_every_moment(match, engine_run):
    _, _, moments = engine_run
    pipe = P.InsightPipeline(match.meta)
    cues = []
    for m in moments:
        cues += run(pipe.process(m.to_dict(), PROFILES))
    assert cues and all(c["verified"] for c in cues)
    assert pipe.stats["fallbacks"] == 0
    langs = {c["language"] for c in cues}
    assert langs == {"es", "en"}
    for c in cues:
        assert c["t_out"] > c["t_in"] and c["persona"] and c["overlay"]


class FakeRunner(P.AgentRunner):
    """Stands in for a model: the analyst hallucinates on its first draft."""

    def __init__(self, analyst_replies):
        self.client, self.agents, self.calls, self.cache_hits = object(), {}, 0, 0
        self.analyst_replies = list(analyst_replies)
        self.seen = []

    async def ask(self, name, payload):
        self.calls += 1
        self.seen.append((name, payload))
        if name == "analyst":
            return self.analyst_replies.pop(0)
        return {"overlay": payload["insight"]["headline"], "line": payload["insight"]["why"]}


def _team_moment(engine_run):
    _, _, moments = engine_run
    return next(m.to_dict() for m in moments if m.team)


def test_verifier_sends_feedback_and_analyst_corrects(match, engine_run):
    m = _team_moment(engine_run)
    team = m["facts"]["team"]
    runner = FakeRunner([{"headline": f"{team} again", "why": "Their 4417th win in 9999 games."},
                         {"headline": f"{team} on top", "why": f"It is {m['facts']['score']} after {m['facts']['minute']} minutes."}])
    pipe = P.InsightPipeline(match.meta, runner)
    cues = run(pipe.process(m, [PROFILES[1]]))
    assert len(cues) == 1 and cues[0]["verified"]
    analyst_calls = [p for n, p in runner.seen if n == "analyst"]
    assert len(analyst_calls) == 2
    assert any("4417" in issue for issue in analyst_calls[1]["previous_draft_problems"])
    assert pipe.stats["first_pass_grounded"] == 0 and pipe.stats["fallbacks"] == 0


def test_persistent_hallucination_falls_back_to_template(match, engine_run):
    m = _team_moment(engine_run)
    bad = {"headline": "Record breaker", "why": "A 99th-minute classic."}
    pipe = P.InsightPipeline(match.meta, FakeRunner([bad, bad]))
    cues = run(pipe.process(m, [PROFILES[1]]))
    assert cues[0]["verified"] and pipe.stats["fallbacks"] == 1
    assert any(c.get("attempt") == "fallback" for c in cues[0]["checks"])


def test_casual_profile_filters_minor_moments(match, engine_run):
    _, _, moments = engine_run
    minor = next(m.to_dict() for m in moments if m.importance < 0.55 and not m.player_ids)
    pipe = P.InsightPipeline(match.meta)
    assert run(pipe.process(minor, [{"mode": "casual", "language": "en"}])) == []
