"""Multi-agent insight pipeline, orchestrated as a Microsoft Agent Framework workflow.

    moment ──► Analyst ──► Verifier ──► Persona ──► insight cues (one per audience)
                  ▲            │
                  └─feedback───┘   (ungrounded draft: retry with the exact problems,
                                    then fall back to a grounded template)

* Analyst agent explains *why* the moment matters, using only the moment's facts.
* Verifier checks every number and name against the facts (deterministic, see grounding.py).
* Persona agent rewrites the verified insight for each audience profile: mode
  (casual/analyst), favourite club, focus player, language and metric of interest;
  each rewrite is verified again.
* Recap agent writes half-time and full-time recaps from the verified insights.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler

from . import grounding, offline
from .llm import ResponseCache, chat_client, parse_json, provider

MAX_ATTEMPTS = 2
LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "pt": "Portuguese", "de": "German",
             "ar": "Arabic", "hi": "Hindi", "ja": "Japanese", "sw": "Swahili", "id": "Indonesian"}

ANALYST_INSTRUCTIONS = """You are the lead analyst for a live football broadcast.
You receive one detected match moment as JSON: its kind and a facts object.
Explain WHY the moment matters (tactical cause, consequence, or game context), not just that it happened.
Rules:
- Use ONLY numbers and names that appear in facts. Never invent statistics, players, history or quotes.
- headline: at most 8 words. why: one or two sentences, at most 45 words.
- Respond with JSON only: {"headline": "...", "why": "..."}"""

PERSONA_INSTRUCTIONS = """You adapt verified football insights for one viewer's on-screen experience.
You receive a verified insight, its facts, and the viewer profile.
- mode "casual": plain, warm, energetic language. No jargon: say "chance quality" instead of xG, "winning the ball back" instead of PPDA.
- mode "analyst": precise, tactical, metric-led.
- favourite_club: frame it from that fan's side (excited when it helps them, wary when it hurts them) without inventing facts.
- focus_player: if that player is in the facts, centre the line on them.
- metric: if that metric is in the facts, lead with it.
- Write in the requested language. Keep numbers as digits and player and club names unchanged.
- Use ONLY numbers and names present in the facts.
- overlay: at most 60 characters, for a lower-third graphic. line: one sentence, at most 28 words, for commentary or text-to-speech.
Respond with JSON only: {"overlay": "...", "line": "..."}"""

RECAP_INSTRUCTIONS = """You write match recaps for a football studio and streaming apps.
You receive the match stats, the verified key moments, and a viewer profile.
Use ONLY numbers and names present in the input. Never invent events.
Write in the requested language, in the register the profile's mode asks for (casual or analyst).
Respond with JSON only:
{"headline": "...", "summary": "3-4 sentences", "key_moments": ["clock - one line", ...], "player_of_the_match": "name - one-line reason", "social_post": "under 240 characters"}"""


def profile_key(p: dict) -> str:
    return "|".join(str(p.get(k) or "-") for k in ("mode", "club", "player_id", "language", "metric"))


@dataclass
class MomentJob:
    moment: dict
    personas: list[dict]
    attempt: int = 0
    feedback: list[str] = field(default_factory=list)


@dataclass
class Draft:
    job: MomentJob
    insight: dict
    source: str


@dataclass
class VerifiedInsight:
    job: MomentJob
    insight: dict
    source: str
    checks: list[dict]


class AgentRunner:
    """Runs a named agent with caching; None client means offline."""

    def __init__(self, cache: ResponseCache | None = None, concurrency: int = 4):
        self.client = chat_client()
        self.cache = cache or ResponseCache()
        self.sem = asyncio.Semaphore(concurrency)
        self.agents: dict[str, Agent] = {}
        self.calls = 0
        self.cache_hits = 0
        if self.client is not None:
            for name, ins in (("analyst", ANALYST_INSTRUCTIONS), ("persona", PERSONA_INSTRUCTIONS),
                              ("recap", RECAP_INSTRUCTIONS)):
                self.agents[name] = Agent(self.client, ins, name=name)

    @property
    def online(self) -> bool:
        return self.client is not None

    async def ask(self, name: str, payload: dict) -> dict:
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        k = ResponseCache.key(provider(), name, prompt)
        hit = self.cache.get(k)
        if hit is not None:
            self.cache_hits += 1
            return parse_json(hit)
        async with self.sem:
            self.calls += 1
            reply = await self.agents[name].run(prompt)
        text = reply.text
        out = parse_json(text)
        self.cache.put(k, text)
        return out


class AnalystExecutor(Executor):
    def __init__(self, runner: AgentRunner):
        super().__init__(id="analyst")
        self.runner = runner

    @handler
    async def draft(self, job: MomentJob, ctx: WorkflowContext[Draft]) -> None:
        m = job.moment
        if not self.runner.online:
            await ctx.send_message(Draft(job, offline.analyst(m["kind"], m["facts"]), "template"))
            return
        payload = {"kind": m["kind"], "facts": m["facts"]}
        if job.feedback:
            payload["previous_draft_problems"] = job.feedback
        try:
            out = await self.runner.ask("analyst", payload)
            insight = {"headline": str(out.get("headline", "")), "why": str(out.get("why", ""))}
        except Exception as exc:  # model or parse failure: let the verifier route to the fallback
            insight = {"headline": "", "why": "", "error": str(exc)[:200]}
        await ctx.send_message(Draft(job, insight, "model"))


class VerifierExecutor(Executor):
    def __init__(self, roster_names: list[str], team_names: list[str]):
        super().__init__(id="verifier")
        self.roster_names, self.team_names = roster_names, team_names

    def problems(self, text: str, facts: dict) -> list[str]:
        if not text.strip():
            return ["empty output"]
        return grounding.check(text, facts, self.roster_names, self.team_names)

    @handler
    async def verify(self, d: Draft, ctx: WorkflowContext[MomentJob | VerifiedInsight]) -> None:
        facts = d.job.moment["facts"]
        issues = d.insight.get("error") and ["model error"] or \
            self.problems(f"{d.insight['headline']} {d.insight['why']}", facts)
        check = {"stage": "analyst", "attempt": d.job.attempt + 1, "source": d.source, "issues": issues}
        checks = d.job.moment.setdefault("_checks", [])
        checks.append(check)
        if not issues:
            await ctx.send_message(VerifiedInsight(d.job, d.insight, d.source, checks))
        elif d.job.attempt + 1 < MAX_ATTEMPTS:
            await ctx.send_message(MomentJob(d.job.moment, d.job.personas, d.job.attempt + 1, issues))
        else:
            fallback = offline.analyst(d.job.moment["kind"], facts)
            checks.append({"stage": "analyst", "attempt": "fallback", "source": "template", "issues": []})
            await ctx.send_message(VerifiedInsight(d.job, fallback, "template-fallback", checks))


class PersonaExecutor(Executor):
    def __init__(self, runner: AgentRunner, verifier: VerifierExecutor, roster: dict[str, dict]):
        super().__init__(id="persona")
        self.runner, self.verifier, self.roster = runner, verifier, roster

    @staticmethod
    def relevant(m: dict, p: dict) -> bool:
        if p.get("player_id") and p["player_id"] in m["player_ids"]:
            return True
        if p.get("mode") == "casual":
            return m["importance"] >= 0.55
        return True

    async def personalise(self, v: VerifiedInsight, p: dict) -> dict:
        m = v.job.moment
        facts = m["facts"]
        involved = bool(p.get("player_id") and p["player_id"] in m["player_ids"])
        favours = None if not p.get("club") or not m.get("team") else p["club"] == m["team"]
        checks = []
        out = None
        if self.runner.online:
            payload = {"insight": v.insight, "facts": facts, "kind": m["kind"], "profile": {
                "mode": p.get("mode", "casual"),
                "favourite_club": facts.get("home_team") if p.get("club") == "home"
                else facts.get("away_team") if p.get("club") == "away" else None,
                "focus_player": self.roster[p["player_id"]]["name"] if p.get("player_id") else None,
                "metric": p.get("metric"), "language": LANGUAGES.get(p.get("language", "en"), "English")}}
            for attempt in range(MAX_ATTEMPTS):
                try:
                    res = await self.runner.ask("persona", payload)
                    cand = {"overlay": str(res.get("overlay", "")), "line": str(res.get("line", ""))}
                    issues = self.verifier.problems(f"{cand['overlay']} {cand['line']}", facts)
                except Exception as exc:
                    cand, issues = None, [f"model error: {str(exc)[:120]}"]
                checks.append({"stage": "persona", "attempt": attempt + 1, "source": "model", "issues": issues})
                if not issues:
                    out = cand
                    break
                payload["previous_draft_problems"] = issues
        if out is None:
            out = offline.persona(m["kind"], facts, v.insight, p, involved, favours)
            checks.append({"stage": "persona", "attempt": "template", "source": "template",
                           "issues": self.verifier.problems(f"{out['overlay']} {out['line']}", facts)})
        return {
            "id": f"{m['id']}:{profile_key(p)}", "lane": "agent", "type": "insight", "moment_id": m["id"],
            "kind": m["kind"], "t_in": m["t"], "t_out": m["t"] + (12 if m["importance"] >= 0.7 else 8),
            "clock": m["clock"], "priority": 3 if m["importance"] >= 0.7 else 2,
            "anchor": {"player_id": m["player_ids"][0]} if m["player_ids"] else {"screen": "lower_third"},
            "persona": profile_key(p), "language": p.get("language", "en"), "involves_focus_player": involved,
            "overlay": out["overlay"], "line": out["line"], "headline": v.insight["headline"], "why": v.insight["why"],
            "verified": all(not c["issues"] for c in checks[-1:]), "checks": v.checks + checks,
            "evidence_ids": m["evidence_ids"], "facts": facts, "importance": m["importance"],
        }

    @handler
    async def fan_out(self, v: VerifiedInsight, ctx: WorkflowContext[None, list]) -> None:
        targets = [p for p in v.job.personas if self.relevant(v.job.moment, p)]
        cues = await asyncio.gather(*(self.personalise(v, p) for p in targets))
        await ctx.yield_output(list(cues))


class InsightPipeline:
    def __init__(self, meta: dict, runner: AgentRunner | None = None):
        self.meta = meta
        self.runner = runner or AgentRunner()
        self.roster = {p["id"]: p for side in ("home", "away") for p in meta["teams"][side]["players"]}
        names = [p["name"] for p in self.roster.values()]
        teams = [meta["teams"][s]["name"] for s in ("home", "away")]
        self.analyst = AnalystExecutor(self.runner)
        self.verifier = VerifierExecutor(names, teams)
        self.persona = PersonaExecutor(self.runner, self.verifier, self.roster)
        self.stats = {"moments": 0, "first_pass_grounded": 0, "fallbacks": 0, "latency_ms": []}

    def _workflow(self):
        # Built per run: workflow instances hold run state and are not re-entrant.
        return (WorkflowBuilder(start_executor=self.analyst, name="match-insights", max_iterations=12)
                .add_edge(self.analyst, self.verifier)
                .add_edge(self.verifier, self.analyst, condition=lambda msg: isinstance(msg, MomentJob))
                .add_edge(self.verifier, self.persona, condition=lambda msg: isinstance(msg, VerifiedInsight))
                .build())

    async def process(self, moment: dict, personas: list[dict]) -> list[dict]:
        t0 = time.perf_counter()
        moment = json.loads(json.dumps(moment))
        result = await self._workflow().run(MomentJob(moment, personas))
        outputs = result.get_outputs()
        cues = outputs[-1] if outputs else []
        checks = moment.get("_checks", [])
        self.stats["moments"] += 1
        if checks and not checks[0]["issues"]:
            self.stats["first_pass_grounded"] += 1
        if any(c.get("attempt") == "fallback" for c in checks):
            self.stats["fallbacks"] += 1
        ms = round(1000 * (time.perf_counter() - t0))
        self.stats["latency_ms"].append(ms)
        for c in cues:
            c["latency_ms"] = ms
        return cues

    async def recap(self, label: str, snapshot: dict, insights: list[dict], players: list[dict],
                    profile: dict) -> dict:
        """Half-time / full-time recap for one audience, grounded in stats and verified moments."""
        teams = self.meta["teams"]
        h, a = snapshot["teams"]["home"], snapshot["teams"]["away"]
        top = sorted(players, key=lambda p: -p["rating"])[:3]
        key = [i for i in insights if i["importance"] >= 0.65][:8]
        facts = {
            "label": label, "score": f"{snapshot['score'][0]}-{snapshot['score'][1]}",
            "home_team": teams["home"]["name"], "away_team": teams["away"]["name"],
            "home": {k: h.get(k) for k in ("possession_pct", "shots", "on_target", "xg", "pass_accuracy_pct",
                                           "pressures", "line_breaking_passes", "distance_km")},
            "away": {k: a.get(k) for k in ("possession_pct", "shots", "on_target", "xg", "pass_accuracy_pct",
                                           "pressures", "line_breaking_passes", "distance_km")},
            "top_players": [{"name": p["name"], "rating": p["rating"], "goals": p["goals"], "assists": p["assists"],
                             "xg": p["xg"], "passes_completed": p["completed"], "distance_km": p["distance_km"]}
                            for p in top],
            "key_moments": [{"clock": i["clock"], "headline": i["overlay"].lstrip("★ "), "why": i["why"],
                             "facts": i["facts"]} for i in key],
        }
        out = None
        checks = []
        if self.runner.online:
            payload = {"match": facts, "profile": {"mode": profile.get("mode", "casual"),
                                                   "language": LANGUAGES.get(profile.get("language", "en"), "English"),
                                                   "favourite_club": teams[profile["club"]]["name"] if profile.get("club") else None}}
            for attempt in range(MAX_ATTEMPTS):
                try:
                    res = await self.runner.ask("recap", payload)
                    text = json.dumps(res, ensure_ascii=False)
                    issues = self.verifier.problems(text, facts)
                except Exception as exc:
                    res, issues = None, [f"model error: {str(exc)[:120]}"]
                checks.append({"stage": "recap", "attempt": attempt + 1, "issues": issues})
                if not issues:
                    out = res
                    break
                payload["previous_draft_problems"] = issues
        if out is None:
            out = self._template_recap(facts, profile)
            checks.append({"stage": "recap", "attempt": "template", "issues": self.verifier.problems(
                json.dumps(out, ensure_ascii=False), facts)})
        return {"type": "recap", "label": label, "persona": profile_key(profile), "recap": out,
                "verified": not checks[-1]["issues"], "checks": checks, "facts": facts}

    @staticmethod
    def _template_recap(f: dict, profile: dict) -> dict:
        h, a = f["home"], f["away"]
        es = profile.get("language") == "es"
        lead = f"{f['home_team']} {f['score']} {f['away_team']}"
        if es:
            summary = (f"{lead}. {f['home_team']} tuvo el {h['possession_pct']}% del balón y generó {h['xg']} xG en "
                       f"{h['shots']} disparos; {f['away_team']} respondió con {a['xg']} xG en {a['shots']} disparos.")
        else:
            summary = (f"{lead}. {f['home_team']} had {h['possession_pct']}% of the ball and created {h['xg']} xG from "
                       f"{h['shots']} shots; {f['away_team']} replied with {a['xg']} xG from {a['shots']} shots.")
        top = f["top_players"][0] if f["top_players"] else None
        label = {"Half time": "Descanso", "Full time": "Final"}.get(f["label"], f["label"]) if es else f["label"]
        return {
            "headline": f"{label}: {lead}",
            "summary": summary,
            "key_moments": [f"{k['clock']} - {k['headline']}" for k in f["key_moments"]],
            "player_of_the_match": f"{top['name']} - {top['rating']}" if top else "",
            "social_post": f"{lead} · xG {h['xg']}-{a['xg']}",
        }
