"""Live match server.

Each WebSocket viewer gets a session that replays the synthetic feed on a
match clock (speed adjustable), runs the engine live, and pushes:

  frame     positions of 22 players + ball (~20 Hz)
  state     scoreboard, team stats, explainability indices, focus-player card (1 Hz)
  cue       fast-lane overlay cues (speed alerts, pass and shot badges, cards)
  insight   agent-lane cues: verified, personalised explanations
  event     ticker items (shots, goals, cards, set pieces)
  recap     half-time / full-time recaps

Run:  uvicorn pitchside.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .agents.llm import provider
from .agents.pipeline import AgentRunner, InsightPipeline, profile_key
from .intel.engine import MatchEngine
from .sim.simulator import MatchData, simulate
from .stream import iter_match

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DATA = ROOT / "data"
SEED = int(os.environ.get("PITCHSIDE_SEED", "4"))
TICK = 0.05
TICKER = {"shot", "goal", "card", "corner", "save", "period_start", "period_end", "full_time", "foul"}

app = FastAPI(title="Pitchside: synthetic match insights engine")
_matches: dict[int, MatchData] = {}
_runner: AgentRunner | None = None


def get_match(seed: int = SEED) -> MatchData:
    if seed not in _matches:
        stem = DATA / f"match_{seed:04d}"
        if stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists():
            _matches[seed] = MatchData.load(stem)
        else:
            _matches[seed] = simulate(seed)
            _matches[seed].save(stem)
    return _matches[seed]


def get_runner() -> AgentRunner:
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner


DEFAULT_PROFILE = {"mode": "casual", "club": None, "player_id": None, "language": "en", "metric": None}


class Session:
    def __init__(self, ws: WebSocket, match: MatchData):
        self.ws = ws
        self.match = match
        self.profile = dict(DEFAULT_PROFILE)
        self.speed = 8.0
        self.paused = False
        self.lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()
        self.pipeline = InsightPipeline(match.meta, get_runner())
        self._reset()

    def _reset(self, start_index: int = 0) -> None:
        self.engine = MatchEngine(self.match.meta)
        self.stream = iter_match(self.match, 0)
        self.pending = None
        self.clock_t = float(self.match.t[start_index]) if start_index else -0.5
        self.frame_i = 0
        self.moments: list[dict] = []
        self.insights: list[dict] = []
        self.done = False

    async def send(self, kind: str, data) -> None:
        try:
            await self.ws.send_text(json.dumps({"type": kind, "data": data}, ensure_ascii=False))
        except Exception:
            pass

    # ---------- stream processing ----------
    def _advance(self, until: float, live: bool = True) -> tuple[list, list, list]:
        cues, moments, ticker = [], [], []
        while True:
            if self.pending is None:
                self.pending = next(self.stream, None)
                if self.pending is None:
                    self.done = True
                    break
            kind, item = self.pending
            t = item["t"] if kind == "event" else float(self.match.t[item])
            if t > until:
                break
            self.pending = None
            if kind == "event":
                c, m = self.engine.on_event(item)
                if item["type"] in TICKER:
                    ticker.append(self._ticker(item))
                if item["type"] in ("period_end", "full_time"):
                    ticker[-1]["recap"] = "Half time" if item["type"] == "period_end" and item["period"] == 1 else \
                        "Full time" if item["type"] == "full_time" else None
            else:
                self.frame_i = item
                c, m = self.engine.on_frame(t, int(self.match.period[item]), self.match.xy[item])
            cues += c
            moments += m
        return cues, moments, ticker

    def _ticker(self, ev: dict) -> dict:
        out = {k: ev.get(k) for k in ("id", "t", "clock", "type", "team", "outcome", "xg", "card", "score")}
        out["player"] = self.engine.name(ev.get("player_id"))
        return out

    def _frame_payload(self) -> dict:
        i = self.frame_i
        xy = self.match.xy[i]
        return {"t": round(float(self.match.t[i]), 1), "p": [[round(float(a), 1), round(float(b), 1)] for a, b in xy],
                "bs": round(self.engine.tracking.ball_speed * 3.6)}

    def _state_payload(self) -> dict:
        snap = self.engine.snapshot()
        pid = self.profile.get("player_id")
        if pid:
            snap["focus"] = self.engine.player_snapshot(pid)
        snap["history"] = [{"t": h["t"], "momentum": h["momentum"], "chaos": h["chaos"],
                            "press_h": h["home"]["press"], "press_a": h["away"]["press"]}
                           for h in self.engine.index_history[-240:]]
        snap["agents"] = {"provider": provider(), **{k: v for k, v in self.pipeline.stats.items() if k != "latency_ms"},
                          "median_latency_ms": sorted(self.pipeline.stats["latency_ms"])[len(self.pipeline.stats["latency_ms"]) // 2]
                          if self.pipeline.stats["latency_ms"] else None,
                          "model_calls": self.pipeline.runner.calls, "cache_hits": self.pipeline.runner.cache_hits}
        return snap

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _explain(self, moment: dict, profile: dict, replay: bool = False) -> None:
        try:
            cues = await self.pipeline.process(moment, [profile])
        except Exception as exc:  # never let one moment kill the stream
            await self.send("error", {"where": "pipeline", "detail": str(exc)[:300]})
            return
        if profile_key(profile) != profile_key(self.profile):
            return
        for c in cues:
            c["replayed"] = replay
            self.insights.append(c)
            await self.send("insight", c)

    async def _recap(self, label: str) -> None:
        players = [self.engine.player_snapshot(p) for p in self.engine.roster]
        mine = [i for i in self.insights if i["persona"] == profile_key(self.profile)]
        r = await self.pipeline.recap(label, self.engine.snapshot(), mine, players, self.profile)
        await self.send("recap", r)

    async def run(self) -> None:
        await self.send("meta", {**self.match.meta, "duration": float(self.match.t[-1]), "provider": provider(),
                                 "profile": self.profile, "speed": self.speed})
        last_state = 0.0
        last = time.perf_counter()
        while True:
            await asyncio.sleep(TICK)
            now = time.perf_counter()
            dt, last = now - last, now
            if self.paused or self.done:
                continue
            async with self.lock:
                self.clock_t += dt * self.speed
                cues, moments, ticker = self._advance(self.clock_t)
            await self.send("frame", self._frame_payload())
            for c in cues:
                await self.send("cue", c)
            for tk in ticker:
                await self.send("event", tk)
                if tk.get("recap"):
                    self._spawn(self._recap(tk["recap"]))
            for m in moments:
                md = m.to_dict()
                self.moments.append(md)
                self._spawn(self._explain(md, dict(self.profile)))
            if now - last_state > 1.0 or moments:
                last_state = now
                await self.send("state", self._state_payload())
            if self.done:
                await self.send("done", {})

    async def control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "speed":
            self.speed = float(max(0.5, min(60.0, msg.get("value", 8))))
        elif kind == "pause":
            self.paused = bool(msg.get("value", not self.paused))
        elif kind == "profile":
            new = {**DEFAULT_PROFILE, **{k: v for k, v in msg.get("value", {}).items() if k in DEFAULT_PROFILE}}
            changed = profile_key(new) != profile_key(self.profile)
            self.profile = new
            if changed:
                await self.send("profile", self.profile)
                for md in self.moments[-6:]:  # re-render the recent story for the new audience
                    self._spawn(self._explain(md, dict(self.profile), replay=True))
            await self.send("state", self._state_payload())
        elif kind == "seek":
            target = float(msg.get("t", 0))
            async with self.lock:
                if target < self.clock_t:
                    self._reset()
                _, moments, _ = self._advance(target)
                self.clock_t = target
                self.moments += [m.to_dict() for m in moments]
            await self.send("seeked", {"t": target})
            for md in self.moments[-4:]:
                self._spawn(self._explain(md, dict(self.profile), replay=True))
            await self.send("frame", self._frame_payload())
            await self.send("state", self._state_payload())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, seed: int = SEED):
    await ws.accept()
    session = Session(ws, get_match(seed))
    runner = asyncio.create_task(session.run())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await session.control(msg)
    except WebSocketDisconnect:
        pass
    finally:
        runner.cancel()
        for t in list(session.tasks):
            t.cancel()


@app.get("/api/health")
def health():
    return {"ok": True, "provider": provider(), "seed": SEED}


@app.get("/api/meta")
def meta(seed: int = SEED):
    return get_match(seed).meta


@app.get("/api/model-report")
def model_report():
    p = DATA / "model_report.json"
    if not p.exists():
        raise HTTPException(404, "run scripts/build_models.py first")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/events")
def events(seed: int = SEED, ids: str = Query("", description="comma-separated event ids")):
    """Evidence lookup: the raw events behind an insight."""
    wanted = {int(i) for i in ids.split(",") if i.strip().isdigit()}
    m = get_match(seed)
    return [e for e in m.events if e["id"] in wanted]


@app.get("/api/overlay-track")
async def overlay_track(seed: int = SEED, mode: str = "casual", language: str = "en", club: str | None = None,
                        player_id: str | None = None, metric: str | None = None):
    """The whole match as a timed, machine-readable overlay track for one audience profile.

    This is the hand-off format for a broadcast graphics renderer: every cue has
    t_in/t_out on the match clock, an anchor (player or screen region), a priority,
    display text, and the evidence event ids behind it.
    """
    m = get_match(seed)
    from .stream import run_engine
    eng = MatchEngine(m.meta)
    fast, moments = run_engine(m, eng)
    profile = {"mode": mode, "language": language, "club": club, "player_id": player_id, "metric": metric}
    pipe = InsightPipeline(m.meta, get_runner())
    agent_cues = []
    for mo in moments:
        agent_cues += await pipe.process(mo.to_dict(), [profile])
    keep = {"speed_alert", "shot_badge", "pass_badge", "card"}
    track = sorted([c for c in fast if c["type"] in keep] + agent_cues, key=lambda c: c["t_in"])
    return {"schema": "pitchside.overlay/v1", "match_id": m.meta["match_id"], "profile": profile,
            "clock": "seconds since kick-off (continuous across halves)", "count": len(track), "cues": track,
            "agent_stats": {k: v for k, v in pipe.stats.items() if k != "latency_ms"}}


@app.get("/api/tts")
async def tts(text: str, lang: str = "en"):
    """Azure AI Speech neural TTS when AZURE_SPEECH_KEY is set; the UI falls back to the browser voice otherwise."""
    key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION", "eastus")
    if not key:
        raise HTTPException(404, "Azure Speech not configured")
    voices = {"en": ("en-GB", "en-GB-RyanNeural"), "es": ("es-ES", "es-ES-AlvaroNeural"),
              "fr": ("fr-FR", "fr-FR-HenriNeural"), "pt": ("pt-BR", "pt-BR-AntonioNeural"),
              "de": ("de-DE", "de-DE-ConradNeural"), "ar": ("ar-SA", "ar-SA-HamedNeural"),
              "hi": ("hi-IN", "hi-IN-MadhurNeural"), "ja": ("ja-JP", "ja-JP-KeitaNeural")}
    loc, voice = voices.get(lang, voices["en"])
    safe = text.replace("&", "and").replace("<", "").replace(">", "")[:400]
    ssml = (f"<speak version='1.0' xml:lang='{loc}'><voice name='{voice}'>"
            f"<prosody rate='+8%'>{safe}</prosody></voice></speak>")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", content=ssml.encode(),
                              headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                                       "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3"})
    if r.status_code != 200:
        raise HTTPException(502, f"speech service returned {r.status_code}")
    return Response(r.content, media_type="audio/mpeg")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
