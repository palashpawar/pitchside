# Pitchside

An AI engine that turns a live football data feed into verified, personalised match insights for studios, broadcasters and fans.

> All data is synthetic. Clubs, players and venues are fictional; no real Premier League data is used.

Built for the *Inside the Game* hackathon (Microsoft × Premier League).

## How it works

```
simulator ─► engine ─► moment detector ─► Analyst agent ─► Verifier ─► Persona agent ─► overlays + live UI
 (tracking    (stats,      (goals, press        (explains why)    (every number   (casual/analyst,
  + events)    xPass,       shifts, chaos,           ▲             and name must    club, player,
               indices)     big chances…)            └── feedback ─┘ be in the data) language)
```

1. **Ingest:** a simulator generates realistic matches (22 players + ball at 10 Hz, plus passes, shots, tackles and pressures), streamed live or over Azure Event Hubs.
2. **Interpret:** the engine computes xG, pass difficulty (an xPass model trained on 42k passes), ball and shot speed, distance and sprints.
3. **Explain:** rolling indices (control vs chaos, press intensity, tempo, momentum) flag the moments that matter. A Microsoft Agent Framework workflow writes the explanation, and a verifier rejects any number or name not in the data.
4. **Render:** timed, machine-readable overlay cues (`/api/overlay-track`) and a broadcast-style web player.
5. **Personalise:** casual or analyst mode, favourite club, Player Focus (camera and stats follow one player), 8 languages, a metric of interest, and read-aloud through Azure AI Speech.

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn pitchside.server:app --port 8000   # http://localhost:8000
pytest -q
```

Agents run offline (grounded templates) by default. To use a model, set `PITCHSIDE_LLM` to `azure`, `foundry_local`, `github` or `openai` (see `.env.example`).

## Deploy

```bash
azd up
```

This provisions Azure Container Apps, Azure OpenAI, AI Speech and Event Hubs from `infra/main.bicep`.

## Limitations

- The simulator scores slightly more goals than real football.
- Auto-eventing from tracking alone is experimental: 60% precision and 79% recall on possession changes.
- Offline templates cover English and Spanish only; other languages need a model.

MIT licence.
