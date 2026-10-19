import json

from fastapi.testclient import TestClient

from pitchside import server


def test_health_and_evidence(match, monkeypatch):
    monkeypatch.setitem(server._matches, server.SEED, match)
    c = TestClient(server.app)
    assert c.get("/api/health").json()["ok"]
    shot = next(e for e in match.events if e["type"] == "shot")
    got = c.get(f"/api/events?ids={shot['id']}").json()
    assert got[0]["id"] == shot["id"]


def test_websocket_streams_meta_frames_and_state(match, monkeypatch):
    monkeypatch.setitem(server._matches, server.SEED, match)
    c = TestClient(server.app)
    with c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "speed", "value": 60}))
        seen = set()
        for _ in range(80):
            seen.add(json.loads(ws.receive_text())["type"])
            if {"meta", "frame", "state"} <= seen:
                break
        assert {"meta", "frame", "state"} <= seen
