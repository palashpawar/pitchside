"""Transport for the live feed.

Locally the server replays a match in-process. In production the provider's
feed arrives on Azure Event Hubs: `publish` pushes a match onto a hub in real
time (standing in for the data provider), and `consume` reads it back and drives
a MatchEngine, which is how a deployed engine would ingest.

    python -m pitchside.bus publish --seed 4 --speed 10
    python -m pitchside.bus consume

Requires `pip install -r requirements-azure.txt` and EVENTHUB_CONNECTION_STRING / EVENTHUB_NAME.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np

from .intel.engine import MatchEngine
from .sim.simulator import simulate
from .stream import iter_match

FRAME_EVERY = 2  # publish tracking at 5 Hz to keep message volume modest


def _hub():
    from azure.eventhub import EventData  # noqa: F401
    from azure.eventhub.aio import EventHubConsumerClient, EventHubProducerClient
    conn, name = os.environ["EVENTHUB_CONNECTION_STRING"], os.environ.get("EVENTHUB_NAME", "match-events")
    return conn, name, EventHubProducerClient, EventHubConsumerClient


def encode(kind: str, payload: dict) -> str:
    return json.dumps({"kind": kind, **payload}, ensure_ascii=False)


async def publish(seed: int, speed: float) -> None:
    from azure.eventhub import EventData
    conn, name, Producer, _ = _hub()
    match = simulate(seed)
    producer = Producer.from_connection_string(conn, eventhub_name=name)
    start, t0 = time.perf_counter(), None
    async with producer:
        await producer.send_batch([EventData(encode("meta", {"meta": match.meta}))])
        for kind, item in iter_match(match):
            if kind == "frame" and item % FRAME_EVERY:
                continue
            t = item["t"] if kind == "event" else float(match.t[item])
            t0 = t if t0 is None else t0
            wait = (t - t0) / speed - (time.perf_counter() - start)
            if wait > 0:
                await asyncio.sleep(wait)
            if kind == "event":
                body = encode("event", {"event": item})
            else:
                body = encode("frame", {"t": t, "period": int(match.period[item]),
                                        "xy": np.round(match.xy[item], 2).tolist()})
            await producer.send_batch([EventData(body)])


async def consume() -> None:
    conn, name, _, Consumer = _hub()
    client = Consumer.from_connection_string(conn, consumer_group="$Default", eventhub_name=name)
    state: dict = {}

    async def on_event(partition_context, event):
        msg = json.loads(event.body_as_str())
        if msg["kind"] == "meta":
            state["engine"] = MatchEngine(msg["meta"])
            return
        eng = state.get("engine")
        if eng is None:
            return
        if msg["kind"] == "event":
            _, moments = eng.on_event(msg["event"])
        else:
            _, moments = eng.on_frame(msg["t"], msg["period"], np.array(msg["xy"]))
        for m in moments:
            print(m.clock, m.kind, json.dumps(m.facts, ensure_ascii=False))

    async with client:
        await client.receive(on_event=on_event, starting_position="-1")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--speed", type=float, default=10.0)
    sub.add_parser("consume")
    args = ap.parse_args()
    asyncio.run(publish(args.seed, args.speed) if args.cmd == "publish" else consume())


if __name__ == "__main__":
    main()
