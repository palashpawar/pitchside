"""Generate synthetic matches: python -m pitchside.sim --seed 4 --out data/"""
import argparse
from pathlib import Path

from .simulator import simulate

ap = argparse.ArgumentParser(description="Generate a synthetic, football-realistic match (tracking + events).")
ap.add_argument("--seed", type=int, nargs="+", default=[4])
ap.add_argument("--out", default="data")
args = ap.parse_args()
for seed in args.seed:
    m = simulate(seed)
    stem = Path(args.out) / f"match_{seed:04d}"
    m.save(stem)
    kinds = {}
    for e in m.events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print(f"{stem}: {m.meta['teams']['home']['name']} {m.events[-1]['score'][0]}-{m.events[-1]['score'][1]} "
          f"{m.meta['teams']['away']['name']} · {len(m.t)} frames · {len(m.events)} events · "
          f"{kinds.get('pass', 0)} passes, {kinds.get('shot', 0)} shots")
