"""Train the xPass model and build league baselines from simulated matches.

    python scripts/build_models.py --matches 30

Outputs pitchside/intel/xpass_model.json and pitchside/intel/league_baselines.json,
plus a validation summary in data/model_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchside.intel.engine import BASELINE_PATH, MatchEngine  # noqa: E402
from pitchside.intel.features import MODEL_PATH, XPassModel, feature_vector, pass_features  # noqa: E402
from pitchside.intel.tracking import AutoEventer, possession_agreement  # noqa: E402
from pitchside.sim.simulator import simulate  # noqa: E402
from pitchside.stream import run_engine  # noqa: E402
from pitchside.football import side_index  # noqa: E402

METRICS = ["turnovers_pm", "passes_per_poss"]
TEAM_METRICS = ["pressures_pm", "press_height_m", "ppda", "tempo", "directness"]


def collect(seed: int) -> dict:
    m = simulate(seed)
    t_index = {round(float(t), 1): i for i, t in enumerate(m.t)}
    X, y = [], []
    for e in m.events:
        if e["type"] != "pass":
            continue
        i = t_index.get(round(e["t"] - 0.1, 1))
        if i is None:
            continue
        f = pass_features(np.array([e["x"], e["y"]]), np.array([e["end_x"], e["end_y"]]),
                          side_index(e["team"]), m.xy[i])
        X.append(feature_vector(f).tolist())
        y.append(1 if e["outcome"] == "complete" else 0)
    eng = MatchEngine(m.meta)
    samples = {k: [] for k in METRICS + TEAM_METRICS}

    orig = eng._update_indices

    def spy():
        if eng.t >= 300:
            w = eng.window_metrics()
            for k in METRICS:
                samples[k].append(w[k])
            for side in ("home", "away"):
                for k in TEAM_METRICS:
                    samples[k].append(w[side][k])
        return orig()
    eng._update_indices = spy
    run_engine(m, eng)
    ae = AutoEventer(np.array([0] * 11 + [1] * 11))
    for i in range(len(m.t)):
        ae.update(float(m.t[i]), m.xy[i])
    agree = possession_agreement(ae.possession_changes, m.truth["possession"], ae.team_of, 5.0)
    return {"X": X, "y": y, "samples": samples, "agree": agree, "score": m.events[-1]["score"]}


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=30)
    args = ap.parse_args()
    seeds = list(range(101, 101 + args.matches))  # training seeds never overlap the demo seeds
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(collect, seeds))
    X = np.array([r for res in results for r in res["X"]])
    y = np.array([r for res in results for r in res["y"]], dtype=float)
    n_train = int(0.8 * len(X))
    coef = XPassModel.fit(X[:n_train], y[:n_train])
    p_test = 1 / (1 + np.exp(-(X[n_train:] @ coef)))
    y_test = y[n_train:]
    bins = np.linspace(0, 1, 11)
    calib = []
    for a, b in zip(bins, bins[1:]):
        mask = (p_test >= a) & (p_test < b)
        if mask.sum() >= 20:
            calib.append({"bin": f"{a:.1f}-{b:.1f}", "n": int(mask.sum()), "predicted": round(float(p_test[mask].mean()), 3),
                          "observed": round(float(y_test[mask].mean()), 3)})
    report = {
        "matches": len(seeds), "passes": int(len(X)), "completion_rate": round(float(y.mean()), 3),
        "xpass_test_auc": round(auc(p_test, y_test), 3),
        "xpass_test_brier": round(float(np.mean((p_test - y_test) ** 2)), 4),
        "xpass_calibration": calib,
        "auto_eventing_possession_changes": {
            "precision": round(float(np.mean([r["agree"]["precision"] for r in results])), 3),
            "recall": round(float(np.mean([r["agree"]["recall"] for r in results])), 3),
            "tolerance_s": 5.0},
        "goals_per_match": round(float(np.mean([sum(r["score"]) for r in results])), 2),
    }
    MODEL_PATH.write_text(json.dumps({"features": ["bias", "length_over_10m", "lane_opponents", "passer_pressed",
                                                   "receiver_marked", "bypassed_over_3", "into_box", "forward_over_10m"],
                                      "coef": [round(float(c), 4) for c in coef],
                                      "meta": {k: report[k] for k in ("matches", "passes", "xpass_test_auc")}}, indent=2))
    q = {}
    for k in METRICS + TEAM_METRICS:
        vals = np.array([v for r in results for v in r["samples"][k]])
        q[k] = [round(float(v), 4) for v in np.percentile(vals, np.linspace(0, 100, 101))]
    BASELINE_PATH.write_text(json.dumps(q))
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
