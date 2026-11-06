import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "kev-v2-scene"
RES = ROOT / "research" / "offline"
GAME = ROOT / "research" / "gameplay"
OUT = ROOT / "research" / "audits"
OUT.mkdir(parents=True, exist_ok=True)
ACTIONS = ("LEFT", "STAY", "RIGHT", "BRAKE")
K = len(ACTIONS)
rng = np.random.default_rng(20261105)


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


meta = load_jsonl(DATA / "kev_test.provenance.jsonl")
orig = {r["example_id"]: r for r in load_jsonl(RES / "predictions_original_test.jsonl")}
fine = {r["example_id"]: r for r in load_jsonl(RES / "predictions_fine_tuned_test.jsonl")}


def matrix(y, p):
    m = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def metrics(y, prob):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1)
    acc = float((pred == y).mean())
    m = matrix(y, pred)
    per = {}
    f1s = []
    supports = []
    for i, name in enumerate(ACTIONS):
        tp = m[i, i]
        fp = m[:, i].sum() - tp
        fn = m[i, :].sum() - tp
        sup = m[i, :].sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(sup)}
        if sup:
            f1s.append(f1)
            supports.append(sup)
    macro = float(np.mean(f1s)) if f1s else 0.0
    weighted = float(np.average(f1s, weights=supports)) if f1s else 0.0
    eps = 1e-12
    ptrue = np.clip(prob[np.arange(len(y)), y], eps, 1.0)
    ll = float(-np.log(ptrue).mean())
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(((prob - onehot) ** 2).sum(1).mean())
    conf = prob.max(1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return {
        "rows": int(len(y)), "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
        "log_loss": ll, "brier": brier, "ece": float(ece), "per_class": per,
        "confusion": m.tolist(), "predicted": {ACTIONS[i]: int((pred == i).sum()) for i in range(K)},
        "actual": {ACTIONS[i]: int((y == i).sum()) for i in range(K)},
    }


def vectors(preds):
    y, prob = [], []
    for m in meta:
        lab = m["action"]
        p = preds[m["example_id"]]["probabilities"]
        y.append(ACTIONS.index(lab))
        prob.append([float(p.get(a, 0.0)) for a in ACTIONS])
    prob = np.asarray(prob, dtype=np.float64)
    prob /= prob.sum(1, keepdims=True)
    return np.asarray(y), prob


yo, po = vectors(orig)
yf, pf = vectors(fine)
assert np.array_equal(yo, yf)
mo, mf = metrics(yo, po), metrics(yf, pf)

# per-seed offline metrics
seed_rows = defaultdict(list)
for i, m in enumerate(meta):
    seed_rows[m["scenario_seed"]].append(i)
seed_metrics = {}
for seed, idx in seed_rows.items():
    seed_metrics[seed] = {
        "rows": len(idx),
        "original": metrics(yo[idx], po[idx]),
        "fine": metrics(yf[idx], pf[idx]),
    }

# cluster bootstrap over test seeds
seeds = sorted(seed_rows)
boot_acc, boot_macro = [], []
for _ in range(5000):
    pick = rng.choice(seeds, size=len(seeds), replace=True)
    idx = np.concatenate([seed_rows[s] for s in pick])
    a_o = float((po[idx].argmax(1) == yo[idx]).mean())
    a_f = float((pf[idx].argmax(1) == yf[idx]).mean())
    boot_acc.append(a_f - a_o)
    # macro f1 delta via quick computation
    def mac(y, p):
        pred = p.argmax(1)
        mm = matrix(y, pred)
        vals = []
        for i in range(K):
            tp = mm[i, i]; fp = mm[:, i].sum() - tp; fn = mm[i, :].sum() - tp
            sup = mm[i, :].sum()
            if not sup: continue
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            vals.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        return float(np.mean(vals)) if vals else 0.0
    boot_macro.append(mac(yf[idx], pf[idx]) - mac(yo[idx], po[idx]))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))


offline = {
    "original": mo, "fine_tuned": mf,
    "delta": {k: (mf[k] - mo[k]) for k in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier", "ece")},
    "cluster_bootstrap": {
        "accuracy_delta_mean": float(np.mean(boot_acc)), "accuracy_delta_ci95": ci(boot_acc),
        "macro_f1_delta_mean": float(np.mean(boot_macro)), "macro_f1_delta_ci95": ci(boot_macro),
        "seeds": len(seeds),
    },
    "per_seed": {str(s): {"rows": v["rows"], "orig_acc": v["original"]["accuracy"], "fine_acc": v["fine"]["accuracy"],
                          "orig_macro": v["original"]["macro_f1"], "fine_macro": v["fine"]["macro_f1"]} for s, v in seed_metrics.items()},
}
# sign tests on per-seed accuracy/macro
for key in ("accuracy", "macro_f1"):
    wins = sum(v["fine"][key] > v["original"][key] for v in seed_metrics.values())
    losses = sum(v["fine"][key] < v["original"][key] for v in seed_metrics.values())
    ties = len(seed_metrics) - wins - losses
    offline[f"per_seed_{key}_sign"] = {"fine_wins": wins, "original_wins": losses, "ties": ties, "p_two_sided": sign_p(wins, losses)}

# gameplay paired
def ep(name):
    return {r["scenario_seed"]: r for r in load_jsonl(GAME / name)}


go, gf = ep("original-12.episodes.jsonl"), ep("fine-12.episodes.jsonl")
assert set(go) == set(gf)
gseeds = sorted(go)
g = {}
for key in ("survival_time", "distance", "score", "overtaken_count"):
    d = np.asarray([float(gf[s].get(key, 0) - go[s].get(key, 0)) for s in gseeds])
    bs = []
    for _ in range(20000):
        pick = rng.choice(d, size=len(d), replace=True)
        bs.append(float(pick.mean()))
    wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
    g[key] = {
        "original_mean": float(np.mean([go[s].get(key, 0) for s in gseeds])),
        "fine_mean": float(np.mean([gf[s].get(key, 0) for s in gseeds])),
        "original_median": float(np.median([go[s].get(key, 0) for s in gseeds])),
        "fine_median": float(np.median([gf[s].get(key, 0) for s in gseeds])),
        "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
        "delta_ci95": ci(bs), "fine_wins": wins, "original_wins": losses, "ties": ties,
        "sign_p_two_sided": sign_p(wins, losses),
    }
for name, rows in (("original", go), ("fine", gf)):
    ac = Counter()
    for r in rows.values():
        ac.update(r.get("action_counts", {}))
    g[name] = {
        "actions": dict(ac),
        "brake_freq": ac.get("BRAKE", 0) / max(1, sum(ac.values())),
        "lane_changes": ac.get("LEFT", 0) + ac.get("RIGHT", 0),
        "crashes": sum(r.get("termination_reason") == "crash" for r in rows.values()),
        "completions": sum(r.get("termination_reason") == "timeout" for r in rows.values()),
    }

summary = {"offline": offline, "gameplay": g}
(OUT / "RECALCULATED-STATS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "offline": {k: offline[k] for k in ("original", "fine_tuned", "delta", "cluster_bootstrap")},
    "gameplay": g,
}, indent=2, sort_keys=True))
