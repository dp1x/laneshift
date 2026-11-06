import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Dhane\dev\cris@ds\LANESHIFT")
D = ROOT / "kev-deliverables-20261105"
DATA = ROOT / "cris@ds\.portable-dev\kev-v2-scene"

records = [json.loads(l) for l in (DATA / "kev_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
meta = [json.loads(l) for l in (DATA / "kev_test.provenance.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
orig = {r["example_id"]: r for r in (json.loads(l) for l in (D / "results/predictions_original_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
fine = {r["example_id"]: r for r in (json.loads(l) for l in (D / "results/predictions_fine_tuned_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
assert len(records) == len(meta) == 4317

for model_name, preds in (("original", orig), ("fine_tuned", fine)):
    print(f"=== {model_name} ===")
    for action in ("LEFT", "RIGHT"):
        rows = [(r, m) for r, m in zip(records, meta) if m["action"] == action]
        offered_ok = [x for x in rows if action in x[0]["questions"]["next_action"]["criteria"]]
        top = Counter()
        ranks = Counter()
        p_true = []
        p_stay = []
        margins = []
        top2_turn = 0
        for r, m in offered_ok:
            q = r["questions"]["next_action"]
            probs = preds[m["example_id"]]["probabilities"]
            offered = list(q["criteria"])
            ordered = sorted(offered, key=lambda a: probs.get(a, 0), reverse=True)
            top[ordered[0]] += 1
            rank = ordered.index(action) + 1
            ranks[rank] += 1
            p_true.append(probs.get(action, 0.0))
            p_stay.append(probs.get("STAY", 0.0))
            margins.append(probs.get("STAY", 0.0) - probs.get(action, 0.0))
            if action in ordered[:2]:
                top2_turn += 1
        print(f" {action}: rows={len(rows)} offered={len(offered_ok)} top={dict(top)} ranks={dict(sorted(ranks.items()))} top2_turn={top2_turn}")
        if p_true:
            print(f"   p_true mean={sum(p_true)/len(p_true):.4f} max={max(p_true):.4f}; p_stay mean={sum(p_stay)/len(p_stay):.4f}; mean stay-minus-true={sum(margins)/len(margins):.4f}")
            print(f"   p_true>0.10={sum(x>0.10 for x in p_true)} p_true>0.25={sum(x>0.25 for x in p_true)} p_true>0.50={sum(x>0.50 for x in p_true)}")

# Representative test turn rows: highest original P(true) and highest fine P(true).
def examples(action, key, count=3):
    rows = [(r, m) for r, m in zip(records, meta) if m["action"] == action and action in r["questions"]["next_action"]["criteria"]]
    rows.sort(key=lambda x: key(x[0], x[1]), reverse=True)
    for r, m in rows[:count]:
        q = r["questions"]["next_action"]
        eid = m["example_id"]
        print(f"--- {action} {eid} seed={m['scenario_seed']} cell={m.get('cell')} density={m.get('traffic_density')} tokens={m.get('total_tokens')} ---")
        for name, preds in (("original", orig), ("fine", fine)):
            p = preds[eid]["probabilities"]
            print(f" {name}: pred={preds[eid]['predicted']} probs={ {k: round(v,4) for k,v in p.items()} }")
        print(" state:", r["state"].replace("\n", " | ")[:700])

print("\n=== representative turns where original was most confident ===")
examples("LEFT", lambda r, m: orig[m["example_id"]]["probabilities"].get("LEFT", 0.0), 3)
examples("RIGHT", lambda r, m: orig[m["example_id"]]["probabilities"].get("RIGHT", 0.0), 3)
print("\n=== representative turns where fine-tuned was closest ===")
examples("LEFT", lambda r, m: fine[m["example_id"]]["probabilities"].get("LEFT", 0.0), 3)
examples("RIGHT", lambda r, m: fine[m["example_id"]]["probabilities"].get("RIGHT", 0.0), 3)
