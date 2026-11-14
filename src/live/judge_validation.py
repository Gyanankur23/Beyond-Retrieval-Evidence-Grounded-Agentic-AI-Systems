"""Validate the LLM judge against the two authors' labels.

  python -m src.live.judge_validation export   # writes results/judge_validation/sample.csv (stratified, blind)
  ... authors fill human_a / human_b (1 = correct, 0 = incorrect) independently ...
  python -m src.live.judge_validation score    # Cohen's kappa: A vs B, and judge vs consensus
"""
import csv, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "judge_validation"; D.mkdir(parents=True, exist_ok=True)


def kappa(x, y):
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    p1x, p1y = sum(x) / n, sum(y) / n
    pe = p1x * p1y + (1 - p1x) * (1 - p1y)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def export(n=60):
    qs = {q["question_id"]: q for q in json.load(open(ROOT / "data" / "questions.json", encoding="utf-8"))}
    log = {(r["question_id"], r["system_config"]): r for r in csv.DictReader(open(ROOT / "results/raw_results/evaluation_log.csv"))}
    tr = [json.loads(l) for l in open(ROOT / "results/raw_results/traces.jsonl", encoding="utf-8")]
    pool = [t for t in tr if not t["abstained"] and qs[t["question_id"]]["category"] != "insufficient_evidence"]
    random.Random(7).shuffle(pool)
    with open(D / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item", "question_id", "system_config", "question", "reference", "answer", "human_a", "human_b", "judge"])
        for i, t in enumerate(pool[:n]):
            q = qs[t["question_id"]]
            w.writerow([i, t["question_id"], t["system_config"], q["question"], q["ground_truth"], t["answer"], "", "",
                        log[(t["question_id"], t["system_config"])]["answer_accuracy"]])
    print(f"wrote {D/'sample.csv'} ({min(n, len(pool))} items). Label human_a / human_b independently.")


def score():
    rows = list(csv.DictReader(open(D / "sample.csv", encoding="utf-8")))
    a = [int(r["human_a"]) for r in rows]; b = [int(r["human_b"]) for r in rows]; j = [int(r["judge"]) for r in rows]
    cons = [x for x, y in zip(a, b) if x == y]; jj = [z for x, y, z in zip(a, b, j) if x == y]
    print(f"n={len(rows)}  kappa(A,B)={kappa(a, b):.3f}  agreed items={len(cons)}  kappa(judge,consensus)={kappa(jj, cons):.3f}")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
