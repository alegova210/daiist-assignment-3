"""
Assignment 3: evaluation (`main.py evaluate`).

Stages (all run by default; `python evaluate.py retrieval` runs only one):
  retrieval  hit rate at k = 1, 3, 5 for each retrieval design   -> eval/retrieval_*.csv/json
  (agent, checks, judge, agreement: added in the next step)

Hit rule: a question's expected passage is (source, article). A retrieved chunk
matches it when the source is the same and the article is among the articles
the chunk covers. A `multi` question is a hit only if ALL its passages are in
the top k (we also report the partial rate). Unanswerable questions are skipped.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import index

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
KS = (1, 3, 5)


def load_questions() -> list[dict]:
    with (EVAL / "questions.csv").open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["question"].strip()]


def expected_passages(row: dict) -> list[tuple[str, str]]:
    sources = [s.strip() for s in row["expected_source"].split(";") if s.strip()]
    locs = [l.strip() for l in row["expected_location"].split(";") if l.strip()]
    if len(sources) == 1 and len(locs) > 1:
        sources = sources * len(locs)
    return list(zip(sources, locs))


def matches(chunk: dict, passage: tuple[str, str]) -> bool:
    source, location = passage
    return chunk["source"] == source and location in chunk["covers"].split("|")


def stage_retrieval() -> dict:
    questions = [q for q in load_questions() if q["kind"] != "unanswerable"]
    summary, rows = {}, []
    for design in index.SEARCH_VARIANTS:
        hits = {k: 0 for k in KS}
        partial = {k: 0 for k in KS}
        for q in questions:
            results = index.search(q["question"], k=max(KS), design=design)
            passages = expected_passages(q)
            first_rank = None
            for rank, chunk in enumerate(results, 1):
                if any(matches(chunk, p) for p in passages):
                    first_rank = first_rank or rank
            row = {"design": design, "id": q["id"], "kind": q["kind"], "first_rank": first_rank or ""}
            for k in KS:
                top = results[:k]
                found = [any(matches(c, p) for c in top) for p in passages]
                row[f"hit@{k}"] = int(all(found))
                row[f"partial@{k}"] = round(sum(found) / len(found), 2)
                hits[k] += int(all(found))
                partial[k] += sum(found) / len(found)
            row["top5"] = " ; ".join(f"{c['source']}:{c['location']}" for c in results)
            rows.append(row)
        n = len(questions)
        summary[design] = {
            "questions": n,
            **{f"hit@{k}": round(hits[k] / n, 3) for k in KS},
            **{f"partial@{k}": round(partial[k] / n, 3) for k in KS},
            **{f"hit@{k}_{kind}": round(
                sum(r[f"hit@{k}"] for r in rows if r["design"] == design and r["kind"] == kind)
                / max(1, sum(1 for q in questions if q["kind"] == kind)), 3)
               for k in KS for kind in ("single", "multi")},
        }
    with (EVAL / "retrieval_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (EVAL / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'design':17s} " + " ".join(f"hit@{k:<3d}" for k in KS) + "  (single / multi @5)")
    for d, s in summary.items():
        print(f"{d:17s} " + " ".join(f"{s[f'hit@{k}']:<7.2f}" for k in KS)
              + f"  {s['hit@5_single']:.2f} / {s['hit@5_multi']:.2f}")
    return summary


STAGES = {"retrieval": stage_retrieval}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    for name in wanted:
        print(f"\n=== {name} ===")
        STAGES[name]()
