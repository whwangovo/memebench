"""Aggregate two MemeBench judge result files with item-level intersection.

The paper protocol marks a checklist item as passed only when both judges pass
that item.  Dimension scores are then recomputed from those aggregated
item-level decisions.

Usage:
    python pipelines/evaluation/aggregate_dual_judge.py \
      --judge-a data/output/gpt5.1/evals/vanilla.gemini_judge.json \
      --judge-b data/output/gpt5.1/evals/vanilla.gpt_judge.json \
      --bench data/annotations/annotations_v9.json \
      --output data/output/gpt5.1/evals/vanilla.dual.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIM_ORDER = ("visual", "identity", "knowledge", "reasoning")


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def item_key(item: dict) -> tuple[str, object]:
    item_id = item.get("id")
    if item_id is not None:
        return ("id", item_id)
    return ("image_path", item.get("image_path"))


def build_map(items: list[dict]) -> dict[tuple[str, object], dict]:
    return {item_key(item): item for item in items}


def failed_evaluation(reason: str = "missing judge result") -> dict:
    return {
        "checklist_score": 0,
        "checklist_result": [0, 0, 0, 0],
        "complete_vikr": False,
        "overall_correctness": "NO",
        "dimension_result": {dim: 0 for dim in DIM_ORDER},
        "item_results": {
            dim: [{"index": 0, "pass": 0, "reason": reason}]
            for dim in DIM_ORDER
        },
        "parse_success": False,
        "parse_error": reason,
        "format": "dual_aggregated",
    }


def get_eval(item: dict | None) -> dict | None:
    if not item:
        return None
    ev = item.get("evaluation")
    return ev if isinstance(ev, dict) else None


def binary(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    return 0


def aggregate_itemized(a: dict, b: dict) -> dict | None:
    a_items = a.get("item_results")
    b_items = b.get("item_results")
    if not isinstance(a_items, dict) or not isinstance(b_items, dict):
        return None

    item_results = {}
    dimension_result = {}
    for dim in DIM_ORDER:
        dim_a = a_items.get(dim)
        dim_b = b_items.get(dim)
        if not isinstance(dim_a, list) or not isinstance(dim_b, list) or len(dim_a) != len(dim_b):
            return None
        merged = []
        for idx, (ia, ib) in enumerate(zip(dim_a, dim_b)):
            pa = binary(ia.get("pass") if isinstance(ia, dict) else 0)
            pb = binary(ib.get("pass") if isinstance(ib, dict) else 0)
            passed = int(pa and pb)
            merged.append({
                "index": ia.get("index", idx) if isinstance(ia, dict) else idx,
                "pass": passed,
                "judge_a_pass": pa,
                "judge_b_pass": pb,
                "judge_a_reason": ia.get("reason", "") if isinstance(ia, dict) else "",
                "judge_b_reason": ib.get("reason", "") if isinstance(ib, dict) else "",
            })
        item_results[dim] = merged
        dimension_result[dim] = int(all(item["pass"] for item in merged))

    vector = [dimension_result[dim] for dim in DIM_ORDER]
    return {
        "checklist_score": sum(vector),
        "checklist_result": vector,
        "complete_vikr": bool(all(vector)),
        "overall_correctness": "YES" if vector[1] and vector[2] and vector[3] else "NO",
        "dimension_result": dimension_result,
        "item_results": item_results,
        "parse_success": True,
        "parse_error": None,
        "format": "dual_aggregated",
    }


def aggregate_dimension_only(a: dict, b: dict) -> dict:
    va = a.get("checklist_result")
    vb = b.get("checklist_result")
    if not (isinstance(va, list) and isinstance(vb, list) and len(va) == 4 and len(vb) == 4):
        return failed_evaluation("missing 4-element checklist_result")
    vector = [int(binary(va[i]) and binary(vb[i])) for i in range(4)]
    return {
        "checklist_score": sum(vector),
        "checklist_result": vector,
        "complete_vikr": bool(all(vector)),
        "overall_correctness": "YES" if vector[1] and vector[2] and vector[3] else "NO",
        "dimension_result": {dim: vector[i] for i, dim in enumerate(DIM_ORDER)},
        "item_results": None,
        "parse_success": True,
        "parse_error": None,
        "format": "dual_aggregated_dimension_fallback",
    }


def aggregate_evaluations(a: dict | None, b: dict | None) -> dict:
    if not a or not b:
        return failed_evaluation("missing one or both judge evaluations")
    if not a.get("parse_success") or not b.get("parse_success"):
        return failed_evaluation("one or both judge evaluations failed to parse")

    itemized = aggregate_itemized(a, b)
    if itemized is not None:
        return itemized
    return aggregate_dimension_only(a, b)


def aggregate_files(judge_a: list[dict], judge_b: list[dict], bench: list[dict] | None = None) -> list[dict]:
    map_a = build_map(judge_a)
    map_b = build_map(judge_b)

    if bench is not None:
        skeleton = [{"id": item.get("id"), "image_path": item.get("image_path")} for item in bench]
    else:
        keys = sorted(set(map_a) | set(map_b), key=lambda x: (x[0], str(x[1])))
        skeleton = []
        for key in keys:
            source = map_a.get(key) or map_b.get(key) or {}
            skeleton.append({"id": source.get("id"), "image_path": source.get("image_path")})

    results = []
    for base in skeleton:
        key = item_key(base)
        item_a = map_a.get(key)
        item_b = map_b.get(key)
        evaluation = aggregate_evaluations(get_eval(item_a), get_eval(item_b))
        results.append({
            "id": base.get("id"),
            "image_path": base.get("image_path"),
            "response": (item_a or item_b or {}).get("response"),
            "evaluation": evaluation,
        })
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    dim_sums = [0, 0, 0, 0]
    complete = 0
    parsed = 0
    for item in results:
        ev = item.get("evaluation") or {}
        parsed += int(bool(ev.get("parse_success")))
        vec = ev.get("checklist_result") or [0, 0, 0, 0]
        if len(vec) == 4:
            for i in range(4):
                dim_sums[i] += binary(vec[i])
            complete += int(all(binary(v) for v in vec))
    return {
        "n": n,
        "parse_success": parsed,
        "complete_vikr_rate": round(complete / n * 100, 2) if n else 0,
        "visual_acc": round(dim_sums[0] / n * 100, 2) if n else 0,
        "identity_acc": round(dim_sums[1] / n * 100, 2) if n else 0,
        "knowledge_acc": round(dim_sums[2] / n * 100, 2) if n else 0,
        "reasoning_acc": round(dim_sums[3] / n * 100, 2) if n else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate two MemeBench judge files with intersection scoring.")
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bench", default=None, help="Optional annotation file defining the denominator/order.")
    args = parser.parse_args()

    judge_a = load_json(args.judge_a)
    judge_b = load_json(args.judge_b)
    bench = load_json(args.bench) if args.bench else None

    results = aggregate_files(judge_a, judge_b, bench)
    dump_json(args.output, results)
    print(json.dumps(summarize(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
