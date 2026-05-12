import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from tqdm.asyncio import tqdm

# Ensure project root is on sys.path
sys.path.append(os.getcwd())
from memebench.llm_client import build_async_client

BENCH_JSON = "data/annotations/annotations_v8.json"
BENCH_JSON_V3 = "data/annotations/annotations_v8.json"
DEFAULT_CANDIDATE_JSON = "data/output/meme_responses_gpt5.1.json"
DEFAULT_OUTPUT_JSON = "data/output/meme_eval_results_gpt5.1.json"
MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "gpt-5"))

SCORE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?checklist[\s_]+score(?:\*\*)?\s*:?\s*(?:CHECKLIST_SCORE\s*:)?\s*([0-4])\s*/\s*[34]\b",
    re.IGNORECASE,
)
# 3-element vector (v2 schema)
VECTOR_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?checklist[\s_]+result(?:[\s_]+vector)?(?:\*\*)?\s*:?\s*(?:CHECKLIST_RESULT\s*:)?\s*"
    r"\[\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\]",
    re.IGNORECASE,
)
# 4-element vector (v3 VIKR schema)
VECTOR_PATTERN_V3 = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?checklist[\s_]+result(?:[\s_]+vector)?(?:\*\*)?\s*:?\s*(?:CHECKLIST_RESULT\s*:)?\s*"
    r"\[\s*([01])\s*,\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\]",
    re.IGNORECASE,
)
CORRECTNESS_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?overall[\s_]+correctness(?:\*\*)?\s*:?\s*(?:OVERALL_CORRECTNESS\s*:)?\s*(YES|NO)\b",
    re.IGNORECASE,
)

DIM_ORDER = ("visual", "identity", "knowledge", "reasoning")

file_lock = asyncio.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model responses against ground truth.")
    parser.add_argument("--bench", type=str, default=None,
                        help="Path to benchmark annotation JSON (defaults to v2 or v3 based on --v3 flag)")
    parser.add_argument(
        "--candidate",
        type=str,
        default=DEFAULT_CANDIDATE_JSON,
        help="Path to candidate response JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to output evaluation JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to judge")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run LLM judge for items that already have complete evaluations",
    )
    parser.add_argument(
        "--reparse-only",
        action="store_true",
        help="Re-parse existing raw judge outputs in the output file without calling the LLM",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="Use the 4-dimension VIKR schema.",
    )
    parser.add_argument(
        "--itemized-json",
        action="store_true",
        help="Expect item-level JSON judge output.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.getenv("MEMEBENCH_JUDGE_PROMPT"),
        help="Path to a judge prompt template. The template must contain {reference_answer} and {generated_answer_to_eval}.",
    )
    return parser


def json_serializer(obj):
    return str(obj)


def load_prompt_template(path: str | None) -> str:
    if not path:
        raise ValueError(
            "Judge prompt is not bundled in this release. Pass --prompt or set MEMEBENCH_JUDGE_PROMPT."
        )
    return Path(path).read_text(encoding="utf-8")


def is_complete_evaluation(eval_data, schema_version: int = 2) -> bool:
    if not eval_data or not isinstance(eval_data, dict):
        return False

    score = eval_data.get("checklist_score")
    vector = eval_data.get("checklist_result")
    correctness = eval_data.get("overall_correctness")

    max_score = 4 if schema_version == 3 else 3
    expected_len = 4 if schema_version == 3 else 3

    return (
        isinstance(score, int)
        and 0 <= score <= max_score
        and isinstance(vector, list)
        and len(vector) in (expected_len, 3)  # accept 3-element fallback in v3
        and all(isinstance(v, int) and v in (0, 1) for v in vector)
        and correctness in {"YES", "NO"}
    )


def _coerce_binary(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def _extract_json_object(text: str) -> dict | None:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        candidates.extend(
            block.strip()
            for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        )
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_itemized_json(raw_content: str) -> dict | None:
    payload = _extract_json_object(raw_content)
    if payload is None:
        return None

    parsed_result = {
        "raw_content": raw_content,
        "checklist_score": None,
        "checklist_result": None,
        "overall_correctness": None,
        "complete_vikr": None,
        "item_results": None,
        "dimension_result": None,
        "parse_success": False,
        "parse_error": None,
        "format": "itemized_json",
    }

    vector = payload.get("checklist_result")
    if not (isinstance(vector, list) and len(vector) == 4):
        parsed_result["parse_error"] = "itemized JSON missing 4-element checklist_result"
        return parsed_result
    vector = [_coerce_binary(v) for v in vector]
    if any(v is None for v in vector):
        parsed_result["parse_error"] = "itemized JSON checklist_result must be binary"
        return parsed_result
    vector = [int(v) for v in vector]

    score = payload.get("checklist_score")
    if not isinstance(score, int):
        score = sum(vector)
    if score != sum(vector):
        parsed_result["parse_error"] = "itemized JSON checklist_score does not match checklist_result"
        return parsed_result

    dimension_result = payload.get("dimension_result")
    if not isinstance(dimension_result, dict):
        dimension_result = {dim: vector[i] for i, dim in enumerate(DIM_ORDER)}
    else:
        normalized_dims = {}
        for i, dim in enumerate(DIM_ORDER):
            value = _coerce_binary(dimension_result.get(dim))
            if value is None:
                value = vector[i]
            normalized_dims[dim] = value
        dimension_result = normalized_dims

    if [dimension_result[dim] for dim in DIM_ORDER] != vector:
        parsed_result["parse_error"] = "itemized JSON dimension_result does not match checklist_result"
        return parsed_result

    item_results = payload.get("item_results")
    if not isinstance(item_results, dict):
        parsed_result["parse_error"] = "itemized JSON missing item_results"
        return parsed_result
    normalized_items = {}
    for dim in DIM_ORDER:
        raw_items = item_results.get(dim)
        if not isinstance(raw_items, list) or not raw_items:
            parsed_result["parse_error"] = f"itemized JSON missing item_results.{dim}"
            return parsed_result
        normalized_dim_items = []
        for idx, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                parsed_result["parse_error"] = f"itemized JSON item_results.{dim}[{idx}] is not an object"
                return parsed_result
            passed = _coerce_binary(raw_item.get("pass"))
            if passed is None:
                parsed_result["parse_error"] = f"itemized JSON item_results.{dim}[{idx}].pass must be binary"
                return parsed_result
            normalized_dim_items.append({
                "index": raw_item.get("index", idx),
                "pass": passed,
                "reason": str(raw_item.get("reason", "")).strip(),
            })
        normalized_items[dim] = normalized_dim_items

    complete_vikr = bool(vector[0] and vector[1] and vector[2] and vector[3])
    legacy_correct = "YES" if (vector[1] and vector[2] and vector[3]) else "NO"
    correctness = str(payload.get("overall_correctness", legacy_correct)).upper()
    if correctness not in {"YES", "NO"}:
        parsed_result["parse_error"] = "itemized JSON overall_correctness must be YES or NO"
        return parsed_result
    if correctness != legacy_correct:
        parsed_result["parse_error"] = "itemized JSON overall_correctness does not match I/K/R legacy rule"
        return parsed_result

    parsed_result.update({
        "checklist_score": score,
        "checklist_result": vector,
        "overall_correctness": correctness,
        "complete_vikr": complete_vikr,
        "item_results": normalized_items,
        "dimension_result": dimension_result,
        "parse_success": True,
    })
    return parsed_result


def parse_judge_output(raw_content: str | None, schema_version: int = 2) -> dict:
    parsed_result = {
        "raw_content": raw_content,
        "checklist_score": None,
        "checklist_result": None,
        "overall_correctness": None,
        "complete_vikr": None,
        "item_results": None,
        "dimension_result": None,
        "parse_success": False,
        "parse_error": None,
    }

    if not isinstance(raw_content, str) or not raw_content.strip():
        parsed_result["parse_error"] = "empty raw_content"
        return parsed_result

    text = raw_content.strip()
    itemized = parse_itemized_json(text)
    if itemized is not None:
        return itemized

    normalized_text = (text
        .replace("**", "")
        .replace("`", "")
        .replace("   - ", "\n")
        .replace("- ", "")
    )

    score_match = SCORE_PATTERN.search(normalized_text)
    if score_match:
        parsed_result["checklist_score"] = int(score_match.group(1))

    # Try 4-element vector first (v3), then fall back to 3-element (v2)
    if schema_version == 3:
        vector_match = VECTOR_PATTERN_V3.search(normalized_text)
        if vector_match:
            parsed_result["checklist_result"] = [int(vector_match.group(i)) for i in range(1, 5)]
        else:
            vector_match = VECTOR_PATTERN.search(normalized_text)
            if vector_match:
                parsed_result["checklist_result"] = [int(vector_match.group(i)) for i in range(1, 4)]
    else:
        vector_match = VECTOR_PATTERN.search(normalized_text)
        if vector_match:
            parsed_result["checklist_result"] = [int(vector_match.group(i)) for i in range(1, 4)]

    correctness_match = CORRECTNESS_PATTERN.search(normalized_text)
    if correctness_match:
        parsed_result["overall_correctness"] = correctness_match.group(1).upper()

    missing_fields = []
    if parsed_result["checklist_score"] is None:
        missing_fields.append("checklist_score")
    if parsed_result["checklist_result"] is None:
        missing_fields.append("checklist_result")
    if parsed_result["overall_correctness"] is None:
        missing_fields.append("overall_correctness")

    if missing_fields:
        parsed_result["parse_error"] = "missing " + ", ".join(missing_fields)
        return parsed_result

    vector = parsed_result["checklist_result"]
    if isinstance(vector, list) and len(vector) == 4:
        parsed_result["complete_vikr"] = bool(all(vector))
        parsed_result["dimension_result"] = {dim: vector[i] for i, dim in enumerate(DIM_ORDER)}
    parsed_result["parse_success"] = True
    return parsed_result


def normalize_evaluation(evaluation, schema_version: int = 3) -> dict | None:
    if not evaluation or not isinstance(evaluation, dict):
        return None

    raw_content = evaluation.get("raw_content")
    if isinstance(raw_content, str) and raw_content.strip():
        parsed = parse_judge_output(raw_content, schema_version)
        if parsed["parse_success"]:
            return parsed
        if is_complete_evaluation(evaluation):
            parsed["checklist_score"] = evaluation.get("checklist_score")
            parsed["checklist_result"] = evaluation.get("checklist_result")
            parsed["overall_correctness"] = evaluation.get("overall_correctness")
            parsed["parse_success"] = True
            parsed["parse_error"] = None
        return parsed

    normalized = {
        "raw_content": raw_content,
        "checklist_score": evaluation.get("checklist_score"),
        "checklist_result": evaluation.get("checklist_result"),
        "overall_correctness": evaluation.get("overall_correctness"),
        "complete_vikr": evaluation.get("complete_vikr"),
        "item_results": evaluation.get("item_results"),
        "dimension_result": evaluation.get("dimension_result"),
        "parse_success": False,
        "parse_error": evaluation.get("parse_error"),
    }
    if is_complete_evaluation(normalized):
        normalized["parse_success"] = True
        normalized["parse_error"] = None
    elif not normalized["parse_error"]:
        normalized["parse_error"] = "missing raw_content and incomplete structured fields"
    return normalized


def summarize_results(results: list[dict], schema_version: int = 2) -> dict:
    summary = {
        "total_items": len(results),
        "with_response": 0,
        "with_evaluation": 0,
        "with_raw_content": 0,
        "parsed_success": 0,
        "parse_failed": 0,
    }

    for item in results:
        if item.get("response"):
            summary["with_response"] += 1

        evaluation = item.get("evaluation")
        if not evaluation or not isinstance(evaluation, dict):
            continue

        summary["with_evaluation"] += 1
        if evaluation.get("raw_content"):
            summary["with_raw_content"] += 1

        if is_complete_evaluation(evaluation, schema_version):
            summary["parsed_success"] += 1
        else:
            summary["parse_failed"] += 1

    return summary


def print_summary(summary: dict) -> None:
    print("\nEvaluation summary")
    print(f"  Total items:           {summary['total_items']}")
    print(f"  With response:         {summary['with_response']}")
    print(f"  With evaluation obj:   {summary['with_evaluation']}")
    print(f"  With raw_content:      {summary['with_raw_content']}")
    print(f"  Parsed successfully:   {summary['parsed_success']}")
    print(f"  Parse failed/incomplete: {summary['parse_failed']}")


async def save_results(results: list[dict], output_json: str) -> None:
    """Saves the results list to JSON file with a lock."""
    async with file_lock:
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=json_serializer)
        except Exception as e:
            print(f"Error saving JSON: {e}")


def get_exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    return None


def is_transient_judge_error(exc: Exception) -> bool:
    status_code = get_exception_status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "try again",
            "too many requests",
        )
    )


async def create_judge_completion_with_retry(client, messages: list[dict]):
    max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "6"))
    base_delay = float(os.getenv("JUDGE_RETRY_BASE_SECONDS", "5"))
    max_delay = float(os.getenv("JUDGE_RETRY_MAX_SECONDS", "120"))
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(model=MODEL, messages=messages)
        except Exception as e:
            last_error = e
            if attempt >= max_retries or not is_transient_judge_error(e):
                raise

            delay = min(max_delay, base_delay * (2 ** attempt))
            delay += random.uniform(0, min(1.0, delay * 0.1))
            status = get_exception_status_code(e)
            status_text = f" status={status}" if status else ""
            print(
                f"Transient judge error{status_text}: {e}. "
                f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s."
            )
            await asyncio.sleep(delay)

    raise last_error


async def process_item(item, gt_map, semaphore, all_results, client, output_json, overwrite,
                       schema_version: int = 2, prompt_template: str = ""):
    """Processes a single item and updates it in-place."""
    async with semaphore:
        if is_complete_evaluation(item.get("evaluation"), schema_version) and not overwrite:
            return

        image_path = item["image_path"]
        candidate_response = item.get("response")

        gt_item = gt_map.get(image_path)
        if not gt_item:
            print(f"Warning: No Ground Truth found for {image_path}")
            return

        if not candidate_response:
            print(f"Warning: No response to evaluate for {image_path}")
            return

        if schema_version == 3:
            # v3: GT is the top-level item (already has V/I/K/R structure)
            gt_json_str = json.dumps(gt_item, ensure_ascii=False, indent=2)
            prompt_text = prompt_template.replace("{reference_answer}", gt_json_str).replace(
                "{generated_answer_to_eval}", candidate_response
            )
        else:
            # v2: GT is inside item["annotation"] — still use V3 judge prompt
            gt_annotation = gt_item.get("annotation")
            if not gt_annotation:
                print(f"Warning: No annotation in Ground Truth for {image_path}")
                return
            gt_json_str = json.dumps(gt_annotation, ensure_ascii=False, indent=2)
            prompt_text = prompt_template.replace("{reference_answer}", gt_json_str).replace(
                "{generated_answer_to_eval}", candidate_response
            )

        try:
            judge_delay = float(os.getenv("JUDGE_DELAY_SECONDS", "0"))
            if judge_delay > 0:
                await asyncio.sleep(judge_delay)
            messages = [{"role": "user", "content": prompt_text}]
            response = await create_judge_completion_with_retry(client, messages)
            raw_content = response.choices[0].message.content
            item["evaluation"] = parse_judge_output(raw_content, schema_version)
            await save_results(all_results, output_json)
        except Exception as e:
            print(f"Error evaluating {image_path}: {e}")


def load_json(path: str, label: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} {label}.")
        return data
    except Exception as e:
        print(f"Error reading {label}: {e}")
        return None


async def reparse_existing_results(output_json: str, schema_version: int = 3) -> int:
    print(f"Re-parsing existing judge outputs from {output_json}...")
    if not os.path.exists(output_json):
        print(f"Error: {output_json} not found.")
        return 1

    results = load_json(output_json, "existing evaluation items")
    if results is None:
        return 1

    for item in results:
        if item.get("evaluation") is not None:
            item["evaluation"] = normalize_evaluation(item.get("evaluation"), schema_version)

    await save_results(results, output_json)
    print_summary(summarize_results(results, schema_version))
    return 0


async def async_main(args) -> int:
    schema_version = 3 if args.v3 else 2
    prompt_template = None

    if args.reparse_only:
        return await reparse_existing_results(args.output, schema_version)

    try:
        prompt_template = load_prompt_template(args.prompt)
    except Exception as exc:
        print(f"Error loading judge prompt: {exc}")
        return 1

    bench_path = args.bench or (BENCH_JSON_V3 if schema_version == 3 else BENCH_JSON)
    print(f"Loading Ground Truth from {bench_path} (schema v{schema_version})...")
    bench_data = load_json(bench_path, "GT items")
    if bench_data is None:
        return 1
    gt_map = {item["image_path"]: item for item in bench_data}

    print(f"Loading Candidate Responses from {args.candidate}...")
    candidate_data = load_json(args.candidate, "candidate items")
    if candidate_data is None:
        return 1

    existing_eval_map = {}
    if os.path.exists(args.output):
        existing_list = load_json(args.output, f"existing evaluations from {args.output}")
        if existing_list is None:
            return 1
        for item in existing_list:
            normalized_eval = normalize_evaluation(item.get("evaluation"))
            existing_eval_map[item.get("image_path")] = normalized_eval

    final_results = []
    for candidate_item in candidate_data:
        image_path = candidate_item.get("image_path")
        result_item = {
            "id": candidate_item.get("id"),
            "image_path": image_path,
            "response": candidate_item.get("response"),
            "evaluation": existing_eval_map.get(image_path),
        }
        final_results.append(result_item)

    print(f"Pre-generating Output JSON with {len(final_results)} items...")
    await save_results(final_results, args.output)

    client = build_async_client()
    judge_concurrency = int(os.getenv("JUDGE_CONCURRENCY", "16"))
    semaphore = asyncio.Semaphore(judge_concurrency)
    print(f"Judge concurrency: {judge_concurrency}")

    print(f"Starting evaluation (LIMIT={args.limit})...")
    tasks = []
    processed_count = 0
    for item in final_results:
        if args.limit is not None and processed_count >= args.limit:
            break
        if not is_complete_evaluation(item.get("evaluation"), schema_version) or args.overwrite:
            tasks.append(
                process_item(
                    item,
                    gt_map,
                    semaphore,
                    final_results,
                    client,
                    args.output,
                    args.overwrite,
                    schema_version,
                    prompt_template,
                )
            )
            processed_count += 1

    if tasks:
        print(f"Evaluating {len(tasks)} items (schema v{schema_version})...")
        await tqdm.gather(*tasks)
        print("All tasks completed.")
    else:
        print("No new items to evaluate.")

    print("Final save...")
    await save_results(final_results, args.output)
    print_summary(summarize_results(final_results, schema_version))
    print("Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
