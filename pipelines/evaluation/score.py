import json
import os
import sys

# Configuration
INPUT_FILE = "data/output/meme_eval_results_qwen3vl_flash.json"


def is_parsed_evaluation(eval_data, schema_version: int = 2):
    if not eval_data or not isinstance(eval_data, dict):
        return False

    vector = eval_data.get("checklist_result")
    expected_len = 4 if schema_version == 3 else 3
    return (
        isinstance(eval_data.get("checklist_score"), int)
        and isinstance(vector, list)
        and len(vector) == expected_len
        and eval_data.get("overall_correctness") in {"YES", "NO"}
    )


def calculate_scores():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print(f"Reading evaluation results from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # Auto-detect schema version from first valid vector
    schema_version = 2
    for item in data:
        ev = item.get("evaluation") or {}
        vec = ev.get("checklist_result")
        if vec and isinstance(vec, list):
            schema_version = 3 if len(vec) == 4 else 2
            break

    total_items = 0
    evaluated_items = 0
    parsed_evals = 0
    valid_evals = 0
    overall_correct_count = 0
    dim_count_slots = 4 if schema_version == 3 else 3
    dim_counts = [0] * dim_count_slots
    total_checklist_score = 0

    for item in data:
        total_items += 1
        eval_data = item.get('evaluation')

        if not eval_data or not isinstance(eval_data, dict):
            continue

        evaluated_items += 1
        if is_parsed_evaluation(eval_data, schema_version):
            parsed_evals += 1

        if eval_data.get('overall_correctness') is None:
            continue

        valid_evals += 1

        if eval_data.get('overall_correctness') == "YES":
            overall_correct_count += 1

        vector = eval_data.get('checklist_result')
        if vector and len(vector) == dim_count_slots:
            for i in range(dim_count_slots):
                dim_counts[i] += vector[i]

        score = eval_data.get('checklist_score')
        if score is not None:
            total_checklist_score += score

    if valid_evals == 0:
        print("No valid evaluations found.")
        return

    pass_rate = (overall_correct_count / valid_evals) * 100
    avg_checklist_score = total_checklist_score / valid_evals

    print("\n" + "=" * 44)
    print(f"  MemeBench Evaluation Report (v{schema_version})")
    print("=" * 44)
    print(f"File: {INPUT_FILE}")
    print(f"Total Items: {total_items}")
    print(f"Evaluated Items: {evaluated_items}")
    print(f"Parsed Evaluations: {parsed_evals}")
    print(f"Valid Evaluations: {valid_evals} / {total_items}")
    print("-" * 44)
    print(f"Overall Pass Rate:       {pass_rate:.2f}%")
    print(f"Avg Checklist Score:     {avg_checklist_score:.2f} / {dim_count_slots}")
    print("-" * 44)
    print("Dimension Accuracy:")
    if schema_version == 3:
        dim_labels = ["Visual (V)", "Identity (I)", "Knowledge (K)", "Reasoning (R)"]
    else:
        dim_labels = ["Visual (V)", "Knowledge (K)", "Reasoning (R)"]
    for i, label in enumerate(dim_labels):
        acc = (dim_counts[i] / valid_evals) * 100
        print(f"  [{label:<15}]: {acc:.2f}%")
    print("=" * 44 + "\n")

def main(argv=None):
    global INPUT_FILE

    if argv is None:
        argv = sys.argv[1:]

    if argv:
        INPUT_FILE = argv[0]

    calculate_scores()


if __name__ == "__main__":
    main()
