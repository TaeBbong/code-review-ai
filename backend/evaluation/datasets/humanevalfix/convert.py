"""
HumanEvalFix 데이터셋을 우리 평가 형식으로 변환하는 스크립트.

Usage:
    uv run python -m backend.evaluation.datasets.humanevalfix.convert

Requirements:
    pip install datasets
"""

import difflib
import yaml
from pathlib import Path
from datasets import load_dataset


# bug_type → severity 매핑
BUG_TYPE_SEVERITY = {
    "missing logic": "high",
    "excess logic": "medium",
    "operator misuse": "high",
    "variable misuse": "high",
    "value misuse": "medium",
    "function misuse": "high",
}

# bug_type → title keywords 매핑
BUG_TYPE_KEYWORDS = {
    "missing logic": ["missing", "incomplete", "forgot", "omit"],
    "excess logic": ["extra", "unnecessary", "redundant", "excess"],
    "operator misuse": ["operator", "wrong", "incorrect", "!=", "==", "<", ">", "+", "-"],
    "variable misuse": ["variable", "wrong", "incorrect", "typo", "name"],
    "value misuse": ["value", "wrong", "incorrect", "constant", "literal"],
    "function misuse": ["function", "method", "call", "wrong", "incorrect"],
}


def create_unified_diff(
    original_code: str,
    buggy_code: str,
    filename: str = "solution.py"
) -> str:
    """정상 코드 → 버그 코드로의 diff 생성 (버그를 도입하는 변경)."""
    original_lines = original_code.splitlines(keepends=True)
    buggy_lines = buggy_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        buggy_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )

    return "".join(diff)


def convert_sample(idx: int, record: dict) -> dict:
    """HumanEvalFix 레코드를 우리 평가 샘플로 변환."""
    task_id = record["task_id"]
    prompt = record["prompt"]
    canonical = record["canonical_solution"]
    buggy = record["buggy_solution"]
    bug_type = record["bug_type"]
    failure_symptoms = record.get("failure_symptoms", "incorrect output")

    # 전체 코드 구성
    original_code = prompt + canonical
    buggy_code = prompt + buggy

    # diff 생성
    diff = create_unified_diff(original_code, buggy_code)

    # 빈 diff 체크
    if not diff.strip():
        return None

    # severity 결정
    severity = BUG_TYPE_SEVERITY.get(bug_type, "medium")

    # keywords
    title_keywords = BUG_TYPE_KEYWORDS.get(bug_type, ["bug", "error", "issue"])

    # description keywords (failure_symptoms 기반)
    desc_keywords = ["bug", "error", "incorrect"]
    if "incorrect output" in failure_symptoms:
        desc_keywords.extend(["output", "result", "return"])
    if "stackoverflow" in failure_symptoms:
        desc_keywords.extend(["recursion", "stack", "overflow", "infinite"])

    return {
        "id": f"humanevalfix-{idx:03d}",
        "metadata": {
            "source": "humanevalfix",
            "difficulty": "medium",  # HumanEvalFix는 대체로 medium 난이도
            "primary_category": "correctness",
            "tags": [bug_type.replace(" ", "-"), failure_symptoms.replace(" ", "-")],
            "description": f"HumanEvalFix {task_id}: {bug_type}",
            "original_task_id": task_id,
        },
        "input": {
            "diff": diff,
        },
        "expected": {
            "issues": [
                {
                    "category": "correctness",
                    "severity_min": severity,
                    "title_keywords": title_keywords,
                    "description_keywords": desc_keywords,
                    "issue_id": f"exp-{idx:03d}",
                    "rationale": f"Bug type: {bug_type}, Symptom: {failure_symptoms}",
                }
            ],
            "min_issues": 1,
            "max_issues": 3,
        },
    }


def main():
    print("Loading HumanEvalFix dataset from HuggingFace...")
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")

    print(f"Loaded {len(ds)} samples")

    samples = []
    skipped = 0

    for idx, record in enumerate(ds):
        sample = convert_sample(idx, record)
        if sample:
            samples.append(sample)
        else:
            skipped += 1
            print(f"  Skipped {record['task_id']} (empty diff)")

    print(f"Converted {len(samples)} samples, skipped {skipped}")

    # YAML 출력
    dataset = {
        "name": "humanevalfix_python",
        "version": "1.0.0",
        "description": f"HumanEvalFix Python dataset - {len(samples)} samples converted from bigcode/humanevalpack",
        "samples": samples,
    }

    output_path = Path(__file__).parent / "humanevalfix_python.yaml"

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Saved to {output_path}")

    # 통계 출력
    bug_types = {}
    for sample in samples:
        for tag in sample["metadata"]["tags"]:
            if tag in ["missing-logic", "excess-logic", "operator-misuse",
                      "variable-misuse", "value-misuse", "function-misuse"]:
                bug_types[tag] = bug_types.get(tag, 0) + 1

    print("\nBug type distribution:")
    for bt, count in sorted(bug_types.items(), key=lambda x: -x[1]):
        print(f"  {bt}: {count}")


if __name__ == "__main__":
    main()
