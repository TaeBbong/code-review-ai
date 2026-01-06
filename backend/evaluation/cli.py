"""
CLI for running evaluations.

Usage:
    # Upload dataset to LangSmith
    python -m backend.evaluation.cli upload v1_initial

    # Run local evaluation (without LangSmith)
    python -m backend.evaluation.cli run-local v1_initial g1-mapreduce

    # Run LangSmith experiment
    python -m backend.evaluation.cli run-langsmith code-review-eval-v1_initial g1-mapreduce

    # Compare variants
    python -m backend.evaluation.cli compare v1_initial g0-baseline g1-mapreduce g2-iterative
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.evaluation.loader import load_dataset_by_name, list_available_datasets
from backend.evaluation.evaluator import Evaluator
from backend.evaluation.schemas import EvalRunResult


def print_result_summary(result: EvalRunResult) -> None:
    """Print evaluation result summary."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Result: {result.variant_id}")
    print("=" * 60)

    print(f"\n  Dataset: {result.dataset_name}")
    print(f"  Samples: {result.total_samples}")
    print(f"  Run ID:  {result.run_id}")
    print(f"  Time:    {result.evaluated_at}")

    print("\n  Overall Metrics:")
    print(f"    Precision: {result.overall_precision:.2%}")
    print(f"    Recall:    {result.overall_recall:.2%}")
    print(f"    F1 Score:  {result.overall_f1:.2%}")

    print(f"\n    TP: {result.total_tp}  FP: {result.total_fp}  FN: {result.total_fn}")

    if result.category_scores:
        print("\n  By Category:")
        print("    " + "-" * 50)
        print(f"    {'Category':<20} {'P':>8} {'R':>8} {'F1':>8} {'Cnt':>6}")
        print("    " + "-" * 50)

        for cs in sorted(result.category_scores, key=lambda x: x.f1_score, reverse=True):
            print(
                f"    {cs.category.value:<20} "
                f"{cs.precision:>7.0%} "
                f"{cs.recall:>7.0%} "
                f"{cs.f1_score:>7.0%} "
                f"{cs.sample_count:>6}"
            )

    print("\n" + "=" * 60 + "\n")


def print_comparison_table(results: list[EvalRunResult]) -> None:
    """Print comparison table for multiple variants."""
    print("\n" + "=" * 70)
    print("  Variant Comparison")
    print("=" * 70)

    print(f"\n  {'Variant':<20} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "-" * 56)

    for r in sorted(results, key=lambda x: x.overall_f1, reverse=True):
        print(
            f"  {r.variant_id:<20} "
            f"{r.overall_precision:>11.1%} "
            f"{r.overall_recall:>11.1%} "
            f"{r.overall_f1:>11.1%}"
        )

    print("\n" + "=" * 70 + "\n")


async def cmd_upload(args: argparse.Namespace) -> None:
    """Upload dataset to LangSmith."""
    from backend.evaluation.langsmith_integration import LangSmithEvaluator

    evaluator = LangSmithEvaluator()
    dataset_id = evaluator.upload_dataset(
        dataset_name=args.dataset,
        langsmith_dataset_name=args.name,
    )
    print(f"\nDataset uploaded: {dataset_id}")


async def cmd_run_local(args: argparse.Namespace) -> None:
    """Run local evaluation without LangSmith."""
    from backend.domain.services.review_service import ReviewService
    from backend.domain.schemas.review import ReviewRequest

    # Load dataset
    dataset = load_dataset_by_name(args.dataset)
    print(f"Loaded dataset: {dataset.name} ({len(dataset.samples)} samples)")

    # Create evaluator
    evaluator = Evaluator(dataset=dataset)

    # Create review service
    review_service = ReviewService()

    # Create review function
    async def review_fn(diff: str, variant_id: str):
        req = ReviewRequest(diff=diff, variant_id=variant_id)
        return await review_service.review(req)

    # Run evaluation
    print(f"Running evaluation with variant: {args.variant}")
    result = await evaluator.run(
        review_fn=review_fn,
        variant_id=args.variant,
        max_concurrency=args.concurrency,
    )

    # Print summary
    print_result_summary(result)

    # Save result if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result.model_dump_json(indent=2))
        print(f"Result saved to: {output_path}")


async def cmd_run_langsmith(args: argparse.Namespace) -> None:
    """Run LangSmith experiment."""
    from backend.evaluation.langsmith_integration import LangSmithEvaluator

    evaluator = LangSmithEvaluator()

    print(f"Running LangSmith experiment:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Variant: {args.variant}")

    results = await evaluator.run_experiment(
        dataset_name=args.dataset,
        variant_id=args.variant,
        experiment_prefix=args.prefix,
        max_concurrency=args.concurrency,
    )

    print("\nExperiment complete!")
    print(f"View results at: https://smith.langchain.com")


async def cmd_compare(args: argparse.Namespace) -> None:
    """Compare multiple variants."""
    from backend.domain.services.review_service import ReviewService
    from backend.domain.schemas.review import ReviewRequest

    # Load dataset
    dataset = load_dataset_by_name(args.dataset)
    print(f"Loaded dataset: {dataset.name} ({len(dataset.samples)} samples)")

    # Create review service (shared across variants)
    review_service = ReviewService()

    results = []

    for variant_id in args.variants:
        print(f"\nEvaluating: {variant_id}")

        # Create evaluator
        evaluator = Evaluator(dataset=dataset)

        # Create review function
        async def review_fn(diff: str, vid: str = variant_id):
            req = ReviewRequest(diff=diff, variant_id=vid)
            return await review_service.review(req)

        # Run evaluation
        result = await evaluator.run(
            review_fn=review_fn,
            variant_id=variant_id,
            max_concurrency=args.concurrency,
        )
        results.append(result)

    # Print comparison
    print_comparison_table(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "results": [r.model_dump() for r in results],
        }
        output_path.write_text(json.dumps(output_data, indent=2, default=str))
        print(f"Comparison saved to: {output_path}")


async def cmd_list(_: argparse.Namespace) -> None:
    """List available datasets."""
    datasets = list_available_datasets()
    print("\nAvailable datasets:")
    for name in datasets:
        dataset = load_dataset_by_name(name)
        print(f"  - {name}: {len(dataset.samples)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Code Review Bot Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload command
    upload_parser = subparsers.add_parser("upload", help="Upload dataset to LangSmith")
    upload_parser.add_argument("dataset", help="Local dataset name")
    upload_parser.add_argument("--name", help="LangSmith dataset name")

    # run-local command
    local_parser = subparsers.add_parser("run-local", help="Run local evaluation")
    local_parser.add_argument("dataset", help="Dataset name")
    local_parser.add_argument("variant", help="Variant ID")
    local_parser.add_argument("--concurrency", type=int, default=4, help="Max concurrency")
    local_parser.add_argument("--output", "-o", help="Output file path")

    # run-langsmith command
    ls_parser = subparsers.add_parser("run-langsmith", help="Run LangSmith experiment")
    ls_parser.add_argument("dataset", help="LangSmith dataset name")
    ls_parser.add_argument("variant", help="Variant ID")
    ls_parser.add_argument("--prefix", help="Experiment name prefix")
    ls_parser.add_argument("--concurrency", type=int, default=4, help="Max concurrency")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare variants")
    compare_parser.add_argument("dataset", help="Dataset name")
    compare_parser.add_argument("variants", nargs="+", help="Variant IDs to compare")
    compare_parser.add_argument("--concurrency", type=int, default=2, help="Max concurrency")
    compare_parser.add_argument("--output", "-o", help="Output file path")

    # list command
    subparsers.add_parser("list", help="List available datasets")

    args = parser.parse_args()

    # Run command
    if args.command == "upload":
        asyncio.run(cmd_upload(args))
    elif args.command == "run-local":
        asyncio.run(cmd_run_local(args))
    elif args.command == "run-langsmith":
        asyncio.run(cmd_run_langsmith(args))
    elif args.command == "compare":
        asyncio.run(cmd_compare(args))
    elif args.command == "list":
        asyncio.run(cmd_list(args))


if __name__ == "__main__":
    main()
