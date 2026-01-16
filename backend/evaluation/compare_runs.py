"""
Compare evaluation runs and generate visualizations.

Usage:
    uv run python -m backend.evaluation.compare_runs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_run(path: Path) -> Dict[str, Any]:
    """Load a single evaluation run from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a run."""
    result = run["result"]
    config = run["config"]

    # Calculate overall metrics
    sample_scores = result.get("sample_scores", [])

    total_tp = sum(s["true_positives"] for s in sample_scores)
    total_fp = sum(s["false_positives"] for s in sample_scores)
    total_fn = sum(s["false_negatives"] for s in sample_scores)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-category metrics
    category_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "samples": []})

    for score in sample_scores:
        sample_id = score["sample_id"]
        # Extract category from sample_id (e.g., "correctness-001" -> "correctness")
        category = sample_id.rsplit("-", 1)[0]

        category_metrics[category]["tp"] += score["true_positives"]
        category_metrics[category]["fp"] += score["false_positives"]
        category_metrics[category]["fn"] += score["false_negatives"]
        category_metrics[category]["samples"].append({
            "id": sample_id,
            "precision": score["precision"],
            "recall": score["recall"],
            "f1_score": score["f1_score"],
            "tp": score["true_positives"],
            "fp": score["false_positives"],
            "fn": score["false_negatives"],
        })

    # Calculate per-category F1
    for cat, metrics in category_metrics.items():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["precision"] = p
        metrics["recall"] = r
        metrics["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {
        "run_id": run["run_id"],
        "variant_id": config["variant_id"],
        "dataset": config["dataset_name"],
        "duration_seconds": run["duration_seconds"],
        "total_samples": len(sample_scores),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "category_metrics": dict(category_metrics),
        "sample_scores": sample_scores,
    }


def print_comparison(runs: List[Dict[str, Any]]) -> None:
    """Print a text comparison of runs."""
    print("\n" + "=" * 80)
    print("EVALUATION RUN COMPARISON")
    print("=" * 80)

    for run in runs:
        print(f"\n--- {run['variant_id']} ---")
        print(f"Dataset: {run['dataset']}")
        print(f"Duration: {run['duration_seconds']:.1f}s")
        print(f"Samples: {run['total_samples']}")
        print(f"\nOverall Metrics:")
        print(f"  Precision: {run['precision']:.1%}")
        print(f"  Recall:    {run['recall']:.1%}")
        print(f"  F1 Score:  {run['f1']:.1%}")
        print(f"\n  TP: {run['total_tp']}, FP: {run['total_fp']}, FN: {run['total_fn']}")

        print(f"\nPer-Category F1:")
        for cat, metrics in sorted(run['category_metrics'].items()):
            print(f"  {cat:20s}: {metrics['f1']:.1%} (P={metrics['precision']:.1%}, R={metrics['recall']:.1%})")

    # Head-to-head comparison
    if len(runs) == 2:
        r1, r2 = runs
        print("\n" + "=" * 80)
        print(f"HEAD-TO-HEAD: {r1['variant_id']} vs {r2['variant_id']}")
        print("=" * 80)

        print(f"\n{'Metric':<20} {r1['variant_id']:<20} {r2['variant_id']:<20} {'Δ':>10}")
        print("-" * 70)

        delta_f1 = r2['f1'] - r1['f1']
        delta_p = r2['precision'] - r1['precision']
        delta_r = r2['recall'] - r1['recall']

        print(f"{'F1 Score':<20} {r1['f1']:<20.1%} {r2['f1']:<20.1%} {delta_f1:>+10.1%}")
        print(f"{'Precision':<20} {r1['precision']:<20.1%} {r2['precision']:<20.1%} {delta_p:>+10.1%}")
        print(f"{'Recall':<20} {r1['recall']:<20.1%} {r2['recall']:<20.1%} {delta_r:>+10.1%}")

        # Per-category comparison
        print(f"\n{'Category':<20} {r1['variant_id']:<12} {r2['variant_id']:<12} {'Δ F1':>10}")
        print("-" * 54)

        all_categories = set(r1['category_metrics'].keys()) | set(r2['category_metrics'].keys())
        for cat in sorted(all_categories):
            f1_1 = r1['category_metrics'].get(cat, {}).get('f1', 0)
            f1_2 = r2['category_metrics'].get(cat, {}).get('f1', 0)
            delta = f1_2 - f1_1
            print(f"{cat:<20} {f1_1:<12.1%} {f1_2:<12.1%} {delta:>+10.1%}")


def analyze_failures(runs: List[Dict[str, Any]]) -> None:
    """Analyze where each run fails."""
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    for run in runs:
        print(f"\n--- {run['variant_id']} ---")

        # Find samples with F1 < 1.0
        failures = []
        for score in run['sample_scores']:
            if score['f1_score'] < 1.0:
                failures.append({
                    'id': score['sample_id'],
                    'f1': score['f1_score'],
                    'tp': score['true_positives'],
                    'fp': score['false_positives'],
                    'fn': score['false_negatives'],
                    'reason': 'FP' if score['false_positives'] > 0 else 'FN' if score['false_negatives'] > 0 else 'Unknown'
                })

        if not failures:
            print("  No failures! Perfect score.")
            continue

        # Group by failure type
        fp_failures = [f for f in failures if f['fp'] > 0]
        fn_failures = [f for f in failures if f['fn'] > 0 and f['fp'] == 0]

        print(f"\n  False Positives (over-reporting): {len(fp_failures)} samples")
        for f in fp_failures[:5]:
            print(f"    - {f['id']}: {f['fp']} extra issue(s)")

        print(f"\n  False Negatives (under-reporting): {len(fn_failures)} samples")
        for f in fn_failures[:5]:
            print(f"    - {f['id']}: missed {f['fn']} issue(s)")


def create_visualizations(runs: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Overall comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    variants = [r['variant_id'] for r in runs]
    x = np.arange(len(variants))
    width = 0.25

    precision = [r['precision'] for r in runs]
    recall = [r['recall'] for r in runs]
    f1 = [r['f1'] for r in runs]

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150)
    plt.close()

    # 2. Per-category F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    all_categories = sorted(set().union(*[r['category_metrics'].keys() for r in runs]))
    x = np.arange(len(all_categories))
    width = 0.35

    colors = ['#3498db', '#e74c3c']

    for i, run in enumerate(runs):
        f1_scores = [run['category_metrics'].get(cat, {}).get('f1', 0) for cat in all_categories]
        offset = (i - len(runs)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=run['variant_id'], color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories)
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=150)
    plt.close()

    # 3. TP/FP/FN stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variants))
    width = 0.5

    tps = [r['total_tp'] for r in runs]
    fps = [r['total_fp'] for r in runs]
    fns = [r['total_fn'] for r in runs]

    ax.bar(x, tps, width, label='True Positives (TP)', color='#2ecc71')
    ax.bar(x, fps, width, bottom=tps, label='False Positives (FP)', color='#e74c3c')
    ax.bar(x, fns, width, bottom=[t+f for t,f in zip(tps, fps)], label='False Negatives (FN)', color='#f39c12')

    ax.set_ylabel('Count')
    ax.set_title('Issue Detection Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()

    # Add counts
    for i, (tp, fp, fn) in enumerate(zip(tps, fps, fns)):
        ax.annotate(f'TP:{tp}', xy=(i, tp/2), ha='center', va='center', color='white', fontweight='bold')
        if fp > 0:
            ax.annotate(f'FP:{fp}', xy=(i, tp + fp/2), ha='center', va='center', color='white', fontweight='bold')
        if fn > 0:
            ax.annotate(f'FN:{fn}', xy=(i, tp + fp + fn/2), ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'detection_breakdown.png', dpi=150)
    plt.close()

    # 4. Per-sample F1 heatmap comparison (if 2 runs)
    if len(runs) == 2:
        r1, r2 = runs

        # Build sample-level comparison
        samples_r1 = {s['sample_id']: s['f1_score'] for s in r1['sample_scores']}
        samples_r2 = {s['sample_id']: s['f1_score'] for s in r2['sample_scores']}

        all_samples = sorted(set(samples_r1.keys()) | set(samples_r2.keys()))

        fig, ax = plt.subplots(figsize=(14, max(8, len(all_samples) * 0.15)))

        data = []
        for sample in all_samples:
            f1_1 = samples_r1.get(sample, 0)
            f1_2 = samples_r2.get(sample, 0)
            data.append([f1_1, f1_2])

        data = np.array(data)

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([r1['variant_id'], r2['variant_id']])
        ax.set_yticks(range(len(all_samples)))
        ax.set_yticklabels(all_samples, fontsize=8)

        plt.colorbar(im, ax=ax, label='F1 Score')
        ax.set_title('Per-Sample F1 Score Comparison')

        # Add text annotations
        for i in range(len(all_samples)):
            for j in range(2):
                val = data[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=color, fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / 'sample_comparison_heatmap.png', dpi=150)
        plt.close()

        # 5. Improvement/Regression scatter
        fig, ax = plt.subplots(figsize=(10, 8))

        improvements = []
        regressions = []
        no_change = []

        for sample in all_samples:
            f1_1 = samples_r1.get(sample, 0)
            f1_2 = samples_r2.get(sample, 0)
            delta = f1_2 - f1_1

            if delta > 0.01:
                improvements.append((sample, f1_1, f1_2))
            elif delta < -0.01:
                regressions.append((sample, f1_1, f1_2))
            else:
                no_change.append((sample, f1_1, f1_2))

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')

        # Plot points
        if improvements:
            x_imp = [p[1] for p in improvements]
            y_imp = [p[2] for p in improvements]
            ax.scatter(x_imp, y_imp, c='#2ecc71', s=100, label=f'Improved ({len(improvements)})', alpha=0.7)

        if regressions:
            x_reg = [p[1] for p in regressions]
            y_reg = [p[2] for p in regressions]
            ax.scatter(x_reg, y_reg, c='#e74c3c', s=100, label=f'Regressed ({len(regressions)})', alpha=0.7)

        if no_change:
            x_nc = [p[1] for p in no_change]
            y_nc = [p[2] for p in no_change]
            ax.scatter(x_nc, y_nc, c='#95a5a6', s=50, label=f'No change ({len(no_change)})', alpha=0.5)

        ax.set_xlabel(f'{r1["variant_id"]} F1 Score')
        ax.set_ylabel(f'{r2["variant_id"]} F1 Score')
        ax.set_title('Per-Sample F1 Score: Improvements vs Regressions')
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_scatter.png', dpi=150)
        plt.close()

    print(f"\nVisualizations saved to: {output_dir}")


def generate_detailed_report(runs: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate a detailed markdown report."""
    report_path = output_dir / "comparison_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Run Comparison Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n\n")

        # Summary table
        f.write("## Overall Summary\n\n")
        f.write("| Metric | " + " | ".join(r['variant_id'] for r in runs) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(runs)) + "|\n")
        f.write("| Dataset | " + " | ".join(r['dataset'] for r in runs) + " |\n")
        f.write("| Samples | " + " | ".join(str(r['total_samples']) for r in runs) + " |\n")
        f.write("| Duration | " + " | ".join(f"{r['duration_seconds']:.0f}s" for r in runs) + " |\n")
        f.write("| **F1 Score** | " + " | ".join(f"**{r['f1']:.1%}**" for r in runs) + " |\n")
        f.write("| Precision | " + " | ".join(f"{r['precision']:.1%}" for r in runs) + " |\n")
        f.write("| Recall | " + " | ".join(f"{r['recall']:.1%}" for r in runs) + " |\n")
        f.write("| TP | " + " | ".join(str(r['total_tp']) for r in runs) + " |\n")
        f.write("| FP | " + " | ".join(str(r['total_fp']) for r in runs) + " |\n")
        f.write("| FN | " + " | ".join(str(r['total_fn']) for r in runs) + " |\n")

        # Category breakdown
        f.write("\n## Per-Category F1 Scores\n\n")
        all_categories = sorted(set().union(*[r['category_metrics'].keys() for r in runs]))

        f.write("| Category | " + " | ".join(r['variant_id'] for r in runs) + " |\n")
        f.write("|----------|" + "|".join(["--------"] * len(runs)) + "|\n")

        for cat in all_categories:
            row = f"| {cat} |"
            for run in runs:
                f1 = run['category_metrics'].get(cat, {}).get('f1', 0)
                row += f" {f1:.1%} |"
            f.write(row + "\n")

        # Delta analysis (for 2 runs)
        if len(runs) == 2:
            r1, r2 = runs
            f.write(f"\n## Delta Analysis: {r1['variant_id']} → {r2['variant_id']}\n\n")

            delta_f1 = r2['f1'] - r1['f1']
            delta_p = r2['precision'] - r1['precision']
            delta_r = r2['recall'] - r1['recall']

            f.write(f"- **F1 Score**: {delta_f1:+.1%}\n")
            f.write(f"- **Precision**: {delta_p:+.1%}\n")
            f.write(f"- **Recall**: {delta_r:+.1%}\n")

            # Sample-level changes
            samples_r1 = {s['sample_id']: s for s in r1['sample_scores']}
            samples_r2 = {s['sample_id']: s for s in r2['sample_scores']}

            improvements = []
            regressions = []

            for sample_id in samples_r1:
                s1 = samples_r1[sample_id]
                s2 = samples_r2.get(sample_id, {'f1_score': 0})
                delta = s2['f1_score'] - s1['f1_score']

                if delta > 0.01:
                    improvements.append((sample_id, s1['f1_score'], s2['f1_score'], delta))
                elif delta < -0.01:
                    regressions.append((sample_id, s1['f1_score'], s2['f1_score'], delta))

            f.write(f"\n### Improvements ({len(improvements)} samples)\n\n")
            if improvements:
                f.write("| Sample | Before | After | Δ |\n")
                f.write("|--------|--------|-------|---|\n")
                for sample_id, before, after, delta in sorted(improvements, key=lambda x: -x[3]):
                    f.write(f"| {sample_id} | {before:.0%} | {after:.0%} | {delta:+.0%} |\n")
            else:
                f.write("None\n")

            f.write(f"\n### Regressions ({len(regressions)} samples)\n\n")
            if regressions:
                f.write("| Sample | Before | After | Δ |\n")
                f.write("|--------|--------|-------|---|\n")
                for sample_id, before, after, delta in sorted(regressions, key=lambda x: x[3]):
                    f.write(f"| {sample_id} | {before:.0%} | {after:.0%} | {delta:+.0%} |\n")
            else:
                f.write("None\n")

        # Visualizations
        f.write("\n## Visualizations\n\n")
        f.write("![Overall Comparison](overall_comparison.png)\n\n")
        f.write("![Category Comparison](category_comparison.png)\n\n")
        f.write("![Detection Breakdown](detection_breakdown.png)\n\n")
        if len(runs) == 2:
            f.write("![Sample Comparison Heatmap](sample_comparison_heatmap.png)\n\n")
            f.write("![Improvement Scatter](improvement_scatter.png)\n\n")

    print(f"Report saved to: {report_path}")


def main():
    """Main entry point."""
    runs_dir = Path(__file__).parent / "data" / "runs"
    output_dir = Path(__file__).parent / "data" / "analysis"

    # Load runs
    run_files = list(runs_dir.glob("*.json"))
    print(f"Found {len(run_files)} run files:")
    for f in run_files:
        print(f"  - {f.name}")

    runs = []
    for f in run_files:
        run_data = load_run(f)
        metrics = extract_metrics(run_data)
        runs.append(metrics)

    # Sort by variant_id for consistent ordering
    runs.sort(key=lambda r: r['variant_id'])

    # Print comparison
    print_comparison(runs)

    # Analyze failures
    analyze_failures(runs)

    # Create visualizations
    create_visualizations(runs, output_dir)

    # Generate report
    generate_detailed_report(runs, output_dir)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
