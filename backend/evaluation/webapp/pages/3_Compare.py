"""
Compare Page

Compare multiple evaluation runs side-by-side:
- Metric comparison charts
- Configuration diff
- Category breakdown comparison
"""

import streamlit as st

st.set_page_config(page_title="Compare Runs", page_icon="⚖️", layout="wide")

st.title("⚖️ Compare Runs")


def main():
    """Main page content."""
    from backend.evaluation.webapp.storage import RunStore
    from backend.evaluation.webapp.utils import format_percentage

    store = RunStore()
    runs = store.list_runs(limit=100)

    if len(runs) < 2:
        st.info("Need at least 2 runs to compare. Go to **Run** page to execute evaluations.")
        return

    # Run selection
    st.subheader("Select Runs to Compare")

    run_options = {
        f"{r.run_id} ({r.variant_id}, {r.dataset_name}, F1={format_percentage(r.overall_f1)})": r.run_id
        for r in runs
    }

    selected = st.multiselect(
        "Select 2-5 runs",
        options=list(run_options.keys()),
        max_selections=5,
        help="Select runs to compare side-by-side",
    )

    if len(selected) < 2:
        st.warning("Please select at least 2 runs to compare.")
        return

    selected_ids = [run_options[s] for s in selected]

    # Load full runs
    loaded_runs = []
    for run_id in selected_ids:
        try:
            loaded_runs.append(store.load(run_id))
        except Exception as e:
            st.error(f"Error loading run {run_id}: {e}")

    if len(loaded_runs) < 2:
        return

    st.divider()

    # Metrics comparison
    _show_metrics_comparison(loaded_runs)

    st.divider()

    # Category comparison
    _show_category_comparison(loaded_runs)

    st.divider()

    # Config diff
    _show_config_diff(loaded_runs)


def _show_metrics_comparison(runs):
    """Show metrics comparison chart."""
    from backend.evaluation.webapp.utils import format_percentage

    st.subheader("📊 Metrics Comparison")

    # Prepare data for chart
    metrics_data = []
    for run in runs:
        metrics_data.append({
            "Run": f"{run.run_id[:8]}... ({run.config.variant_id})",
            "Precision": run.result.overall_precision,
            "Recall": run.result.overall_recall,
            "F1": run.result.overall_f1,
        })

    # Display as bar chart
    import pandas as pd

    df = pd.DataFrame(metrics_data)
    df_melted = df.melt(id_vars=["Run"], var_name="Metric", value_name="Score")

    st.bar_chart(
        df_melted,
        x="Run",
        y="Score",
        color="Metric",
        use_container_width=True,
    )

    # Also show as table
    with st.expander("View as Table"):
        table_data = []
        for run in runs:
            table_data.append({
                "Run ID": run.run_id,
                "Variant": run.config.variant_id,
                "Dataset": run.config.dataset_name,
                "Precision": format_percentage(run.result.overall_precision),
                "Recall": format_percentage(run.result.overall_recall),
                "F1": format_percentage(run.result.overall_f1),
                "TP": run.result.total_tp,
                "FP": run.result.total_fp,
                "FN": run.result.total_fn,
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Highlight best
    best_f1 = max(runs, key=lambda r: r.result.overall_f1)
    best_precision = max(runs, key=lambda r: r.result.overall_precision)
    best_recall = max(runs, key=lambda r: r.result.overall_recall)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🏆 Best F1: {best_f1.config.variant_id} ({format_percentage(best_f1.result.overall_f1)})")
    with col2:
        st.info(f"🎯 Best Precision: {best_precision.config.variant_id}")
    with col3:
        st.info(f"🔍 Best Recall: {best_recall.config.variant_id}")


def _show_category_comparison(runs):
    """Show category-wise comparison."""
    from backend.evaluation.webapp.utils import format_percentage

    st.subheader("📁 Category Breakdown")

    # Get all categories
    all_categories = set()
    for run in runs:
        for cat in run.result.category_scores:
            all_categories.add(cat.category.value)

    if not all_categories:
        st.info("No category breakdown available.")
        return

    # Build comparison table
    table_data = []
    for category in sorted(all_categories):
        row = {"Category": category}
        for run in runs:
            cat_score = next(
                (c for c in run.result.category_scores if c.category.value == category),
                None,
            )
            if cat_score:
                row[f"{run.config.variant_id} F1"] = format_percentage(cat_score.f1_score)
            else:
                row[f"{run.config.variant_id} F1"] = "-"
        table_data.append(row)

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _show_config_diff(runs):
    """Show configuration differences between runs."""
    st.subheader("⚙️ Configuration Diff")

    if len(runs) != 2:
        st.info("Select exactly 2 runs for detailed config diff.")

        # Show basic info for all
        for run in runs:
            with st.expander(f"Config: {run.run_id} ({run.config.variant_id})"):
                st.json(run.config.get_effective_params())
        return

    run_a, run_b = runs[0], runs[1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Run A:** `{run_a.run_id}`")
        st.caption(f"Variant: {run_a.config.variant_id}")

    with col2:
        st.markdown(f"**Run B:** `{run_b.run_id}`")
        st.caption(f"Variant: {run_b.config.variant_id}")

    # Compare parameters
    params_a = run_a.config.get_effective_params()
    params_b = run_b.config.get_effective_params()

    all_keys = set(params_a.keys()) | set(params_b.keys())

    diff_data = []
    for key in sorted(all_keys):
        val_a = params_a.get(key, "(not set)")
        val_b = params_b.get(key, "(not set)")
        changed = val_a != val_b
        diff_data.append({
            "Parameter": key,
            f"Run A ({run_a.config.variant_id})": str(val_a),
            f"Run B ({run_b.config.variant_id})": str(val_b),
            "Changed": "✏️" if changed else "",
        })

    st.dataframe(diff_data, use_container_width=True, hide_index=True)

    # Prompt comparison
    st.subheader("📝 Prompt Comparison")

    prompt_a = run_a.config.prompt_snapshot
    prompt_b = run_b.config.prompt_snapshot

    prompt_changed = (
        prompt_a.review_system_hash != prompt_b.review_system_hash
        or prompt_a.review_user_hash != prompt_b.review_user_hash
    )

    if prompt_changed:
        st.warning("⚠️ Prompts are different between these runs")

        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Run A - System Hash: {prompt_a.review_system_hash}")
            st.text(f"Run A - User Hash: {prompt_a.review_user_hash}")
        with col2:
            st.text(f"Run B - System Hash: {prompt_b.review_system_hash}")
            st.text(f"Run B - User Hash: {prompt_b.review_user_hash}")

        # Show full prompts if available
        if prompt_a.review_system_content and prompt_b.review_system_content:
            with st.expander("View System Prompts"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Run A:**")
                    st.code(prompt_a.review_system_content, language="text")
                with col2:
                    st.markdown("**Run B:**")
                    st.code(prompt_b.review_system_content, language="text")
    else:
        st.success("✅ Prompts are identical")
        st.caption(f"System Hash: {prompt_a.review_system_hash}")


if __name__ == "__main__":
    main()
