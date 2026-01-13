"""
History Page

View past evaluation runs with:
- Filtering by variant/dataset
- Config snapshot viewing
- Result export
"""

import streamlit as st
import json

st.set_page_config(page_title="Run History", page_icon="📜", layout="wide")

st.title("📜 Run History")


def main():
    """Main page content."""
    from backend.evaluation.webapp.storage import RunStore
    from backend.evaluation.webapp.utils import format_duration, format_percentage

    store = RunStore()

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Variant filter
        variants = ["All"] + store.get_unique_variants()
        variant_filter = st.selectbox("Filter by Variant", variants)

    with col2:
        # Dataset filter
        datasets = ["All"] + store.get_unique_datasets()
        dataset_filter = st.selectbox("Filter by Dataset", datasets)

    with col3:
        # Limit
        limit = st.number_input("Max Results", min_value=10, max_value=100, value=50)

    # Apply filters
    runs = store.list_runs(
        variant_id=variant_filter if variant_filter != "All" else None,
        dataset_name=dataset_filter if dataset_filter != "All" else None,
        limit=limit,
    )

    if not runs:
        st.info("No runs found. Go to **Run** page to execute your first evaluation.")
        return

    st.divider()

    # Run list
    st.subheader(f"Runs ({len(runs)})")

    # Create table data
    table_data = []
    for run in runs:
        table_data.append({
            "Run ID": run.run_id,
            "Variant": run.variant_id,
            "Dataset": run.dataset_name,
            "Samples": run.total_samples,
            "Precision": format_percentage(run.overall_precision),
            "Recall": format_percentage(run.overall_recall),
            "F1": format_percentage(run.overall_f1),
            "Duration": format_duration(run.duration_seconds),
            "Date": run.created_at[:10],
        })

    # Display as dataframe with selection
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.divider()

    # Run detail selector
    run_ids = [r.run_id for r in runs]
    selected_run_id = st.selectbox(
        "Select Run for Details",
        options=run_ids,
        help="Select a run to view its configuration and full results",
    )

    if selected_run_id:
        _show_run_details(store, selected_run_id)


def _show_run_details(store, run_id: str):
    """Show detailed information for a run."""
    from backend.evaluation.webapp.utils import format_duration, format_percentage

    try:
        run = store.load(run_id)
    except FileNotFoundError:
        st.error(f"Run not found: {run_id}")
        return

    st.subheader(f"Run Details: `{run_id}`")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Variant", run.config.variant_id)
    with col2:
        st.metric("Dataset", run.config.dataset_name)
    with col3:
        st.metric("Duration", format_duration(run.duration_seconds))
    with col4:
        st.metric("Concurrency", run.config.max_concurrency)

    # Metrics
    st.subheader("Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", format_percentage(run.result.overall_precision))
    with col2:
        st.metric("Recall", format_percentage(run.result.overall_recall))
    with col3:
        st.metric("F1 Score", format_percentage(run.result.overall_f1))
    with col4:
        st.metric("Samples", run.result.total_samples)

    # TP/FP/FN breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Positives", run.result.total_tp, help="Correctly identified issues")
    with col2:
        st.metric("False Positives", run.result.total_fp, help="Incorrectly identified issues")
    with col3:
        st.metric("False Negatives", run.result.total_fn, help="Missed issues")

    # Configuration snapshot
    with st.expander("📋 Configuration Snapshot", expanded=False):
        st.subheader("Preset Parameters")
        st.json(run.config.preset_params)

        if run.config.overrides:
            st.subheader("Overrides Applied")
            st.json(run.config.overrides)

        st.subheader("Effective Parameters")
        st.json(run.config.get_effective_params())

    # Prompt snapshot
    with st.expander("📝 Prompt Snapshot", expanded=False):
        prompt = run.config.prompt_snapshot

        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Pack ID: {prompt.pack_id}")
        with col2:
            st.text(f"System Hash: {prompt.review_system_hash}")

        if prompt.review_system_content:
            st.subheader("System Prompt")
            st.code(prompt.review_system_content, language="text")

        if prompt.review_user_content:
            st.subheader("User Prompt Template")
            st.code(prompt.review_user_content, language="text")

    # Category breakdown
    if run.result.category_scores:
        with st.expander("📊 Category Breakdown", expanded=False):
            cat_data = []
            for cat in run.result.category_scores:
                cat_data.append({
                    "Category": cat.category.value,
                    "Samples": cat.sample_count,
                    "TP": cat.total_tp,
                    "FP": cat.total_fp,
                    "FN": cat.total_fn,
                    "Precision": format_percentage(cat.precision),
                    "Recall": format_percentage(cat.recall),
                    "F1": format_percentage(cat.f1_score),
                })
            st.dataframe(cat_data, use_container_width=True, hide_index=True)

    # Round-by-round results (for g4-multireview)
    if run.result.round_predictions:
        _show_round_details(run)

    # Tags and notes
    with st.expander("🏷️ Tags & Notes", expanded=False):
        # Tags editor
        current_tags = run.tags or []
        tags_str = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(current_tags),
            help="Add tags to categorize this run",
        )
        new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        # Notes editor
        notes = st.text_area(
            "Notes",
            value=run.notes,
            help="Add notes about this run",
        )

        if st.button("Save Tags & Notes"):
            if new_tags != current_tags:
                store.update_tags(run_id, new_tags)
            if notes != run.notes:
                store.update_notes(run_id, notes)
            st.success("Saved!")
            st.rerun()

    # Export
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        # Export full run
        run_json = run.model_dump_json(indent=2)
        st.download_button(
            "📥 Export Full Run (JSON)",
            data=run_json,
            file_name=f"run_{run_id}.json",
            mime="application/json",
        )

    with col2:
        # Export results only
        result_json = run.result.model_dump_json(indent=2)
        st.download_button(
            "📥 Export Results Only",
            data=result_json,
            file_name=f"results_{run_id}.json",
            mime="application/json",
        )

    with col3:
        # Delete button
        if st.button("🗑️ Delete Run", type="secondary"):
            store.delete(run_id)
            st.success(f"Deleted run: {run_id}")
            st.rerun()


def _show_round_details(run):
    """Show round-by-round results for multi-review runs."""
    from backend.evaluation.webapp.storage.schemas import StoredRun

    round_predictions = run.result.round_predictions
    if not round_predictions:
        return

    with st.expander("🔄 Round-by-Round Results (Multi-Review)", expanded=False):
        st.caption(
            "Each sample was reviewed multiple times independently. "
            "Compare what each round found before aggregation."
        )

        # Sample selector
        sample_ids = list(round_predictions.keys())
        selected_sample = st.selectbox(
            "Select Sample",
            options=sample_ids,
            help="Choose a sample to view its round-by-round results",
            key="round_sample_selector",
        )

        if selected_sample and selected_sample in round_predictions:
            rounds = round_predictions[selected_sample]
            num_rounds = len(rounds)

            st.subheader(f"Sample: `{selected_sample}` ({num_rounds} rounds)")

            # Show aggregated result for comparison
            if selected_sample in run.result.predictions:
                final_result = run.result.predictions[selected_sample]
                st.markdown("**Final Aggregated Result:**")
                st.markdown(f"- Issues: **{len(final_result.issues)}**")
                st.markdown(f"- Risk Level: **{final_result.summary.overall_risk.value}**")

            st.divider()

            # Round comparison tabs
            if num_rounds > 0:
                tab_labels = [f"Round {i+1}" for i in range(num_rounds)]
                tabs = st.tabs(tab_labels)

                for i, tab in enumerate(tabs):
                    with tab:
                        round_result = rounds[i]
                        _show_single_round(round_result, i + 1)

            # Summary comparison table
            st.divider()
            st.subheader("Round Comparison")

            comparison_data = []
            for i, round_result in enumerate(rounds, 1):
                issue_titles = [iss.title[:50] + "..." if len(iss.title) > 50 else iss.title
                               for iss in round_result.issues]
                comparison_data.append({
                    "Round": i,
                    "Issues Found": len(round_result.issues),
                    "Risk Level": round_result.summary.overall_risk.value,
                    "Issue Titles": ", ".join(issue_titles) if issue_titles else "(none)",
                })

            st.dataframe(comparison_data, use_container_width=True, hide_index=True)

            # Check consistency
            issue_counts = [len(r.issues) for r in rounds]
            if len(set(issue_counts)) > 1:
                st.info(
                    f"⚠️ Issue counts vary across rounds: {issue_counts}. "
                    "This is expected with LLM variability (even at temperature=0)."
                )
            else:
                st.success(
                    f"✅ All {num_rounds} rounds found the same number of issues: {issue_counts[0]}"
                )


def _show_single_round(round_result, round_num: int):
    """Display a single round's review result."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Issues Found", len(round_result.issues))

    with col2:
        st.metric("Risk Level", round_result.summary.overall_risk.value)

    if round_result.issues:
        st.markdown("**Issues:**")
        for issue in round_result.issues:
            severity_emoji = {
                "blocker": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(issue.severity.value, "⚪")

            with st.container():
                st.markdown(f"{severity_emoji} **{issue.title}**")
                st.caption(f"Category: {issue.category.value} | Severity: {issue.severity.value}")
                if issue.description:
                    st.markdown(f"> {issue.description[:200]}..." if len(issue.description) > 200 else f"> {issue.description}")
    else:
        st.info("No issues found in this round.")


if __name__ == "__main__":
    main()
