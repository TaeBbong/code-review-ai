"""
Samples Page

Explore per-sample results:
- Sample list with metrics
- Issue matching details
- Diff and response viewer
"""

import streamlit as st

st.set_page_config(page_title="Sample Details", page_icon="🔍", layout="wide")

st.title("🔍 Sample Details")


def main():
    """Main page content."""
    from backend.evaluation.webapp.storage import RunStore
    from backend.evaluation.webapp.utils import format_percentage

    store = RunStore()
    runs = store.list_runs(limit=100)

    if not runs:
        st.info("No runs found. Go to **Run** page to execute your first evaluation.")
        return

    # Run selector
    run_options = {
        f"{r.run_id} ({r.variant_id}, F1={format_percentage(r.overall_f1)})": r.run_id
        for r in runs
    }

    selected_label = st.selectbox(
        "Select Run",
        options=list(run_options.keys()),
        help="Choose a run to explore its samples",
    )

    if not selected_label:
        return

    run_id = run_options[selected_label]

    try:
        run = store.load(run_id)
    except Exception as e:
        st.error(f"Error loading run: {e}")
        return

    st.divider()

    # Sample overview
    _show_sample_overview(run)

    st.divider()

    # Sample selector and details
    _show_sample_details(run)


def _show_sample_overview(run):
    """Show overview of all samples."""
    from backend.evaluation.webapp.utils import format_percentage

    st.subheader(f"Samples Overview ({run.result.total_samples} samples)")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        perfect = sum(1 for s in run.result.sample_scores if s.f1_score == 1.0)
        st.metric("Perfect (F1=1.0)", perfect)
    with col2:
        good = sum(1 for s in run.result.sample_scores if 0.5 <= s.f1_score < 1.0)
        st.metric("Good (F1>=0.5)", good)
    with col3:
        poor = sum(1 for s in run.result.sample_scores if 0 < s.f1_score < 0.5)
        st.metric("Poor (F1<0.5)", poor)
    with col4:
        failed = sum(1 for s in run.result.sample_scores if s.f1_score == 0)
        st.metric("Failed (F1=0)", failed)

    # Sample table
    table_data = []
    for score in run.result.sample_scores:
        # Determine status emoji
        if score.f1_score == 1.0:
            status = "✅"
        elif score.f1_score >= 0.5:
            status = "⚠️"
        elif score.f1_score > 0:
            status = "🔶"
        else:
            status = "❌"

        table_data.append({
            "Sample ID": score.sample_id,
            "TP": score.true_positives,
            "FP": score.false_positives,
            "FN": score.false_negatives,
            "Precision": format_percentage(score.precision),
            "Recall": format_percentage(score.recall),
            "F1": format_percentage(score.f1_score),
            "Status": status,
        })

    # Sort by F1 ascending (worst first for debugging)
    table_data.sort(key=lambda x: float(x["F1"].rstrip("%")))

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _show_sample_details(run):
    """Show detailed view of a specific sample."""
    from backend.evaluation.webapp.utils import format_percentage
    from backend.evaluation.loader import load_dataset_by_name

    st.subheader("Sample Details")

    # Sample selector
    sample_ids = [s.sample_id for s in run.result.sample_scores]
    selected_sample = st.selectbox(
        "Select Sample",
        options=sample_ids,
        help="Choose a sample to view details",
    )

    if not selected_sample:
        return

    # Get sample score
    score = next(
        (s for s in run.result.sample_scores if s.sample_id == selected_sample),
        None,
    )
    if not score:
        st.error("Sample score not found")
        return

    # Get prediction
    prediction = run.result.predictions.get(selected_sample)

    # Try to load original sample from dataset
    original_sample = None
    try:
        dataset = load_dataset_by_name(run.config.dataset_name)
        original_sample = next(
            (s for s in dataset.samples if s.id == selected_sample),
            None,
        )
    except Exception:
        pass

    # Sample metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Precision", format_percentage(score.precision))
    with col2:
        st.metric("Recall", format_percentage(score.recall))
    with col3:
        st.metric("F1", format_percentage(score.f1_score))
    with col4:
        st.metric("TP", score.true_positives)
    with col5:
        st.metric("FP", score.false_positives)
    with col6:
        st.metric("FN", score.false_negatives)

    # Expected vs Predicted issues
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Expected Issues")
        if original_sample:
            for issue in original_sample.expected.issues:
                # Check if this issue was matched
                match = next(
                    (m for m in score.issue_matches if m.expected_id == issue.issue_id),
                    None,
                )
                matched = match and match.matched if match else False

                status = "✅ MATCHED" if matched else "❌ NOT FOUND"
                st.markdown(f"""
                **[{issue.category.value}]** {', '.join(issue.title_keywords[:3]) if issue.title_keywords else 'Issue'}

                - Severity Min: `{issue.severity_min.value}`
                - Keywords: {', '.join(issue.title_keywords) if issue.title_keywords else 'N/A'}
                - Status: {status}
                """)
                if issue.rationale:
                    st.caption(f"Rationale: {issue.rationale}")
                st.divider()
        else:
            st.info("Original sample not available")

    with col2:
        st.markdown("### Predicted Issues")
        if prediction:
            for i, issue in enumerate(prediction.issues):
                # Check if this is a TP or FP
                issue_id = issue.id if hasattr(issue, 'id') else str(i)
                is_fp = issue_id in score.unmatched_predictions or str(i) in score.unmatched_predictions

                status = "⚠️ FP" if is_fp else "✅ TP"

                st.markdown(f"""
                **[{issue.category.value}]** {issue.title}

                - Severity: `{issue.severity.value}`
                - File: `{issue.file_path or 'N/A'}`
                - Line: {issue.line_start or 'N/A'}
                - Status: {status}
                """)
                if issue.description:
                    with st.expander("Description"):
                        st.write(issue.description)
                st.divider()

            if not prediction.issues:
                st.warning("No issues predicted")
        else:
            st.info("Prediction not available")

    # Issue matching details
    if score.issue_matches:
        with st.expander("🔗 Issue Matching Details"):
            for match in score.issue_matches:
                st.markdown(f"""
                - Expected: `{match.expected_id}`
                - Predicted: `{match.predicted_id or 'None'}`
                - Matched: {'✅' if match.matched else '❌'}
                """)
                if match.match_details:
                    st.json(match.match_details)

    # Original diff
    if original_sample:
        with st.expander("📄 Original Diff"):
            st.code(original_sample.input.diff, language="diff")

    # Full prediction response
    if prediction:
        with st.expander("📋 Full Prediction Response"):
            st.json(prediction.model_dump())

    # Summary section
    if prediction and prediction.summary:
        with st.expander("📝 Review Summary"):
            summary = prediction.summary
            st.markdown(f"**Risk Level:** {summary.risk_level.value if summary.risk_level else 'N/A'}")
            st.markdown(f"**Has Blockers:** {'Yes' if summary.has_blockers else 'No'}")

            if summary.key_points:
                st.markdown("**Key Points:**")
                for point in summary.key_points:
                    st.markdown(f"- {point}")


if __name__ == "__main__":
    main()
