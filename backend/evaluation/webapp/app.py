"""
Evaluation Dashboard - Streamlit App

Main entry point for the evaluation web UI.
Run with: streamlit run backend/evaluation/webapp/app.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main app entry point."""
    st.title("📊 Code Review Bot - Evaluation Dashboard")

    st.markdown("""
    Welcome to the Evaluation Dashboard. Use the sidebar to navigate:

    - **Run**: Execute new evaluations with custom configurations
    - **History**: View past evaluation runs and their results
    - **Compare**: Compare multiple runs side-by-side
    - **Samples**: Explore per-sample results in detail
    """)

    # Quick stats in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Available Datasets", _count_datasets())

    with col2:
        st.metric("Available Variants", _count_variants())

    with col3:
        st.metric("Stored Runs", _count_runs())

    # Recent runs preview
    st.subheader("Recent Runs")
    _show_recent_runs()


def _count_datasets() -> int:
    """Count available datasets."""
    try:
        from backend.evaluation.webapp.utils import get_available_datasets
        return len(get_available_datasets())
    except Exception:
        return 0


def _count_variants() -> int:
    """Count available variants."""
    try:
        from backend.evaluation.webapp.utils import get_available_variants
        return len(get_available_variants())
    except Exception:
        return 0


def _count_runs() -> int:
    """Count stored runs."""
    try:
        from backend.evaluation.webapp.storage import RunStore
        store = RunStore()
        return len(store.list_all_run_ids())
    except Exception:
        return 0


def _show_recent_runs():
    """Show recent evaluation runs."""
    try:
        from backend.evaluation.webapp.storage import RunStore
        from backend.evaluation.webapp.utils import format_duration, format_percentage

        store = RunStore()
        runs = store.list_runs(limit=5)

        if not runs:
            st.info("No evaluation runs yet. Go to **Run** page to start your first evaluation.")
            return

        # Create a table
        data = []
        for run in runs:
            data.append({
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

        st.dataframe(data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading runs: {e}")


if __name__ == "__main__":
    main()
