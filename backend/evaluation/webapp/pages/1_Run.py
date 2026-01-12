"""
Run Evaluation Page

Allows users to:
- Select dataset and variant
- Configure parameters
- Execute evaluation with progress tracking
- Save results
"""

import streamlit as st
from datetime import datetime, timezone

st.set_page_config(page_title="Run Evaluation", page_icon="▶️", layout="wide")

st.title("▶️ Run Evaluation")


def main():
    """Main page content."""
    # Initialize session state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "progress" not in st.session_state:
        st.session_state.progress = {"completed": 0, "total": 0, "f1": 0.0}

    # Configuration section
    col1, col2 = st.columns(2)

    with col1:
        dataset_name = _select_dataset()

    with col2:
        variant_id = _select_variant()

    # Parameter editor
    params, overrides = _parameter_editor(variant_id)

    # Concurrency setting
    max_concurrency = st.slider(
        "Max Concurrency",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of concurrent LLM calls",
    )

    st.divider()

    # Run button and progress
    if st.session_state.running:
        _show_progress()
    else:
        if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
            _run_evaluation(dataset_name, variant_id, params, overrides, max_concurrency)


def _select_dataset() -> str:
    """Dataset selector."""
    from backend.evaluation.webapp.utils import get_available_datasets

    st.subheader("Dataset")

    datasets = get_available_datasets()
    if not datasets:
        st.error("No datasets found")
        return ""

    options = [d["name"] for d in datasets]
    selected = st.selectbox(
        "Select Dataset",
        options=options,
        help="Choose the evaluation dataset",
    )

    # Show dataset info
    ds_info = next((d for d in datasets if d["name"] == selected), None)
    if ds_info:
        st.caption(f"📊 {ds_info['sample_count']} samples | v{ds_info['version']}")
        if ds_info.get("description"):
            st.caption(ds_info["description"])

    return selected


def _select_variant() -> str:
    """Variant selector."""
    from backend.evaluation.webapp.utils import get_available_variants

    st.subheader("Variant")

    variants = get_available_variants()
    if not variants:
        st.error("No variants found")
        return ""

    options = [v["id"] for v in variants]
    selected = st.selectbox(
        "Select Variant",
        options=options,
        help="Choose the pipeline variant",
    )

    # Show variant info
    v_info = next((v for v in variants if v["id"] == selected), None)
    if v_info:
        pipeline_name = v_info["pipeline"].split(":")[-1] if v_info.get("pipeline") else "Unknown"
        st.caption(f"🔧 Pipeline: {pipeline_name}")

    return selected


def _parameter_editor(variant_id: str) -> tuple[dict, dict]:
    """
    Parameter editor for the selected variant.

    Returns:
        Tuple of (full_params, overrides)
    """
    from backend.evaluation.webapp.utils import get_variant_preset

    st.subheader("Parameters")

    if not variant_id:
        return {}, {}

    try:
        preset = get_variant_preset(variant_id)
        params = preset.get("params", {})
    except Exception as e:
        st.error(f"Error loading preset: {e}")
        return {}, {}

    if not params:
        st.info("This variant has no configurable parameters.")
        return {}, {}

    # Create parameter editors
    overrides = {}

    with st.expander("Configure Parameters", expanded=False):
        for key, default_value in params.items():
            col1, col2 = st.columns([3, 1])

            with col1:
                if isinstance(default_value, bool):
                    new_value = st.checkbox(key, value=default_value)
                elif isinstance(default_value, int):
                    new_value = st.number_input(
                        key,
                        value=default_value,
                        step=1,
                        format="%d",
                    )
                elif isinstance(default_value, float):
                    new_value = st.number_input(
                        key,
                        value=default_value,
                        format="%.2f",
                    )
                elif isinstance(default_value, list):
                    # For lists, show as multiselect if small
                    st.text(f"{key}: {default_value}")
                    new_value = default_value
                else:
                    new_value = st.text_input(key, value=str(default_value))

            with col2:
                if new_value != default_value:
                    st.caption("✏️ Modified")
                    overrides[key] = new_value
                else:
                    st.caption("Default")

    # Show effective parameters
    effective = dict(params)
    effective.update(overrides)

    if overrides:
        st.info(f"📝 {len(overrides)} parameter(s) overridden")

    return effective, overrides


def _show_progress():
    """Show evaluation progress."""
    progress = st.session_state.progress

    st.subheader("Running...")

    # Progress bar
    if progress["total"] > 0:
        pct = progress["completed"] / progress["total"]
        st.progress(pct, text=f"{progress['completed']}/{progress['total']} samples")

        # Current metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Completed", f"{progress['completed']}/{progress['total']}")
        with col2:
            st.metric("Current F1", f"{progress['f1']:.1%}")
    else:
        st.progress(0, text="Starting...")


def _run_evaluation(
    dataset_name: str,
    variant_id: str,
    params: dict,
    overrides: dict,
    max_concurrency: int,
):
    """Execute evaluation and save results."""
    import asyncio
    from backend.evaluation.webapp.utils import (
        run_evaluation_async,
        generate_run_id,
        create_prompt_snapshot,
        get_variant_preset,
    )
    from backend.evaluation.webapp.storage import RunStore, StoredRun, RunConfig

    if not dataset_name or not variant_id:
        st.error("Please select both dataset and variant")
        return

    st.session_state.running = True
    st.session_state.progress = {"completed": 0, "total": 0, "f1": 0.0}

    # Progress placeholder
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    def on_progress(completed: int, total: int, f1: float):
        st.session_state.progress = {"completed": completed, "total": total, "f1": f1}

    try:
        status_placeholder.info("🚀 Starting evaluation...")

        # Run evaluation
        result, duration = asyncio.run(
            run_evaluation_async(
                dataset_name=dataset_name,
                variant_id=variant_id,
                param_overrides=overrides,
                max_concurrency=max_concurrency,
                on_progress=on_progress,
            )
        )

        # Create config snapshot
        preset = get_variant_preset(variant_id)
        prompt_snapshot = create_prompt_snapshot(variant_id)

        config = RunConfig(
            variant_id=variant_id,
            dataset_name=dataset_name,
            preset_params=preset.get("params", {}),
            overrides=overrides,
            prompt_snapshot=prompt_snapshot,
            max_concurrency=max_concurrency,
        )

        # Create stored run
        run_id = generate_run_id()
        stored_run = StoredRun(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            config=config,
            result=result,
        )

        # Save
        store = RunStore()
        store.save(stored_run)

        # Show success
        status_placeholder.success(f"✅ Evaluation complete! Run ID: `{run_id}`")

        # Show results summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precision", f"{result.overall_precision:.1%}")
        with col2:
            st.metric("Recall", f"{result.overall_recall:.1%}")
        with col3:
            st.metric("F1 Score", f"{result.overall_f1:.1%}")
        with col4:
            st.metric("Duration", f"{duration:.1f}s")

        # Link to details
        st.info(f"View details in **History** or **Samples** pages.")

    except Exception as e:
        status_placeholder.error(f"❌ Evaluation failed: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        st.session_state.running = False


if __name__ == "__main__":
    main()
