"""Screen 3: BA Studio & Project Tracking — Auto-generate BRD, FRD, SRS + Gantt."""
import streamlit as st
from config import get_api_key, get_provider
from agents import generate_ba_document, generate_effort_estimate


def render():
    st.header("📋 Screen 3: BA Studio & Project Tracking")
    st.caption("Auto-generate project documents and track effort estimates.")

    api_key = get_api_key()
    provider = get_provider()
    project_scope = st.session_state.get("project_scope", "")
    data_summary = st.session_state.get("data_summary", "")

    if not project_scope:
        st.warning("⬅️ Please define project scope in **Screen 1** first.")
        return

    # ── Document Generation ───────────────────────────────────────────
    st.subheader("📝 Auto-Generate Documents")

    doc_cols = st.columns(3)
    with doc_cols[0]:
        if st.button("📄 Generate BRD", use_container_width=True, type="primary"):
            with st.spinner("Generating BRD..."):
                st.session_state["brd_doc"] = generate_ba_document(
                    "BRD", project_scope, data_summary, api_key, provider
                )

    with doc_cols[1]:
        if st.button("📄 Generate FRD", use_container_width=True, type="primary"):
            with st.spinner("Generating FRD..."):
                st.session_state["frd_doc"] = generate_ba_document(
                    "FRD", project_scope, data_summary, api_key, provider
                )

    with doc_cols[2]:
        if st.button("📄 Generate SRS", use_container_width=True, type="primary"):
            with st.spinner("Generating SRS..."):
                st.session_state["srs_doc"] = generate_ba_document(
                    "SRS", project_scope, data_summary, api_key, provider
                )

    # Display generated documents
    for doc_key, doc_name in [("brd_doc", "BRD"), ("frd_doc", "FRD"), ("srs_doc", "SRS")]:
        doc = st.session_state.get(doc_key, "")
        if doc:
            with st.expander(f"📄 {doc_name} Document", expanded=False):
                st.markdown(doc)
                st.download_button(
                    f"⬇ Download {doc_name}",
                    data=doc,
                    file_name=f"{doc_name.lower()}_document.md",
                    mime="text/markdown",
                )

    # ── Effort Estimation ─────────────────────────────────────────────
    st.divider()
    st.subheader("⏱ AI-Powered Effort Estimation")

    if st.button("🤖 Generate Effort Estimate", type="primary"):
        with st.spinner("Estimating project effort..."):
            st.session_state["effort_estimate"] = generate_effort_estimate(
                project_scope, api_key, provider
            )

    if st.session_state.get("effort_estimate"):
        st.markdown(st.session_state["effort_estimate"])

    # ── Simple Gantt Chart ────────────────────────────────────────────
    st.divider()
    st.subheader("📅 Project Timeline")

    if st.session_state.get("effort_estimate"):
        # Auto Gantt from common project phases
        import plotly.figure_factory as ff
        from datetime import datetime, timedelta

        today = datetime.now()
        phases = [
            {"Task": "Data Collection & Import", "Start": today, "Finish": today + timedelta(weeks=1)},
            {"Task": "EDA & Data Preparation", "Start": today + timedelta(weeks=1), "Finish": today + timedelta(weeks=2)},
            {"Task": "Requirements & Documentation", "Start": today + timedelta(weeks=1), "Finish": today + timedelta(weeks=3)},
            {"Task": "Model Development", "Start": today + timedelta(weeks=3), "Finish": today + timedelta(weeks=5)},
            {"Task": "Optimization & Testing", "Start": today + timedelta(weeks=5), "Finish": today + timedelta(weeks=7)},
            {"Task": "Reporting & Delivery", "Start": today + timedelta(weeks=7), "Finish": today + timedelta(weeks=8)},
        ]

        fig = ff.create_gantt(
            phases,
            index_col="Task",
            show_colorbar=True,
            title="Project Gantt Chart",
            showgrid_x=True,
            showgrid_y=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Generate an effort estimate to see the Gantt chart.")
