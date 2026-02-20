"""Screen 5: Optimization & Prescriptive Insights — Goal seek, recommendations."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import get_api_key, get_provider
from agents import run_prescriptive_analysis


def render():
    st.header("⚡ Screen 5: Optimization & Prescriptive Insights")
    st.caption("Goal-seek analysis, scenario simulation, and actionable recommendations.")

    api_key = get_api_key()
    provider = get_provider()

    df = st.session_state.get("clean_data") or st.session_state.get("raw_data")
    if df is None:
        st.warning("⬅️ Please upload data first.")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # ── Goal Seek ─────────────────────────────────────────────────────
    st.subheader("🎯 Goal Seek Analysis")

    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Target Metric", numeric_cols)
        with col2:
            current_val = df[target_col].mean()
            st.metric("Current Average", f"{current_val:,.2f}")
        with col3:
            goal_val = st.number_input("Goal Value", value=float(current_val * 1.1), step=0.01)

        change_pct = ((goal_val - current_val) / current_val * 100) if current_val != 0 else 0
        st.info(f"📊 To reach the goal, **{target_col}** needs to change by **{change_pct:+.1f}%**")

        # Sensitivity analysis
        if len(numeric_cols) > 1:
            st.subheader("📐 Sensitivity Analysis")
            other_cols = [c for c in numeric_cols if c != target_col]

            correlations = []
            for col in other_cols:
                corr = df[target_col].corr(df[col])
                if not np.isnan(corr):
                    correlations.append({"Variable": col, "Correlation": corr, "Impact": abs(corr)})

            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values("Impact", ascending=False)
                fig = px.bar(
                    corr_df.head(10), x="Variable", y="Correlation",
                    color="Correlation", color_continuous_scale="RdYlGn",
                    title=f"Variables Most Correlated with '{target_col}'",
                )
                st.plotly_chart(fig, use_container_width=True)

                # What-if simulation
                st.subheader("🔮 What-If Simulation")
                top_driver = corr_df.iloc[0]["Variable"]
                driver_current = df[top_driver].mean()

                sim_change = st.slider(
                    f"Adjust '{top_driver}' by %", -50, 50, 10
                )
                new_driver = driver_current * (1 + sim_change / 100)
                estimated_impact = corr_df.iloc[0]["Correlation"] * (sim_change / 100) * current_val

                sim_cols = st.columns(3)
                sim_cols[0].metric(f"{top_driver} (Current)", f"{driver_current:,.2f}")
                sim_cols[1].metric(f"{top_driver} (Simulated)", f"{new_driver:,.2f}", f"{sim_change:+}%")
                sim_cols[2].metric(f"Estimated {target_col} Impact", f"{estimated_impact:+,.2f}")

    # ── AI Prescriptive Recommendations ───────────────────────────────
    st.divider()
    st.subheader("🤖 AI Prescriptive Recommendations")

    goal_desc = st.text_area(
        "Describe your business goal:",
        placeholder='e.g., "Adjust pricing to increase margins by 15%" or "Reduce customer churn by 20%"',
        height=100,
    )

    if st.button("⚡ Generate Recommendations", type="primary", disabled=not api_key):
        if goal_desc:
            with st.spinner("AI is running scenario simulations..."):
                result = run_prescriptive_analysis(
                    st.session_state.get("data_summary", ""),
                    goal_desc, api_key, provider,
                )
                st.session_state["recommendations"] = result

    if st.session_state.get("recommendations"):
        st.markdown(st.session_state["recommendations"])
