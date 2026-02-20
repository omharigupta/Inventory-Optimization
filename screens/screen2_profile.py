"""Screen 2: Profile & Data Preparation — EDA, cleaning, diagnostics."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config import get_api_key, get_provider
from agents import run_diagnostic_analysis


def render():
    st.header("🔍 Screen 2: Profile & Data Preparation")
    st.caption("Exploratory Data Analysis, data cleaning, and diagnostic analytics.")

    df = st.session_state.get("raw_data")
    if df is None:
        st.warning("⬅️ Please upload data in **Screen 1** first.")
        return

    api_key = get_api_key()
    provider = get_provider()

    # ── Data Health Overview ──────────────────────────────────────────
    st.subheader("📊 Data Health Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    duplicates = df.duplicated().sum()

    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Completeness", f"{completeness:.1f}%")
    col4.metric("Duplicates", f"{duplicates:,}")

    # ── Missing Values Chart ──────────────────────────────────────────
    with st.expander("🔴 Missing Values Analysis", expanded=True):
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values!")
        else:
            fig = px.bar(
                x=missing.index, y=missing.values,
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Values by Column",
                color=missing.values,
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Column Distribution ───────────────────────────────────────────
    with st.expander("📈 Column Distributions"):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            selected_num = st.selectbox("Numeric Column", numeric_cols)
            fig = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution: {selected_num}")
            st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            selected_cat = st.selectbox("Categorical Column", cat_cols)
            value_counts = df[selected_cat].value_counts().head(20)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                         labels={"x": selected_cat, "y": "Count"},
                         title=f"Top Values: {selected_cat}")
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Matrix ────────────────────────────────────────────
    with st.expander("🔗 Correlation Matrix"):
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation.")

    # ── Data Cleaning ─────────────────────────────────────────────────
    st.divider()
    st.subheader("🧹 Data Cleaning")

    clean_actions = st.multiselect(
        "Select cleaning actions:",
        [
            "Drop duplicate rows",
            "Drop columns with >50% missing",
            "Fill numeric nulls with median",
            "Fill categorical nulls with mode",
            "Remove outliers (IQR method)",
        ],
    )

    if st.button("🧹 Apply Cleaning", type="primary"):
        cleaned = df.copy()
        log = []

        if "Drop duplicate rows" in clean_actions:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            log.append(f"Removed {before - len(cleaned)} duplicate rows")

        if "Drop columns with >50% missing" in clean_actions:
            threshold = len(cleaned) * 0.5
            cols_before = cleaned.shape[1]
            cleaned = cleaned.dropna(axis=1, thresh=int(threshold))
            log.append(f"Dropped {cols_before - cleaned.shape[1]} columns")

        if "Fill numeric nulls with median" in clean_actions:
            num_cols = cleaned.select_dtypes(include=["number"]).columns
            for col in num_cols:
                nulls = cleaned[col].isnull().sum()
                if nulls > 0:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    log.append(f"Filled {nulls} nulls in '{col}' with median")

        if "Fill categorical nulls with mode" in clean_actions:
            cat_cols = cleaned.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                nulls = cleaned[col].isnull().sum()
                if nulls > 0 and not cleaned[col].mode().empty:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
                    log.append(f"Filled {nulls} nulls in '{col}' with mode")

        if "Remove outliers (IQR method)" in clean_actions:
            num_cols = cleaned.select_dtypes(include=["number"]).columns
            before = len(cleaned)
            for col in num_cols:
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                cleaned = cleaned[(cleaned[col] >= Q1 - 1.5 * IQR) & (cleaned[col] <= Q3 + 1.5 * IQR)]
            log.append(f"Removed {before - len(cleaned)} outlier rows")

        st.session_state["clean_data"] = cleaned

        for msg in log:
            st.write(f"✔ {msg}")
        st.success(f"✅ Cleaned data: {cleaned.shape[0]} rows × {cleaned.shape[1]} columns")

    # ── Diagnostic Analytics ──────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Diagnostic Analytics")
    st.caption("Ask WHY something happened in your data")

    diag_question = st.text_input(
        "Diagnostic question:",
        placeholder='e.g., "Why did sales drop in Q4?" or "What correlates with churn?"',
    )

    if st.button("🔬 Run Diagnostic", disabled=not api_key):
        if diag_question:
            with st.spinner("Running diagnostic analysis..."):
                result = run_diagnostic_analysis(
                    st.session_state.get("data_summary", ""), diag_question, api_key, provider
                )
                st.session_state["diagnostic_result"] = result

    if st.session_state.get("diagnostic_result"):
        st.markdown(st.session_state["diagnostic_result"])
