"""Screen 2: Profile & Data Preparation — EDA, cleaning, diagnostics, chat."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config import get_api_key, get_provider
from agents import run_diagnostic_analysis


def _build_chat_engine(df, api_key, provider):
    """Build a lightweight chat engine that answers questions from the DataFrame directly."""
    from agents import run_agent

    def chat_with_data(question: str) -> str:
        # Build a rich data context for the LLM
        summary_lines = []
        summary_lines.append(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        summary_lines.append(f"Columns: {', '.join(df.columns.tolist())}")
        summary_lines.append(f"\nData Types:")
        for col in df.columns:
            summary_lines.append(f"  {col}: {df[col].dtype} ({df[col].nunique()} unique)")

        # Numeric stats
        numeric = df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary_lines.append(f"\nNumeric Statistics:")
            summary_lines.append(numeric.describe().to_string())

        # Categorical value counts
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols[:5]:
            vc = df[col].value_counts().head(10)
            summary_lines.append(f"\nTop values for '{col}':")
            for val, cnt in vc.items():
                summary_lines.append(f"  {val}: {cnt}")

        # Sample rows
        summary_lines.append(f"\nFirst 10 rows:")
        summary_lines.append(df.head(10).to_string(index=False))

        # Full data for small datasets, tail for larger
        if df.shape[0] <= 100:
            summary_lines.append(f"\nFull data ({df.shape[0]} rows):")
            summary_lines.append(df.to_string(index=False))
        else:
            summary_lines.append(f"\nLast 10 rows:")
            summary_lines.append(df.tail(10).to_string(index=False))

        data_context = "\n".join(summary_lines)

        return run_agent(
            role="Data Analyst & Question Answerer",
            goal="Answer questions accurately based ONLY on the provided dataset. If the data doesn't contain the information needed, clearly say what data is missing.",
            backstory=(
                "You are a precise data analyst. You answer questions using ONLY the data provided. "
                "You can calculate totals, averages, find maximums, compare values, identify trends. "
                "If the question asks about data that doesn't exist in the dataset, you clearly state: "
                "'⚠️ This data is not available in the uploaded file. To answer this question, "
                "I would need [specific data/columns needed]. Please upload a file containing that data.' "
                "You NEVER make up data or guess values not in the dataset."
            ),
            task_description=(
                f"DATASET:\n{data_context}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Look through the data carefully for information relevant to the question.\n"
                f"2. If the data CONTAINS relevant information → Answer with specific values, "
                f"calculations, etc. Show your work.\n"
                f"3. If the data PARTIALLY relates → Answer what you can and note what's missing.\n"
                f"4. If the data has NO relevant information at all → Respond with:\n"
                f"   '⚠️ The uploaded file does not contain data about [topic]. "
                f"To answer this, I would need a file with [specific columns/data needed]. "
                f"Please upload additional data.'\n"
                f"5. NEVER invent or hallucinate data values."
            ),
            expected_output="A data-backed answer with specific values, or a clear message about what data is missing.",
            api_key=api_key,
            provider=provider,
        )

    return chat_with_data


def render():
    st.header("🔍 Screen 2: Profile & Data Preparation")
    st.caption("Exploratory Data Analysis, data cleaning, and diagnostic analytics.")

    df = st.session_state.get("raw_data")
    if df is None:
        st.warning("⬅️ Please upload data in **Screen 1** first.")
        return

    api_key = get_api_key()
    provider = get_provider()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_eda, tab_clean, tab_chat = st.tabs(["📊 EDA & Diagnostics", "🧹 Data Cleaning", "💬 Chat with Data"])

    # ══════════════════════════════════════════════════════════════════
    #  TAB 1 — EDA & Diagnostics
    # ══════════════════════════════════════════════════════════════════
    with tab_eda:
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

        # Missing Values Chart
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

        # Column Distribution
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

        # Correlation Matrix
        with st.expander("🔗 Correlation Matrix"):
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation.")

        # Diagnostic Analytics
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

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2 — Data Cleaning
    # ══════════════════════════════════════════════════════════════════
    with tab_clean:
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

    # ══════════════════════════════════════════════════════════════════
    #  TAB 3 — Chat with Data
    # ══════════════════════════════════════════════════════════════════
    with tab_chat:
        st.subheader("💬 Chat with Your Data")
        st.caption(
            "Ask questions about your data. The AI will answer using actual values "
            "from your file. If the data doesn't contain the answer, it will tell you "
            "what additional data is needed."
        )

        if not api_key:
            st.warning("⚠️ Enter an API key in the sidebar to enable chat.")
        else:
            # Initialize chat history
            if "chat_messages" not in st.session_state:
                st.session_state["chat_messages"] = []

            # Display chat history
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            if user_q := st.chat_input("Ask about your data... e.g. 'What are the total sales?'"):
                # Show user message
                st.session_state["chat_messages"].append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your data..."):
                        chat_fn = _build_chat_engine(df, api_key, provider)
                        response = chat_fn(user_q)
                    st.markdown(response)

                st.session_state["chat_messages"].append({"role": "assistant", "content": response})