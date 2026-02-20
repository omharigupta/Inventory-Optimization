"""Screen 1: Import & Intent Mapping — Data ingestion + conversational scope definition."""
import streamlit as st
import pandas as pd
import tempfile
import os
import pdfplumber
from config import get_api_key, get_provider
from agents import parse_business_intent


def get_data_summary(df: pd.DataFrame) -> str:
    """Create a rich text summary of the DataFrame for LLM consumption."""
    lines = []
    lines.append(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"**Columns**: {', '.join(df.columns.tolist())}")

    # Dtypes
    lines.append(f"\n**Data Types**:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype} ({df[col].nunique()} unique, {df[col].isnull().sum()} nulls)")

    # Numeric stats
    numeric = df.select_dtypes(include=["number"])
    if not numeric.empty:
        lines.append(f"\n**Numeric Summary**:")
        for col in numeric.columns:
            s = df[col].describe()
            lines.append(
                f"  - {col}: min={s['min']}, max={s['max']}, "
                f"mean={s['mean']:.2f}, std={s['std']:.2f}"
            )

    # Sample
    lines.append(f"\n**First 3 Rows**:")
    lines.append(df.head(3).to_string(index=False))

    return "\n".join(lines)


def render():
    st.header("📥 Screen 1: Import & Intent Mapping")
    st.caption("Upload your dataset and define your business objectives through conversation.")

    api_key = get_api_key()
    provider = get_provider()

    # ── File Upload ───────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "xls", "pdf"],
            help="CSV, Excel, or PDF files",
        )
    with col2:
        st.metric("Status", "Ready" if st.session_state.get("raw_data") is not None else "No Data")

    if uploaded:
        fname = uploaded.name.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(tmp_path)
            elif fname.endswith((".xlsx", ".xls")):
                # Read all sheet names first
                xls = pd.ExcelFile(tmp_path)
                sheet_names = xls.sheet_names

                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "📑 Select Sheet", sheet_names,
                        help=f"This file has {len(sheet_names)} sheets",
                    )
                else:
                    selected_sheet = sheet_names[0]

                df = pd.read_excel(tmp_path, sheet_name=selected_sheet, header=0)

                # If the result looks like only 1-2 columns, try without header
                if df.shape[1] <= 2 and len(xls.parse(selected_sheet, header=None)) > 0:
                    df_no_header = pd.read_excel(tmp_path, sheet_name=selected_sheet, header=None)
                    if df_no_header.shape[1] > df.shape[1]:
                        df = df_no_header
                        df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]

            elif fname.endswith(".pdf"):
                # Extract tables from PDF
                tables = []
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        for table in page.extract_tables():
                            if table and len(table) > 1:
                                tdf = pd.DataFrame(table[1:], columns=table[0])
                                tables.append(tdf)
                if tables:
                    df = pd.concat(tables, ignore_index=True)
                else:
                    st.warning("No tabular data found in PDF.")
                    df = None
            else:
                df = None

            if df is not None:
                st.session_state["raw_data"] = df
                st.session_state["file_name"] = uploaded.name
                st.session_state["data_summary"] = get_data_summary(df)
                st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except (PermissionError, FileNotFoundError):
                pass

    # ── Data Preview ──────────────────────────────────────────────────
    df = st.session_state.get("raw_data")
    if df is not None:
        with st.expander("📄 Data Preview", expanded=True):
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(f"Showing first 50 of {df.shape[0]} rows × {df.shape[1]} columns")

        with st.expander("📊 Data Summary"):
            st.markdown(st.session_state.get("data_summary", ""))

        # ── Conversational Intent Mapping ─────────────────────────────
        st.divider()
        st.subheader("💬 Define Your Business Intent")

        intent = st.text_area(
            "What do you want to achieve with this data?",
            placeholder='e.g., "Understand retail sales trends" or "Predict customer churn"',
            height=100,
        )

        if st.button("🎯 Parse Intent & Define Scope", type="primary", disabled=not api_key):
            if not intent:
                st.warning("Please enter your business intent.")
            else:
                with st.spinner("AI is analyzing your data and defining project scope..."):
                    result = parse_business_intent(
                        st.session_state["data_summary"], intent, api_key, provider
                    )
                    st.session_state["business_intent"] = intent
                    st.session_state["project_scope"] = result

        if st.session_state.get("project_scope"):
            st.divider()
            st.subheader("📋 Project Scope & Objectives")
            st.markdown(st.session_state["project_scope"])
            st.success("✅ Scope defined — proceed to **Screen 2: Profile & EDA**")
