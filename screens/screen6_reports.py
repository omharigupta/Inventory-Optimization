"""Screen 6: Visualization & Reporting — NL chart builder, PDF export."""
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from config import get_api_key, get_provider
from agents import generate_executive_summary


def render():
    st.header("📊 Screen 6: Visualization & Reporting")
    st.caption("Natural language chart builder, executive summaries, and PDF reports.")

    api_key = get_api_key()
    provider = get_provider()

    df = st.session_state.get("clean_data") or st.session_state.get("raw_data")
    if df is None:
        st.warning("⬅️ Please upload data first.")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ── Quick Chart Builder ───────────────────────────────────────────
    st.subheader("📈 Quick Chart Builder")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap"])

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X axis", all_cols)
    with c2:
        y_col = st.selectbox("Y axis", numeric_cols if numeric_cols else all_cols)

    color_col = st.selectbox("Color (optional)", ["None"] + all_cols)
    color = None if color_col == "None" else color_col

    if st.button("📊 Generate Chart"):
        try:
            fig = None
            if chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color, title=f"{y_col} by {x_col}")
            elif chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=color, title=f"{y_col} over {x_col}")
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color, title=f"{y_col} vs {x_col}")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color, title=f"Distribution of {x_col}")
            elif chart_type == "Box":
                fig = px.box(df, x=color, y=y_col, title=f"{y_col} Distribution")
            elif chart_type == "Pie":
                pie_data = df[x_col].value_counts().head(15).reset_index()
                pie_data.columns = [x_col, "count"]
                fig = px.pie(pie_data, names=x_col, values="count", title=f"{x_col} Distribution")
            elif chart_type == "Heatmap":
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                                color_continuous_scale="RdBu_r")

            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["last_chart"] = fig
        except Exception as e:
            st.error(f"Chart error: {e}")

    # ── AI Executive Summary ──────────────────────────────────────────
    st.divider()
    st.subheader("📝 AI Executive Summary")

    audience = st.selectbox("Target Audience", ["C-Suite / Board", "Technical Team", "Business Stakeholders", "General"])

    if st.button("🤖 Generate Executive Summary", type="primary", disabled=not api_key):
        with st.spinner("Compiling executive summary..."):
            summary_text = st.session_state.get("data_summary", "")
            model_res = st.session_state.get("model_results", "No models trained yet.")
            recs = st.session_state.get("recommendations", "No recommendations yet.")

            context = f"""DATA SUMMARY:\n{summary_text}\n\nMODEL RESULTS:\n{model_res}\n\nRECOMMENDATIONS:\n{recs}"""
            result = generate_executive_summary(context, audience, api_key, provider)
            st.session_state["executive_summary"] = result

    if st.session_state.get("executive_summary"):
        st.markdown(st.session_state["executive_summary"])

    # ── PDF Report Download ───────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export Report")

    if st.button("📄 Download PDF Report"):
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 18)
            pdf.cell(0, 12, "Enterprise Analytics Report", ln=True, align="C")
            pdf.ln(8)

            sections = [
                ("Data Summary", st.session_state.get("data_summary", "N/A")),
                ("Model Results", str(st.session_state.get("model_results", "N/A"))),
                ("Recommendations", st.session_state.get("recommendations", "N/A")),
                ("Executive Summary", st.session_state.get("executive_summary", "N/A")),
            ]

            for title, content in sections:
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, title, ln=True)
                pdf.set_font("Helvetica", "", 10)
                # Clean markdown-ish content for PDF
                clean = content.replace("**", "").replace("###", "").replace("##", "").replace("#", "")
                for line in clean.split("\n"):
                    line = line.strip()
                    if line:
                        pdf.multi_cell(0, 6, line)
                pdf.ln(4)

            buf = io.BytesIO()
            pdf.output(buf)
            buf.seek(0)
            st.download_button("⬇️ Download PDF", buf, file_name="analytics_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation error: {e}")
