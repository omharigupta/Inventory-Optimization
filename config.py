"""Shared configuration and state management for Enterprise Analytics OS."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Enterprise Analytics OS"
APP_ICON = "🏢"

SCREENS = {
    "1_import": "📥 Import & Intent",
    "2_profile": "🔍 Profile & EDA",
    "3_ba_studio": "📋 BA Studio",
    "4_modeling": "🧠 Modeling Suite",
    "5_optimize": "⚡ Optimization",
    "6_reports": "📊 Visualization & Reports",
}


def init_session_defaults():
    """Initialize all session state defaults."""
    defaults = {
        # Global
        "provider": "gemini",
        "api_key": "",
        # Screen 1 — Import
        "raw_data": None,
        "file_name": "",
        "file_type": "",
        "business_intent": "",
        "project_scope": "",
        "data_summary": "",
        # Screen 2 — EDA
        "clean_data": None,
        "eda_report": "",
        "diagnostic_result": "",
        # Screen 3 — BA Studio
        "brd_doc": "",
        "frd_doc": "",
        "srs_doc": "",
        "effort_estimate": "",
        "project_plan": "",
        # Screen 4 — Modeling
        "model_type": "classification",
        "target_column": "",
        "model_results": None,
        "predictions": None,
        # Screen 5 — Optimization
        "optimization_result": "",
        "recommendations": "",
        # Screen 6 — Reports
        "charts": [],
        "executive_summary": "",
        # Chat
        "chat_messages": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_api_key():
    return st.session_state.get("api_key", os.getenv("GOOGLE_API_KEY", ""))


def get_provider():
    return st.session_state.get("provider", "gemini")
