"""Enterprise Analytics OS — Main navigation app."""
import warnings
warnings.filterwarnings("ignore", message=".*signal only works in main thread.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import APP_NAME, APP_ICON, SCREENS, init_session_defaults

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_defaults()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_NAME}")
    st.divider()

    # LLM Provider
    provider = st.radio("LLM Provider", ["gemini", "openai"], format_func=lambda x: x.upper())
    st.session_state["provider"] = provider

    # API Key
    env_key = os.getenv("GOOGLE_API_KEY", "") if provider == "gemini" else os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        f"{provider.upper()} API Key",
        value=env_key,
        type="password",
    )
    st.session_state["api_key"] = api_key

    if not api_key:
        st.warning("⚠️ Enter an API key to enable AI features.")

    st.divider()

    # Navigation
    st.subheader("Navigation")
    screen = st.radio(
        "Go to",
        options=list(SCREENS.keys()),
        format_func=lambda k: SCREENS[k],
        label_visibility="collapsed",
    )

    # Data status
    st.divider()
    if st.session_state.get("raw_data") is not None:
        df = st.session_state["raw_data"]
        st.success(f"📁 **{st.session_state.get('file_name', 'data')}**\n\n{df.shape[0]} rows × {df.shape[1]} cols")
    else:
        st.info("No data loaded yet.")

# ── Route to screen ──────────────────────────────────────────────────
if screen == "1_import":
    from screens.screen1_import import render
elif screen == "2_profile":
    from screens.screen2_profile import render
elif screen == "3_ba_studio":
    from screens.screen3_ba_studio import render
elif screen == "4_modeling":
    from screens.screen4_modeling import render
elif screen == "5_optimize":
    from screens.screen5_optimize import render
elif screen == "6_reports":
    from screens.screen6_reports import render

render()
