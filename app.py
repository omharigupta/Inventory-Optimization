import streamlit as st
import tempfile
import os
import warnings
import signal
from dotenv import load_dotenv
from file_processor import FileRAGProcessor
from rag_engine import RAGChatEngine

# Suppress CrewAI signal handler warnings in Streamlit threads
warnings.filterwarnings("ignore", message=".*signal only works in main thread.*")

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="📊", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; }
    [data-testid="stSidebar"] { background-color: #f8f9fb; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-ready { background: #d4edda; color: #155724; }
    .status-empty { background: #fff3cd; color: #856404; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────
for key, default in {
    "messages": [],
    "file_processed": False,
    "file_names": [],
    "num_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.radio(
        "LLM Provider",
        ["Google Gemini", "OpenAI"],
        horizontal=True,
        help="Choose which AI provider to use for embeddings and chat",
    )
    provider_key = "gemini" if provider == "Google Gemini" else "openai"

    if provider_key == "gemini":
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Get your key from https://aistudio.google.com/apikey",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Get your key from https://platform.openai.com/api-keys",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    st.header("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["xlsx", "xls", "csv", "pdf"],
        accept_multiple_files=True,
        help="Supports Excel (.xlsx, .xls), CSV (.csv), and PDF (.pdf). Upload multiple files!",
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            total_chunks = 0
            file_names = []
            first_file = True

            progress = st.progress(0, text="Processing files…")
            for file_idx, uploaded_file in enumerate(uploaded_files):
                fname = uploaded_file.name.lower()
                if fname.endswith((".xlsx", ".xls")):
                    file_type, suffix = "excel", ".xlsx"
                elif fname.endswith(".csv"):
                    file_type, suffix = "csv", ".csv"
                elif fname.endswith(".pdf"):
                    file_type, suffix = "pdf", ".pdf"
                else:
                    file_type, suffix = "excel", ".xlsx"

                progress.progress(
                    (file_idx) / len(uploaded_files),
                    text=f"Processing {uploaded_file.name}…",
                )

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    if first_file:
                        processor = FileRAGProcessor(api_key, provider=provider_key)
                        num_chunks = processor.process_file(
                            tmp_path, file_type=file_type, reset=True
                        )
                        first_file = False
                    else:
                        num_chunks = processor.process_file(
                            tmp_path, file_type=file_type, reset=False
                        )

                    total_chunks += num_chunks
                    file_names.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass

            progress.progress(1.0, text="Done!")

            if file_names:
                st.session_state["processor"] = processor
                st.session_state["engine"] = RAGChatEngine(
                    processor, api_key, provider=provider_key
                )
                st.session_state["messages"] = []
                st.session_state["file_processed"] = True
                st.session_state["file_names"] = file_names
                st.session_state["num_chunks"] = total_chunks

                st.success(
                    f"✅ Processed **{len(file_names)} file(s)** → **{total_chunks}** data chunks!"
                )

    elif uploaded_files and not api_key:
        st.warning("Enter your API Key above first.")

    # Status panel
    if st.session_state["file_processed"]:
        st.divider()
        files_list = ", ".join(st.session_state["file_names"])
        st.markdown(
            f"📄 **Files:** {files_list}  \n"
            f"📦 **Chunks:** {st.session_state['num_chunks']}  \n"
            f'<span class="status-badge status-ready">● Ready</span>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                if "engine" in st.session_state:
                    st.session_state["engine"].clear_history()
                st.rerun()
        with col2:
            if st.button("🔄 New File", use_container_width=True):
                for k in [
                    "processor",
                    "engine",
                    "messages",
                    "file_processed",
                    "file_names",
                    "num_chunks",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

# ── Main area ─────────────────────────────────────────────────────────

# Delete All Data button at the top
if st.session_state["file_processed"]:
    col_title, col_delete = st.columns([4, 1])
    with col_title:
        st.title("📊 RAG Chat")
    with col_delete:
        st.write("")  # spacing
        if st.button("🗑️ Delete All Data", type="secondary", use_container_width=True):
            # Wipe ChromaDB
            if "processor" in st.session_state:
                try:
                    st.session_state["processor"].delete_all_data()
                except Exception:
                    pass
            # Reset all session state
            for k in [
                "processor",
                "engine",
                "messages",
                "file_processed",
                "file_names",
                "num_chunks",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("All data deleted from ChromaDB. Upload new files to start fresh.")
            st.rerun()
else:
    st.title("📊 RAG Chat")
st.caption(
    "Upload an Excel, CSV, or PDF file → process it → ask questions about your data  \n"
    "Powered by **Google Gemini** or **OpenAI** + **CrewAI**"
)

if not st.session_state["file_processed"]:
    st.info(
        "👈 **Get started:** Pick a provider, enter your API key, upload a file "
        "(Excel, CSV, or PDF), and click **Process File** in the sidebar."
    )
    st.stop()

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data…"):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data…"):
            try:
                engine: RAGChatEngine = st.session_state["engine"]
                response = engine.chat(prompt)
                st.markdown(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"Sorry, something went wrong: {e}"
                st.error(error_msg)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )
