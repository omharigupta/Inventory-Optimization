# Excel RAG Chat

Chat with your Excel data using **Google Gemini** LLM + embeddings and **CrewAI** agents.

## Architecture

```
Excel File  →  pandas  →  Chunks  →  Gemini Embeddings  →  ChromaDB
                                                              ↓
User Question  →  CrewAI Agent  →  Semantic Search  →  Gemini LLM  →  Answer
```

| Component          | Technology               |
|--------------------|--------------------------|
| LLM                | Google Gemini 2.0 Flash  |
| Embeddings         | Gemini Embedding 001     |
| Vector Store       | ChromaDB (in-memory)     |
| Agent Framework    | CrewAI                   |
| UI                 | Streamlit                |
| Excel Parsing      | pandas + openpyxl        |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Either create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your Google API key
```

Or enter the key directly in the sidebar when the app starts.

**Get your key:** https://aistudio.google.com/apikey

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Enter your **Google API Key** in the sidebar
2. Upload an **Excel file** (.xlsx or .xls)
3. Click **Process File** — the data is chunked, embedded, and indexed
4. Start chatting! Ask questions like:
   - "What are the total sales for Q3?"
   - "Which employee has the highest salary?"
   - "Summarize the data in Sheet1"
   - "Compare revenue across all regions"

## Project Structure

```
project/
├── app.py               # Streamlit UI (chat interface + file upload)
├── rag_engine.py         # CrewAI agent setup with RAG tool
├── excel_processor.py    # Excel → chunks → Gemini embeddings → ChromaDB
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

## How It Works

1. **Excel Processing** — pandas reads all sheets; each row becomes a text chunk, plus sheet summaries and column statistics
2. **Embedding** — chunks are embedded using `gemini-embedding-001` via the Google GenAI SDK
3. **Indexing** — embeddings are stored in an in-memory ChromaDB collection
4. **Query** — when you ask a question, a CrewAI agent uses a custom search tool to retrieve the most relevant chunks via cosine similarity
5. **Answer** — Gemini 2.0 Flash generates a grounded answer from the retrieved context
6. **Conversation** — chat history is maintained so follow-up questions work naturally
