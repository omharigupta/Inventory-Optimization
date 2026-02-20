import pandas as pd
import chromadb
from google import genai
import uuid
import openai


class ExcelRAGProcessor:
    """Processes Excel files into chunks, generates embeddings, and stores in ChromaDB."""

    def __init__(self, api_key: str, provider: str = "gemini", chroma_host: str = "localhost", chroma_port: int = 8000):
        self.provider = provider
        if provider == "gemini":
            self.genai_client = genai.Client(api_key=api_key)
        else:
            self.openai_client = openai.OpenAI(api_key=api_key)
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.collection = None

    def process_excel(self, file_path: str, collection_name: str = "excel_data") -> int:
        """Read Excel file, chunk it, embed with Gemini, store in ChromaDB. Returns chunk count."""

        # Reset collection
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Read all sheets
        xls = pd.ExcelFile(file_path)
        all_chunks = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            chunks = self._dataframe_to_chunks(df, sheet_name)
            all_chunks.extend(chunks)

        # Batch embed & store
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]

            embeddings = self._get_embeddings(texts)

            self.collection.add(
                ids=[str(uuid.uuid4()) for _ in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[c["metadata"] for c in batch],
            )

        return len(all_chunks)

    # ── internal helpers ──────────────────────────────────────────────

    def _dataframe_to_chunks(self, df: pd.DataFrame, sheet_name: str) -> list:
        chunks = []
        columns = df.columns.tolist()

        # Sheet-level summary
        summary = (
            f"Sheet '{sheet_name}' contains {len(df)} rows and "
            f"{len(columns)} columns: {', '.join(str(c) for c in columns)}"
        )
        chunks.append(
            {"text": summary, "metadata": {"sheet": sheet_name, "type": "summary"}}
        )

        # Column statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                stats = df[col].describe()
                stats_text = (
                    f"Sheet '{sheet_name}', Column '{col}' statistics: "
                    f"count={stats.get('count', 'N/A')}, "
                    f"mean={stats.get('mean', 'N/A'):.2f}, "
                    f"min={stats.get('min', 'N/A')}, "
                    f"max={stats.get('max', 'N/A')}, "
                    f"std={stats.get('std', 'N/A'):.2f}"
                )
                chunks.append(
                    {
                        "text": stats_text,
                        "metadata": {
                            "sheet": sheet_name,
                            "type": "column_stats",
                            "column": str(col),
                        },
                    }
                )

        # Row-level chunks
        for idx, row in df.iterrows():
            parts = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    parts.append(f"{col} = {val}")
            if parts:
                row_text = f"Sheet '{sheet_name}', Row {idx + 1}: " + ", ".join(parts)
                chunks.append(
                    {
                        "text": row_text,
                        "metadata": {
                            "sheet": sheet_name,
                            "type": "row",
                            "row_index": str(idx),
                        },
                    }
                )

        return chunks

    def _get_embeddings(self, texts: list) -> list:
        if self.provider == "gemini":
            result = self.genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
            return [e.values for e in result.embeddings]
        else:
            result = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [e.embedding for e in result.data]

    def search(self, query: str, n_results: int = 10) -> list[str]:
        """Semantic search against the stored Excel data."""
        if not self.collection:
            return []

        query_embedding = self._get_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []
