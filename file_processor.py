import pandas as pd
import chromadb
from google import genai
import uuid
import openai
import csv
import pdfplumber


class FileRAGProcessor:
    """Processes Excel, CSV, and PDF files into chunks, generates embeddings, and stores in ChromaDB."""

    def __init__(self, api_key: str, provider: str = "gemini", chroma_host: str = "localhost", chroma_port: int = 8000):
        self.provider = provider
        if provider == "gemini":
            self.genai_client = genai.Client(api_key=api_key)
        else:
            self.openai_client = openai.OpenAI(api_key=api_key)
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.collection = None

    # ── public API ────────────────────────────────────────────────────

    def delete_all_data(self, collection_name: str = "file_data"):
        """Delete all data from ChromaDB and reset state."""
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = None

    def process_file(self, file_path: str, file_type: str, collection_name: str = "file_data", reset: bool = True) -> int:
        """Route to the correct processor based on file type. Returns chunk count.
        Set reset=False to add data to existing collection (multi-file support)."""
        if reset:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if file_type == "excel":
            all_chunks = self._chunks_from_excel(file_path)
        elif file_type == "csv":
            all_chunks = self._chunks_from_csv(file_path)
        elif file_type == "pdf":
            all_chunks = self._chunks_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Batch embed & store
        batch_size = 100
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

    def search(self, query: str, n_results: int = 25) -> list[str]:
        """Semantic search against the stored data."""
        if not self.collection:
            return []
        query_embedding = self._get_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []

    def keyword_search(self, keyword: str, n_results: int = 25) -> list[str]:
        """Keyword-based search using ChromaDB's where_document filter."""
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[keyword],
                n_results=n_results,
                where_document={"$contains": keyword},
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            # Fallback to semantic if keyword filter fails
            return self.search(keyword, n_results)

    # ── Excel processing ──────────────────────────────────────────────

    def _chunks_from_excel(self, file_path: str) -> list:
        xls = pd.ExcelFile(file_path)
        all_chunks = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            all_chunks.extend(self._dataframe_to_chunks(df, source=f"Sheet '{sheet_name}'"))
        return all_chunks

    # ── CSV processing ────────────────────────────────────────────────

    def _chunks_from_csv(self, file_path: str) -> list:
        df = pd.read_csv(file_path)
        return self._dataframe_to_chunks(df, source="CSV")

    # ── PDF processing ────────────────────────────────────────────────

    def _chunks_from_pdf(self, file_path: str) -> list:
        chunks = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            chunks.append({
                "text": f"PDF document with {total_pages} pages.",
                "metadata": {"source": "PDF", "type": "summary"},
            })

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    # Split long pages into ~500 char paragraphs
                    paragraphs = self._split_text(text, max_chars=500)
                    for para_idx, para in enumerate(paragraphs):
                        chunks.append({
                            "text": f"Page {page_num}, Section {para_idx + 1}: {para}",
                            "metadata": {
                                "source": "PDF",
                                "type": "text",
                                "page": str(page_num),
                            },
                        })

                # Extract tables from PDF pages
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    headers = [str(h) if h else f"Col{i}" for i, h in enumerate(table[0])]
                    for row_idx, row in enumerate(table[1:], start=1):
                        parts = []
                        for h, val in zip(headers, row):
                            if val and str(val).strip():
                                parts.append(f"{h} = {val}")
                        if parts:
                            row_text = (
                                f"Page {page_num}, Table {table_idx + 1}, Row {row_idx}: "
                                + ", ".join(parts)
                            )
                            chunks.append({
                                "text": row_text,
                                "metadata": {
                                    "source": "PDF",
                                    "type": "table_row",
                                    "page": str(page_num),
                                    "table": str(table_idx),
                                },
                            })
        return chunks

    # ── shared helpers ────────────────────────────────────────────────

    def _dataframe_to_chunks(self, df: pd.DataFrame, source: str) -> list:
        chunks = []
        columns = df.columns.tolist()

        # Summary
        summary = (
            f"{source} contains {len(df)} rows and "
            f"{len(columns)} columns: {', '.join(str(c) for c in columns)}"
        )
        chunks.append({"text": summary, "metadata": {"source": source, "type": "summary"}})

        # Column statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            stats = df[col].describe()
            stats_text = (
                f"{source}, Column '{col}' statistics: "
                f"count={stats.get('count', 'N/A')}, "
                f"mean={stats.get('mean', 'N/A'):.2f}, "
                f"min={stats.get('min', 'N/A')}, "
                f"max={stats.get('max', 'N/A')}, "
                f"std={stats.get('std', 'N/A'):.2f}, "
                f"sum={df[col].sum()}"
            )
            chunks.append({
                "text": stats_text,
                "metadata": {"source": source, "type": "column_stats", "column": str(col)},
            })

        # Unique values per column (for categorical discovery)
        for col in columns:
            unique_vals = df[col].dropna().unique()
            if 1 < len(unique_vals) <= 50:  # Only for columns with reasonable cardinality
                vals_str = ", ".join(str(v) for v in unique_vals[:50])
                chunks.append({
                    "text": f"{source}, Column '{col}' has {len(unique_vals)} unique values: {vals_str}",
                    "metadata": {"source": source, "type": "unique_values", "column": str(col)},
                })

        # Sample data: first 3 rows as example
        sample_rows = []
        for idx, row in df.head(3).iterrows():
            parts = [f"{col} = {row[col]}" for col in columns if pd.notna(row[col])]
            if parts:
                sample_rows.append(f"Row {idx + 1}: " + ", ".join(parts))
        if sample_rows:
            chunks.append({
                "text": f"{source}, Sample data (first 3 rows):\n" + "\n".join(sample_rows),
                "metadata": {"source": source, "type": "sample"},
            })

        # Row-level chunks — group every 3 rows together for better precision
        rows_per_chunk = 3
        row_buffer = []
        for idx, row in df.iterrows():
            parts = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    parts.append(f"{col} = {val}")
            if parts:
                row_buffer.append(f"Row {idx + 1}: " + ", ".join(parts))

            if len(row_buffer) >= rows_per_chunk:
                chunk_text = f"{source}:\n" + "\n".join(row_buffer)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"source": source, "type": "rows", "row_index": str(idx)},
                })
                row_buffer = []

        # Remaining rows
        if row_buffer:
            chunk_text = f"{source}:\n" + "\n".join(row_buffer)
            chunks.append({
                "text": chunk_text,
                "metadata": {"source": source, "type": "rows", "row_index": str(idx)},
            })

        return chunks

    def _split_text(self, text: str, max_chars: int = 500) -> list[str]:
        """Split text into chunks, trying to break at paragraph/sentence boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_chars:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                # If a single paragraph exceeds max_chars, split by sentences
                if len(para) > max_chars:
                    sentences = para.replace(". ", ".\n").split("\n")
                    sub = ""
                    for s in sentences:
                        if len(sub) + len(s) + 1 <= max_chars:
                            sub = f"{sub} {s}" if sub else s
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = s
                    current = sub
                else:
                    current = para
        if current:
            chunks.append(current)
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
