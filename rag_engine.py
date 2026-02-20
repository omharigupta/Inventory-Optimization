import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from file_processor import FileRAGProcessor


class RAGChatEngine:
    """CrewAI-powered chat engine that answers questions over uploaded file data."""

    def __init__(self, processor: FileRAGProcessor, api_key: str, provider: str = "gemini"):
        self.processor = processor
        self.api_key = api_key
        self.provider = provider
        self.chat_history: list[dict] = []

        # Set env var for litellm (used by CrewAI under the hood)
        if provider == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key
            self.llm = LLM(
                model="gemini/gemini-2.5-flash-preview-05-20",
                api_key=api_key,
            )
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = LLM(
                model="openai/gpt-4o",
                api_key=api_key,
            )

    # ── public API ────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a user message and get a response grounded in the Excel data."""

        self.chat_history.append({"role": "user", "content": user_message})

        search_tool, keyword_tool = self._build_search_tools()

        # Build conversation context
        history_text = self._format_history()

        analyst = Agent(
            role="Senior Data Analyst & Reasoning Expert",
            goal=(
                "Answer user questions accurately using uploaded file data. "
                "Search the data thoroughly, analyze what you find, and give a helpful answer. "
                "Use reasoning and logic to interpret the data. "
                "Only say data is missing if your searches truly return nothing relevant at all."
            ),
            backstory=(
                "You are a skilled data analyst. You search the uploaded files, find relevant data, "
                "and provide clear, well-reasoned answers. You are good at interpreting data — "
                "even if the data doesn't directly spell out the answer, you can analyze patterns, "
                "calculate values, and draw logical conclusions from what's available. "
                "You always try your best to answer using whatever data is available. "
                "You only say 'insufficient data' as a last resort when searches return absolutely nothing useful."
            ),
            tools=[search_tool, keyword_tool],
            llm=self.llm,
            verbose=False,
            max_iter=15,
        )

        task = Task(
            description=(
                f"Answer the following question using the uploaded file data.\n\n"
                f"--- Conversation History ---\n{history_text}\n"
                f"--- Current Question ---\n{user_message}\n\n"
                f"HOW TO ANSWER:\n\n"
                f"1. SEARCH the data using both 'Search File Data' and 'Keyword Search' tools. "
                f"Try at least 2-3 different search queries to find relevant information.\n\n"
                f"2. ANALYZE what you found. Even if the data doesn't perfectly match the question, "
                f"use your reasoning skills to interpret it. Calculate totals, find patterns, "
                f"compare values, and draw logical conclusions.\n\n"
                f"3. ANSWER the question based on what you found. Include:\n"
                f"   - A clear, direct answer\n"
                f"   - The actual data values that support your answer\n"
                f"   - Your reasoning/calculations if any\n\n"
                f"4. ONLY if your searches return absolutely NO relevant data at all (not even "
                f"partially related data), then say:\n"
                f"   '⚠️ The uploaded files don't contain data about [topic]. "
                f"Please upload a file with [specific data needed].'\n\n"
                f"KEY RULES:\n"
                f"- ALWAYS try to answer first. Most questions CAN be answered from the data.\n"
                f"- If you find ANY relevant data, use it to form an answer — don't reject it.\n"
                f"- Use reasoning and logic to interpret data, not just exact lookups.\n"
                f"- It's OK to say 'Based on the available data...' if the answer is partial.\n"
                f"- Only say 'insufficient data' when searches genuinely return nothing useful."
            ),
            expected_output=(
                "A helpful, data-backed answer. Use the actual values from the files. "
                "Show reasoning where needed. Only say 'insufficient data' if searches returned nothing."
            ),
            agent=analyst,
        )

        crew = Crew(
            agents=[analyst],
            tasks=[task],
            verbose=False,
        )

        result = crew.kickoff()
        response = str(result)

        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.chat_history.clear()

    # ── internal helpers ──────────────────────────────────────────────

    def _build_search_tools(self):
        processor = self.processor

        @tool("Search File Data")
        def search_file_data(query: str) -> str:
            """Semantic search across all uploaded files. Use this for meaning-based queries
            like 'total revenue', 'employee performance', 'sales trends'. Returns the most
            semantically relevant chunks. Use different phrasings to find more data."""
            results = processor.search(query, n_results=25)
            if not results:
                return "No relevant data found. Try different search terms or the user may need to upload additional files."
            return "\n---\n".join(results)

        @tool("Keyword Search")
        def keyword_search(keyword: str) -> str:
            """Exact keyword search across all uploaded files. Use this to find specific values,
            names, codes, or exact text matches. For example: search for '2024', 'John Smith',
            'Category A', specific product names, or exact numbers."""
            results = processor.keyword_search(keyword, n_results=25)
            if not results:
                return f"No data found containing '{keyword}'. Try a different keyword or check spelling."
            return "\n---\n".join(results)

        return search_file_data, keyword_search

    def _format_history(self) -> str:
        if len(self.chat_history) <= 1:
            return "(no prior conversation)"
        lines = []
        for msg in self.chat_history[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
