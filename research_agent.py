from langchain_text_splitters import RecursiveCharacterTextSplitter
# [FIXED] Chuyển Vector Store từ FAISS (RAM) sang ChromaDB PersistentClient (ổ đĩa)
import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
except ImportError:  # môi trường chưa cài langchain-chroma
    from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
import re
import os
import time
import hashlib
from operator import itemgetter

# Import the tools this agent will use
from tools import get_stock_info, scrape_website

# [FIXED] Hằng số cấu hình chunking theo TOKEN của all-MiniLM-L6-v2 (giới hạn 256 token)
EMBEDDING_MAX_TOKENS = 256
CHUNK_SIZE_TOKENS = 220       # chừa biên an toàn cho 2 token đặc biệt [CLS]/[SEP]
CHUNK_OVERLAP_TOKENS = 30
RETRIEVER_K = 4
VECTOR_TTL_SECONDS = 3600     # [FIXED] TTL cache vector = 1 giờ

class ResearchAgent:
    def __init__(self, llm, embeddings_model, search_tool, persist_dir=None):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.search_tool = search_tool
        self.cache = {}

        # [FIXED] Đếm token bằng tokenizer của chính model embedding, rồi cấu hình splitter theo token
        self._token_counter = self._build_token_counter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=self._count_tokens,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # [FIXED] ChromaDB lưu xuống đĩa cục bộ (nên đặt ở đường dẫn KHÔNG có ký tự tiếng Việt)
        self.persist_dir = (
            persist_dir
            or os.environ.get("CHROMA_PERSIST_DIR")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma_cache")
        )
        os.makedirs(self.persist_dir, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    # [FIXED] Token counter dựa trên tokenizer của model embedding
    # ------------------------------------------------------------------
    def _build_token_counter(self):
        client = getattr(self.embeddings_model, "_client", None) or getattr(self.embeddings_model, "client", None)
        tokenizer = getattr(client, "tokenizer", None)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                tokenizer = None

        if tokenizer is not None:
            def count(text):
                return len(tokenizer.encode(text, add_special_tokens=False))
            return count

        print("--- Warning: tokenizer unavailable, estimating tokens as len(text)/4. ---")
        return lambda text: max(1, len(text) // 4)

    def _count_tokens(self, text):
        return self._token_counter(text)

    # ------------------------------------------------------------------
    # [FIXED] Quản lý ChromaDB + TTL
    # ------------------------------------------------------------------
    def _collection_name(self, entity_name, ticker):
        # Gắn tên model vào khóa để đổi model embedding (khác số chiều) không đụng collection cũ
        model = getattr(self.embeddings_model, "model_name", "unknown-model")
        raw = f"{str(entity_name).strip().lower()}|{str(ticker).strip().upper()}|{model}"
        return "ctx_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:24]

    def _drop_collection(self, name):
        try:
            self._chroma_client.delete_collection(name)
        except Exception:
            pass

    def clear_cache(self):
        """Clears the agent's in-memory cache AND the persisted Chroma collections."""
        print("Cache cleared.")
        self.cache = {}
        try:
            for c in self._chroma_client.list_collections():
                name = getattr(c, "name", c)
                if str(name).startswith("ctx_"):
                    self._drop_collection(str(name))
        except Exception as e:
            print(f"Could not clear persisted vector cache: {e}")

    def _load_cached_vector_store(self, entity_name, ticker):
        """Trả về Chroma store nếu còn hạn TTL trên đĩa, ngược lại None (và xóa bản hết hạn)."""
        name = self._collection_name(entity_name, ticker)
        try:
            collection = self._chroma_client.get_collection(name)
        except Exception:
            return None

        created_at = (collection.metadata or {}).get("created_at")
        try:
            fresh = created_at is not None and (time.time() - float(created_at)) <= VECTOR_TTL_SECONDS
            fresh = fresh and collection.count() > 0
        except Exception:
            fresh = False

        if not fresh:
            print("--- Cached vectors expired or empty, rebuilding. ---")
            self._drop_collection(name)
            return None

        print("--- Reusing persisted vector store from disk (TTL still valid). ---")
        return Chroma(
            client=self._chroma_client,
            collection_name=name,
            embedding_function=self.embeddings_model,
        )

    def _build_vector_store(self, entity_name, ticker, source_documents):
        """Chunk -> embed -> lưu xuống Chroma (kèm mốc thời gian created_at để kiểm tra TTL)."""
        # [FIXED] GỌI split_documents (bản gốc khai báo splitter nhưng không bao giờ dùng)
        chunks = self.text_splitter.split_documents(source_documents)
        print(f"--- Chunking: {len(source_documents)} documents -> {len(chunks)} chunks "
              f"(<= {CHUNK_SIZE_TOKENS} tokens each, model limit {EMBEDDING_MAX_TOKENS}). ---")

        name = self._collection_name(entity_name, ticker)
        self._drop_collection(name)  # đảm bảo bắt đầu sạch
        vector_store = Chroma(
            client=self._chroma_client,
            collection_name=name,
            embedding_function=self.embeddings_model,
            collection_metadata={
                "created_at": time.time(),
                "entity": str(entity_name),
                "ticker": str(ticker),
            },
        )
        vector_store.add_documents(chunks)
        return vector_store

    # ------------------------------------------------------------------
    # [FIXED] Dữ liệu tài chính: ưu tiên chuỗi được truyền từ UI
    # ------------------------------------------------------------------
    def _resolve_financial_data(self, ticker, financial_data_result=None):
        if financial_data_result is not None:
            result = financial_data_result  # dùng ngay, KHÔNG gọi lại get_stock_info
        else:
            fin_key = f"fin_{ticker}"
            if fin_key not in self.cache:
                self.cache[fin_key] = get_stock_info.run(ticker)
            result = self.cache[fin_key]

        if not isinstance(result, str) or result.startswith("Error"):
            return "Financial data is currently unavailable."
        print("Using collected financial data.")
        return result

    def _get_context(self, entity_name, ticker, financial_data_result=None):
        """Returns (financial_data, source_documents). Web results are cached in self.cache."""
        financial_data = self._resolve_financial_data(ticker, financial_data_result)

        web_cache_key = f"web_{entity_name}_{ticker}"  # [FIXED] chỉ cache phần web; số liệu tài chính do UI cấp
        if web_cache_key in self.cache:
            print("Returning cached web context.")
            return financial_data, self.cache[web_cache_key]

        source_documents = []

        if not self.search_tool:
            print("--- Web search tool not available, skipping web search. ---")
            self.cache[web_cache_key] = source_documents
            return financial_data, source_documents

        queries = {
            "Official News & Analysis": f'"{entity_name}" recent news',
            "Critical News & Sentiment": f'\"{entity_name}\" issues OR concerns OR investigation OR recall OR safety OR "short interest"',
            "Retail Investor Sentiment": f'site:reddit.com/r/wallstreetbets OR site:reddit.com/r/stocks OR site:seekingalpha.com OR site:fool.com "{entity_name}" OR "{ticker}"',
        }

        for tier, query in queries.items():
            print(f"--- Searching for: {tier} ---")
            try:
                search_result_text = self.search_tool.run(query)
                if search_result_text and "No good search result found" not in search_result_text:
                    doc = Document(page_content=search_result_text, metadata={"source": "DuckDuckGo Search", "title": f"{tier} for {entity_name}"})
                    source_documents.append(doc)
                    print(f"Collected results for {tier}.")
                else:
                    print(f"No good search results for {tier}.")
            except Exception as e:
                print(f"An error occurred during web search for tier '{tier}': {e}")
                continue

        self.cache[web_cache_key] = source_documents
        return financial_data, source_documents

    def _create_rag_chain(self, system_prompt, vector_store):
        if vector_store is None:
            def no_rag_chain(input_data):
                prompt = ChatPromptTemplate.from_template(system_prompt).format(
                    financial_data=input_data.get("financial_data", "N/A"),
                    context_formatted="No web context was available for this analysis."
                )
                response = self.llm.invoke(prompt)
                return {"answer": response, "sources": []}
            # [FIXED] Bọc thành Runnable -> có .invoke(), hết lỗi AttributeError khi web search rỗng
            return RunnableLambda(no_rag_chain)

        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

        def format_docs_with_citations(docs):
            return "\n\n".join([
                f"[Source {i+1}]: Title: {doc.metadata.get('title', 'N/A')}\n"
                f"Published: {doc.metadata.get('published', 'N/A')}\n"
                f"Content: {doc.page_content}"
                for i, doc in enumerate(docs)
            ])

        rag_chain = (
            {
                "context_docs": itemgetter("input") | retriever,
                "input": itemgetter("input"),
                "financial_data": itemgetter("financial_data"),
            }
            | RunnablePassthrough.assign(context_formatted=lambda x: format_docs_with_citations(x["context_docs"]))
            | { "answer": ChatPromptTemplate.from_template(system_prompt) | self.llm, "sources": itemgetter("context_docs") }
        )
        return rag_chain

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input, financial_data_result=None):
        # [FIXED] 1) Ưu tiên vector store còn hạn trên đĩa -> không cần cào web / embed lại
        vector_store = self._load_cached_vector_store(entity_name, ticker)
        if vector_store is not None:
            financial_data = self._resolve_financial_data(ticker, financial_data_result)
        else:
            # 2) Không có cache hợp lệ: lấy web (hoặc self.cache), chunk, embed, lưu xuống đĩa
            financial_data, source_documents = self._get_context(entity_name, ticker, financial_data_result)
            vector_store = (
                self._build_vector_store(entity_name, ticker, source_documents)
                if source_documents else None
            )

        rag_chain = self._create_rag_chain(system_prompt, vector_store)
        response = rag_chain.invoke({"input": query_input, "financial_data": financial_data})
        if isinstance(response, dict):
            return {"answer": response['answer'].content, "sources": response['sources']}
        else:
            return {"answer": response.content, "sources": []}

    # [FIXED] Các hàm generate_* nhận thêm financial_data_result=None
    def generate_market_outlook(self, entity_name, ticker, financial_data_result=None):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a report with these sections:\n"
            "1. **Professional Market Sentiment:** Based on official news/reports.\n"
            "2. **Retail Investor Sentiment:** Based on forum snippets.\n"
            "3. **Valuation Analysis:** Is it expensive or cheap? Reference 'Trailing P/E' and 'Trailing EPS'.\n"
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "This is an objective summary, not financial advice. "
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Market outlook for {entity_name}", financial_data_result)

    def generate_value_analysis(self, entity_name, ticker, financial_data_result=None):
        print("\nGenerating Value Investor Analysis...")
        system_prompt = (
            "You are a 'Value Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a business brief with these sections:\n"
            "1. **Valuation Summary:** State if it's 'Overvalued', 'Undervalued', or 'Fairly Valued', justifying with data.\n"
            "2. **SWOT Analysis:** Bulleted Strengths, Weaknesses, Opportunities, Threats.\n"
            "3. **Competitive Moat:** Describe its long-term advantages.\n"
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}", financial_data_result)

    def generate_devils_advocate_view(self, entity_name, ticker, financial_data_result=None):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' analyst. Your purpose is to challenge the bullish thesis. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, identify the single strongest counter-argument or hidden risk in a concise paragraph.\n"
            "**Cite the source of your information with [1], [2] at the end of the sentence.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Strongest bearish case against {entity_name}?", financial_data_result)

    def generate_final_summary(self, entity_name, market_outlook, value_analysis, devils_advocate):
        print("\nGenerating Final Consensus Summary...")
        combined_analysis = (
            f"--- Market Investor Outlook ---\n{market_outlook}\n\n"
            f"--- Value Investor Analysis ---\n{value_analysis}\n\n"
            f"--- Devil's Advocate View ---\n{devils_advocate}"
        )
        # --- FIX: Replaced prompt with a version that includes a clear rubric ---
        system_prompt = (
            "You are a 'Lead Analyst' synthesizing your team's views for {entity_name}.\n"
            "Return exactly two sections in valid Markdown:\n"
            "1. **Consensus Rating:** Choose exactly one: **Bullish**, **Bearish**, or **Neutral with Caution**.\n"
            "   • Choose **Bullish** if BOTH (A) overall sentiment is net positive AND (B) valuation is not clearly overextended.\n"
            "   • Choose **Bearish** if EITHER (A) major red flags/negative catalysts are present OR (B) valuation is extreme and unsupported.\n"
            "   • Choose **Neutral with Caution** ONLY if evidence is genuinely mixed or insufficient.\n"
            "   • If evidence is mixed but slightly positive, prefer **Bullish**; if mixed but slightly negative, prefer **Bearish**.\n"
            "2. **Summary Justification:** 3–5 sentences, weighing the three views. No line breaks inside words."
            # [FIXED] Khối TEAM REPORTS khớp với khóa "input" của chain.invoke -> Lead Analyst thật sự nhận được 3 báo cáo
            "\n\n--- TEAM REPORTS ---\n{input}"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        response = chain.invoke({"entity_name": entity_name, "input": combined_analysis})
        return response.content