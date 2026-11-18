#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_phan2_rag.py  —  DEMO PHẦN 2: DYNAMIC RAG & TRUY XUẤT NGỮ CẢNH
====================================================================
Dữ liệu đầu vào (NVDA) là MOCK, viết sẵn trong file -> không cần mạng, không
DuckDuckGo, không yfinance, không Gemini API Key.

Toàn bộ pipeline xử lý phía sau chạy THẬT 100% trên máy:
    RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings(all-MiniLM-L6-v2)
    -> FAISS (IndexFlatL2) -> LCEL (itemgetter / RunnablePassthrough / RunnableLambda)

(Riêng Trạm 6 dùng một LLM giả FakeListChatModel để không cần API key.
 Trạm 5 dừng ngay tại "tờ prompt hoàn chỉnh" trước khi vào LLM.)

Cách chạy (đặt file này trong thư mục ai-agent-moat):
    python demo_phan2_rag.py            # dừng chờ Enter giữa các trạm
    python demo_phan2_rag.py --auto     # chạy một mạch
    python demo_phan2_rag.py --station 4   # chỉ chạy từ trạm 4 (vẫn dựng đủ pipeline cần thiết)

Lần chạy đầu tiên cần tải model all-MiniLM-L6-v2 (~90MB) từ HuggingFace.
Hãy chạy thử một lần TRƯỚC buổi thuyết trình để model được cache sẵn trong máy.
"""
import sys
import os
import re
import time
import argparse
import textwrap
import warnings

# --- Hiển thị tiếng Việt có dấu mượt trên Terminal Windows ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")
if os.name == "nt":
    os.system("")  # bật xử lý mã ANSI trong cmd / PowerShell

try:
    import numpy as np
    from operator import itemgetter
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as _e:  # pragma: no cover
    print(f"Thiếu thư viện: {_e}\n"
          "Cài đặt:  pip install langchain-core langchain-community langchain-text-splitters "
          "langchain-huggingface sentence-transformers faiss-cpu numpy")
    sys.exit(1)


# ======================================================================
#  TIỆN ÍCH GIAO DIỆN TERMINAL (khung + màu ANSI)
# ======================================================================
class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GRN = "\033[92m"; YEL = "\033[93m"; BLU = "\033[94m"
    MAG = "\033[95m"; CYN = "\033[96m"; GRY = "\033[90m"
    YBG = "\033[43;30m"   # nền vàng, chữ đen  (tô phần chunk_overlap)
    RBG = "\033[41;97m"   # nền đỏ, chữ trắng  (tô phần bị cắt cụt)


W = 104  # bề rộng khung
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BORDER = [C.CYN]  # màu viền của trạm hiện tại


def vlen(s):
    return len(ANSI_RE.sub("", s))


def pad(s, w):
    return s + " " * max(0, w - vlen(s))


def trunc(s, w):
    plain = ANSI_RE.sub("", s)
    if len(plain) <= w or plain != s:
        return s
    return s[: w - 1] + "…"


def row(text="", ):
    b = _BORDER[0]
    print(f"{b}║{C.R} {pad(text, W - 4)} {b}║{C.R}")


def rows_text(text, indent=2, color=""):
    """In đoạn văn (giữ xuống dòng gốc) vào trong khung, tự wrap."""
    for logical in str(text).split("\n"):
        if not logical.strip():
            row()
            continue
        for ln in textwrap.wrap(logical, W - 4 - indent, break_long_words=True) or [""]:
            row(" " * indent + color + ln + C.R)


def rows_marked(text, indent=2, base="", hl=C.YBG):
    """Wrap đoạn văn có chèn \\x01 (bắt đầu tô) và \\x02 (kết thúc tô)."""
    inhit = False
    for ln in textwrap.wrap(text, W - 4 - indent - 2, break_long_words=True):
        out = hl if inhit else ""
        for ch in ln:
            if ch == "\x01":
                out += hl; inhit = True
            elif ch == "\x02":
                out += C.R + base; inhit = False
            else:
                out += ch
        if inhit:
            out += C.R
        row(" " * indent + base + out + C.R)


def sub(title, color=C.YEL):
    b = _BORDER[0]
    print(f"{b}╟{'─' * (W - 2)}╢{C.R}")
    row(f"{color}{C.B}▌ {title}{C.R}")


def table(headers, data, widths, hcolor=C.B + C.YEL):
    def fmt(cells):
        return " │ ".join(pad(trunc(str(c), w), w) for c, w in zip(cells, widths))
    row(hcolor + fmt(headers) + C.R)
    row(C.GRY + "─┼─".join("─" * w for w in widths) + C.R)
    for r in data:
        row(fmt(r))


def station_open(num, title, subtitle, color):
    _BORDER[0] = color
    print()
    print(f"{color}╔{'═' * (W - 2)}╗{C.R}")
    row(f"{C.B}{color}TRẠM {num}/6  ▸  {title}{C.R}")
    row(f"{C.DIM}{subtitle}{C.R}")
    print(f"{color}╠{'═' * (W - 2)}╣{C.R}")


def station_close():
    b = _BORDER[0]
    print(f"{b}╚{'═' * (W - 2)}╝{C.R}")


ARGS = None


def pause(next_label):
    if ARGS and ARGS.auto:
        return
    try:
        input(f"\n{C.DIM}Nhấn Enter để sang {next_label}...{C.R}")
    except EOFError:
        pass


# ======================================================================
#  DỮ LIỆU MOCK VỀ NVIDIA (NVDA)  — chỉ dữ liệu ĐẦU VÀO là giả lập
# ======================================================================
ENTITY = "NVIDIA"
TICKER = "NVDA"

# Định dạng khóa giống tools.py::get_stock_info (+ vài chỉ số bổ sung)
financial_data = {
    "Company Name": "NVIDIA Corporation",
    "Symbol": "NVDA",
    "Current Price": 140.85,
    "Previous Close": 139.40,
    "Open": 139.95,
    "Day Low": 138.72,
    "Day High": 141.60,
    "Market Cap": 3_450_000_000_000,
    "Trailing P/E": 54.2,
    "Forward P/E": 36.8,
    "Trailing EPS": 2.60,
    "Volume": 212_450_000,
    "52 Week High": 152.89,
    "52 Week Low": 75.61,
    "Return on Equity (ROE)": "115.4%",
    "Gross Margins": "75.8%",
    "Revenue Growth (YoY)": "122%",
}


def financial_data_to_string(d):
    """Cùng định dạng 'Key: value' mà tools.get_stock_info trả về cho LLM."""
    return "\n".join(f"{k}: {v if v is not None else 'N/A'}" for k, v in d.items())


def money_short(num):
    if num >= 1e12: return f"${num / 1e12:.2f}T"
    if num >= 1e9: return f"${num / 1e9:.2f}B"
    if num >= 1e6: return f"${num / 1e6:.2f}M"
    return f"${num:,.2f}"


# LƯU Ý: các bài dưới đây là VĂN BẢN GIẢ LẬP phục vụ demo (không phải trích dẫn thật).
# Hai bài dài được viết thành MỘT đoạn liền (không có dòng trống) để splitter phải
# rơi xuống mức tách theo khoảng trắng -> mới nhìn thấy rõ chunk_overlap hoạt động.
DOC_OFFICIAL = (
    "NVIDIA reported record second-quarter fiscal 2025 revenue of $30.0 billion, up 122% from a year ago, "
    "driven by Data Center revenue of $26.3 billion. Management said demand for the Blackwell platform is "
    "well ahead of supply and that the company expects to ship several billion dollars of Blackwell revenue "
    "in the fourth quarter. The Blackwell GPU architecture packs 208 billion transistors, pairs two "
    "reticle-limited dies through a 10 TB/s chip-to-chip link and, according to NVIDIA, delivers up to 30 "
    "times the inference performance of the H100 on large language models while cutting energy use by up to "
    "25 times. Cloud providers including Amazon Web Services, Google Cloud, Microsoft Azure and Oracle have "
    "announced plans to deploy GB200 NVL72 rack-scale systems, which connect 72 Blackwell GPUs and 36 Grace "
    "CPUs into a single liquid-cooled domain. Analysts at several brokerages noted that NVIDIA's moat extends "
    "beyond silicon: the CUDA software ecosystem, now used by more than four million developers, along with "
    "NVLink interconnect, the networking portfolio from the Mellanox acquisition and libraries such as cuDNN "
    "and TensorRT-LLM, creates high switching costs for customers. Gross margin held at 75.8 percent, and the "
    "company returned $15.4 billion to shareholders through buybacks and dividends during the first half of "
    "the fiscal year. Chief Executive Jensen Huang described the shift to accelerated computing and generative "
    "AI as a new industrial revolution, saying that sovereign AI initiatives in Europe, Japan and the Middle "
    "East are opening a fresh demand pipeline."
)

DOC_CRITICAL = (
    "A research note from JP Morgan warned that the economics of AI infrastructure spending are becoming "
    "harder to justify, estimating that hyperscalers would need roughly $600 billion in annual AI-related "
    "revenue to earn an adequate return on the data-center capital now being committed, against a current "
    "run-rate that is a small fraction of that figure. The analysts cautioned that NVIDIA's valuation, at more "
    "than 54 times trailing earnings, already discounts several years of flawless execution and leaves little "
    "room for a digestion phase if customers pause orders. Separately, the US Department of Justice has opened "
    "an antitrust investigation into NVIDIA, sending subpoenas to the company and its customers to examine "
    "whether it penalizes buyers that also use rival accelerators and whether bundling of GPUs, networking and "
    "software amounts to anticompetitive conduct. Regulators in the European Union and China are reportedly "
    "conducting parallel inquiries. Additional risks cited by investors include customer concentration, with a "
    "handful of cloud providers accounting for roughly half of Data Center revenue, the development of "
    "in-house custom chips at Google, Amazon and Microsoft, tightening US export controls that restrict sales "
    "of advanced accelerators to China, and reports of overheating and design delays affecting early Blackwell "
    "racks. Short interest has risen modestly, and several strategists compared the current enthusiasm with the "
    "late-1990s telecom build-out, when infrastructure investment ran far ahead of end-user demand and "
    "suppliers saw revenue collapse once orders were cancelled."
)

DOC_RETAIL = (
    "r/wallstreetbets daily thread: NVDA to the moon, Blackwell is sold out for the next 12 months and every "
    "hyperscaler is begging for allocation. Top comment with 4.2k upvotes: I'm up 340 percent since last year "
    "and I'm not selling before earnings, the CUDA moat is real and AMD is years behind. Others are more "
    "nervous: 54x earnings is insane, one bad guide and this drops 20 percent in a day, and the DOJ probe is "
    "not priced in at all. A popular reply argues that puts are a donation to Jensen's leather jacket fund. "
    "Options volume on the 150 strike calls expiring Friday is at a record, implied volatility is elevated, and "
    "several users posted gain screenshots from leveraged call spreads. The mood is overwhelmingly bullish but "
    "with a visible minority warning about a bubble reminiscent of Cisco in 2000."
)

source_documents = [
    Document(page_content=DOC_OFFICIAL, metadata={
        "tier": "Official News & Analysis",
        "title": "Official News & Analysis for NVIDIA",
        "source": "https://mock.example/nvidia-blackwell-q2-fy2025",
        "published": "2024-08-28"}),
    Document(page_content=DOC_CRITICAL, metadata={
        "tier": "Critical News & Sentiment",
        "title": "Critical News & Sentiment for NVIDIA",
        "source": "https://mock.example/nvidia-jpmorgan-doj-risks",
        "published": "2024-09-05"}),
    Document(page_content=DOC_RETAIL, metadata={
        "tier": "Retail Investor Sentiment",
        "title": "Retail Investor Sentiment for NVIDIA",
        "source": "https://mock.example/r-wallstreetbets-nvda-thread",
        "published": "2024-09-06"}),
]

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 100
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "Risks, valuation bubble, and regulatory counterarguments for NVIDIA"

# Prompt Devil's Advocate lấy nguyên văn từ research_agent.py
DEVIL_PROMPT = (
    "You are a skeptical 'Devil's Advocate' analyst. Your purpose is to challenge the bullish thesis. Key financial data:\n{financial_data}\n\n"
    "Using the retrieved context below, identify the single strongest counter-argument or hidden risk in a concise paragraph.\n"
    "**Cite the source of your information with [1], [2] at the end of the sentence.**\n\nContext:\n{context_formatted}\n\n"
    "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
)


def format_docs_with_citations(docs):
    """Bản demo: như research_agent.py nhưng thêm dòng Published để thấy rõ metadata."""
    return "\n\n".join(
        f"[Source {i + 1}]: Title: {d.metadata.get('title', 'N/A')}\n"
        f"Published: {d.metadata.get('published', 'N/A')}\n"
        f"Content: {d.page_content}"
        for i, d in enumerate(docs)
    )


def find_overlap(a, b, maxk=250):
    """Độ dài chuỗi dài nhất là hậu tố của a đồng thời là tiền tố của b."""
    for k in range(min(len(a), len(b), maxk), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


# Trạng thái dùng chung giữa các trạm
STATE = {}


# ======================================================================
#  TRẠM 1
# ======================================================================
def station1():
    station_open(1, "DỮ LIỆU ĐẦU VÀO CỦA TẦNG RAG",
                 "Structured (con số tài chính)  vs.  Unstructured (tin tức, diễn đàn)", C.CYN)

    sub("A) financial_data  —  DỮ LIỆU CÓ CẤU TRÚC (mock NVDA, cùng khóa với tools.get_stock_info)")
    rows = []
    for k, v in financial_data.items():
        if k == "Market Cap":
            shown = f"{money_short(v)}   ({v:,})"
        elif isinstance(v, float):
            shown = f"{v:,.2f}"
        elif isinstance(v, int):
            shown = f"{v:,}"
        else:
            shown = str(v)
        rows.append((k, shown))
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    for i in range(half):
        l = f"{C.GRN}{left[i][0]:<24}{C.R}{left[i][1]:<22}"
        r = (f"{C.GRN}{right[i][0]:<24}{C.R}{right[i][1]}") if i < len(right) else ""
        row(l + "  " + r)
    row(f"{C.DIM}→ Cấu trúc 'khóa: giá trị' gọn, ít token  ⇒  KHÔNG cần vector hóa, đi thẳng vào prompt.{C.R}")

    sub("B) source_documents  —  DỮ LIỆU PHI CẤU TRÚC (3 tầng tin tức giả lập)")
    data = []
    for i, d in enumerate(source_documents, 1):
        n = len(d.page_content)
        flag = f"{C.RED}{n:>5} ký tự ⚠ >1000{C.R}" if n > CHUNK_SIZE else f"{C.GRN}{n:>5} ký tự  ✔{C.R}"
        data.append((str(i), d.metadata["tier"], d.metadata["source"].replace("https://", ""),
                     d.metadata["published"], flag))
    table(["#", "Tầng (tier)", "Nguồn (metadata.source)", "Xuất bản", "Độ dài"], data, [2, 25, 40, 10, 17])
    row()
    for i, d in enumerate(source_documents, 1):
        row(f"{C.B}[{i}] {d.metadata['title']}{C.R}")
        rows_text(d.page_content[:170] + " …", indent=5, color=C.GRY)
    row(f"{C.DIM}→ Văn bản dài, đa chủ đề  ⇒  phải CẮT NHỎ + VECTOR HÓA + TRUY XUẤT mới đưa vào prompt được.{C.R}")
    station_close()


# ======================================================================
#  TRẠM 2
# ======================================================================
def station2():
    station_open(2, "THỰC NGHIỆM CHUNKING: CODE GỐC vs. ĐÚNG THIẾT KẾ",
                 f"RecursiveCharacterTextSplitter(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})", C.YEL)

    # ---------------- Trạng thái A ----------------
    sub("TRẠNG THÁI A — Code gốc: self.text_splitter được khai báo trong __init__ nhưng KHÔNG BAO GIỜ được gọi", C.RED)
    row(f"{C.GRY}  vector_store = FAISS.from_documents(documents=source_documents, embedding=self.embeddings_model){C.R}")
    row(f"{C.GRY}                                       └── truyền thẳng 3 bài nguyên khối, bỏ qua splitter{C.R}")
    row()
    row(f"  Số Document đưa vào FAISS: {C.B}{C.RED}{len(source_documents)}{C.R}  (mỗi bài = 1 vector duy nhất)")
    row(f"  all-MiniLM-L6-v2 chỉ đọc tối đa 256 token (~{CHUNK_SIZE} ký tự tiếng Anh); phần đuôi bị bỏ qua khi tạo vector:")
    row()
    data = []
    for i, d in enumerate(source_documents, 1):
        n = len(d.page_content)
        lost = max(0, n - CHUNK_SIZE)
        verdict = (f"{C.RED}CẮT CỤT ~{lost} ký tự cuối ({lost * 100 // n}% nội dung 'mù' với retrieval){C.R}"
                   if lost else f"{C.GRN}Vừa khung, không mất gì{C.R}")
        data.append((str(i), d.metadata["tier"], f"{n:>5}", verdict))
    table(["#", "Tầng", "Ký tự", "Hậu quả với MiniLM"], data, [2, 25, 5, 60])
    row()
    for i, d in enumerate(source_documents, 1):
        n = len(d.page_content)
        if n > CHUNK_SIZE:
            row(f"{C.B}  Bài [{i}] — phần đuôi KHÔNG được embedding (hiển thị 200 ký tự cuối):{C.R}")
            rows_marked("…" + "\x01" + d.page_content[-200:] + "\x02", indent=4, hl=C.RBG)

    # ---------------- Trạng thái B ----------------
    sub("TRẠNG THÁI B — Đúng thiết kế: kích hoạt splitter.split_documents(source_documents)", C.GRN)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(source_documents)
    # gán nhãn chunk (D<bài>-<thứ tự>) để theo dõi ở các trạm sau
    counters = {}
    title_to_idx = {d.metadata["title"]: i for i, d in enumerate(source_documents, 1)}
    for c in chunks:
        di = title_to_idx[c.metadata["title"]]
        counters[di] = counters.get(di, 0) + 1
        c.metadata["chunk_id"] = f"D{di}-{counters[di]}"
    STATE["chunks"] = chunks

    row(f"  Sau khi cắt: {len(source_documents)} Document  →  {C.B}{C.GRN}{len(chunks)} chunks{C.R}"
        f"   (mỗi chunk ≤ {CHUNK_SIZE} ký tự, giữ nguyên metadata của bài gốc)")
    row()
    data = []
    for k, c in enumerate(chunks, 1):
        n = len(c.page_content)
        bar = "█" * (n * 30 // CHUNK_SIZE)
        data.append((str(k), c.metadata["chunk_id"], c.metadata["tier"], f"{n:>4}", f"{C.GRN}{bar}{C.R}"))
    table(["#", "Chunk", "Tầng (bài gốc)", "Ký tự", f"Độ dài (thang {CHUNK_SIZE})"], data, [2, 6, 25, 5, 30])

    c1, c2 = chunks[0], chunks[1]
    a, b = c1.page_content, c2.page_content
    k = find_overlap(a, b)
    row()
    row(f"{C.B}  ĐỐI CHIẾU Chunk 1 ({c1.metadata['chunk_id']}) ↔ Chunk 2 ({c2.metadata['chunk_id']}) — "
        f"phần trùng thực đo = {C.YBG} {k} ký tự {C.R}{C.B}  (cấu hình chunk_overlap={CHUNK_OVERLAP}){C.R}")
    row()
    row(f"  {C.CYN}◀ ĐUÔI Chunk 1:{C.R}")
    rows_marked("…" + a[-(k + 130):-k] + "\x01" + a[-k:] + "\x02", indent=4)
    row()
    row(f"  {C.CYN}▶ ĐẦU Chunk 2:{C.R}")
    rows_marked("\x01" + b[:k] + "\x02" + b[k:k + 130] + "…", indent=4)
    row()
    row(f"{C.DIM}  → Phần tô vàng xuất hiện ở CẢ HAI chunk: câu không bị chặt đứt giữa chừng, ngữ cảnh liền mạch.{C.R}")
    station_close()


# ======================================================================
#  TRẠM 3
# ======================================================================
def station3():
    station_open(3, "VECTOR HÓA all-MiniLM-L6-v2 & GIẢI PHẪU RAM CỦA FAISS",
                 "HuggingFaceEmbeddings chạy thật trên CPU  →  FAISS.from_documents()", C.GRN)
    chunks = STATE["chunks"]

    row(f"  {C.YEL}⏳ Đang nạp model {MINILM} lên CPU...{C.R}")
    t0 = time.perf_counter()
    emb = HuggingFaceEmbeddings(model_name=MINILM, model_kwargs={"device": "cpu"})
    t_load = (time.perf_counter() - t0) * 1000
    emb.embed_query("warm-up")  # làm nóng để số đo phản ánh đúng chi phí embedding, không dính khởi động lười
    STATE["emb"] = emb

    t0 = time.perf_counter()
    vs = FAISS.from_documents(documents=chunks, embedding=emb)
    t_build = (time.perf_counter() - t0) * 1000
    STATE["vs"] = vs

    idx = vs.index
    n, d = idx.ntotal, idx.d
    ram = n * d * 4
    client = getattr(emb, "_client", None) or getattr(emb, "client", None)
    max_seq = getattr(client, "max_seq_length", "N/A")

    sub("A) Thời gian (đo thật trên máy bạn)")
    row(f"  Nạp model embedding                    : {C.B}{t_load:>9,.0f} ms{C.R}   (chỉ trả MỘT lần, sau đó nằm trong RAM)")
    row(f"  FAISS.from_documents({n} chunks)          : {C.B}{C.GRN}{t_build:>9,.1f} ms{C.R}   (= embedding {n} chunk + dựng index)")
    row(f"  max_seq_length của model               : {C.B}{max_seq}{C.R} token")

    sub("B) Thông số kỹ thuật đọc trực tiếp từ đối tượng FAISS")
    row(f"  type(vector_store.index).__name__      : {C.B}{C.MAG}{type(idx).__name__}{C.R}"
        f"   → tìm kiếm CHÍNH XÁC (brute-force), so khớp với TẤT CẢ vector")
    row(f"  index.ntotal  (số vector trong RAM)    : {C.B}{n}{C.R}")
    row(f"  index.d       (số chiều mỗi vector)    : {C.B}{d}{C.R}")
    row(f"  RAM ước tính  = {n} × {d} × 4 bytes       : {C.B}{ram:,} bytes ≈ {ram / 1024:.1f} KB{C.R}   (float32 = 4 byte/số)")

    sub("C) Mổ xẻ vector thật của Chunk 1")
    v = idx.reconstruct(0)
    norm = float(np.linalg.norm(v))
    first6 = ", ".join(f"{x:+.4f}" for x in v[:6])
    row(f"  Chunk 1 = {STATE['chunks'][0].metadata['chunk_id']}  ({len(chunks[0].page_content)} ký tự)  →  vector {d} chiều")
    row(f"  6 số đầu tiên : [{C.CYN}{first6}{C.R}, ... còn {d - 6} số]")
    row(f"  ‖v‖₂ = np.linalg.norm(v) = {C.B}{C.GRN}{norm:.4f}{C.R}"
        + ("   ✔ vector đơn vị" if abs(norm - 1) < 1e-3 else f"   {C.YEL}(không chuẩn hóa — kiểm tra encode_kwargs){C.R}"))
    row()
    # Kiểm chứng thực nghiệm đẳng thức  ||a-b||² = 2 - 2·cos(a,b)
    v2 = idx.reconstruct(1)
    l2sq = float(np.sum((v - v2) ** 2))
    cos = float(np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2)))
    row(f"  {C.B}Vì sao L2 ≡ Cosine ở đây?{C.R}  Với ‖a‖ = ‖b‖ = 1:   ‖a − b‖² = 2 − 2·cos(a, b)")
    row(f"  Kiểm chứng giữa Chunk 1 và Chunk 2:  ‖a−b‖² = {C.B}{l2sq:.4f}{C.R}   |   2 − 2·cos = {C.B}{2 - 2 * cos:.4f}{C.R}"
        f"   (cos = {cos:.4f})")
    row(f"{C.DIM}  → L2 nhỏ ⇔ cosine lớn: xếp hạng theo khoảng cách L2 chính là xếp hạng theo độ tương đồng ngữ nghĩa.{C.R}")
    station_close()


# ======================================================================
#  TRẠM 4
# ======================================================================
def station4():
    station_open(4, "TRUY XUẤT NGỮ CẢNH (SIMILARITY SEARCH) & ĐÓNG DẤU NGUỒN",
                 "Persona Devil's Advocate  →  similarity_search_with_score(k=3)  →  format_docs_with_citations", C.MAG)
    vs = STATE["vs"]
    n = vs.index.ntotal

    sub("Truy vấn của Devil's Advocate")
    row(f"  {C.B}{C.YEL}\"{QUERY}\"{C.R}")

    t0 = time.perf_counter()
    results = vs.similarity_search_with_score(QUERY, k=3)
    t_ms = (time.perf_counter() - t0) * 1000
    STATE["top_docs"] = [d for d, _ in results]

    sub(f"Top-3 chunks gần nghĩa nhất (score = L2² trong FAISS — CÀNG NHỎ CÀNG SÁT)   ⏱ {t_ms:.1f} ms")
    data = []
    for rank, (d, s) in enumerate(results, 1):
        s = float(s)
        data.append((f"{rank}", f"{C.GRN}{s:.4f}{C.R}", f"{1 - s / 2:.4f}", d.metadata["chunk_id"],
                     f"{d.metadata['title']}"))
    table(["#", "L2² score", "Cosine", "Chunk", "Title"], data, [2, 10, 8, 6, 60])
    for rank, (d, s) in enumerate(results, 1):
        rows_text(f"↳ #{rank}: " + d.page_content[:150] + " …", indent=4, color=C.GRY)

    allr = vs.similarity_search_with_score(QUERY, k=n)
    top_ids = {d.metadata["chunk_id"] for d, _ in results}
    rest = [(d, float(s)) for d, s in allr if d.metadata["chunk_id"] not in top_ids]
    if rest:
        row()
        row(f"  {C.RED}✘ Bị loại:{C.R} " + "   ".join(f"{d.metadata['chunk_id']} (L2²={s:.4f})" for d, s in rest))
    row(f"{C.DIM}  → Các chunk rủi ro (JP Morgan / DOJ) có khoảng cách nhỏ nhất và được đẩy lên đầu; "
        f"chunk 'tung hô' Blackwell xa hơn.{C.R}")

    sub("format_docs_with_citations(docs)  —  văn bản đã đóng dấu [Source i] đưa cho LLM")
    formatted = format_docs_with_citations(STATE["top_docs"])
    for line in formatted.split("\n"):
        if not line.strip():
            row(); continue
        m = re.match(r"^(\[Source \d+\]): (.*)$", line)
        if m:
            row(f"  {C.B}{C.YEL}{m.group(1)}{C.R}: {C.CYN}{m.group(2)}{C.R}")
        elif line.startswith("Published:"):
            row(f"  {C.CYN}{line}{C.R}")
        else:
            rows_text(line, indent=2, color=C.GRY)
    station_close()


# ======================================================================
#  TRẠM 5
# ======================================================================
def station5():
    station_open(5, "'CHIA 2 LUỒNG' TRONG CHUỖI LCEL (rag_chain)",
                 "itemgetter + RunnablePassthrough.assign  →  ChatPromptTemplate  (dừng trước LLM để lộ tờ prompt)", C.BLU)
    vs = STATE["vs"]
    retriever = vs.as_retriever(search_kwargs={"k": 3})  # repo gốc dùng mặc định k=4; demo dùng k=3 cho khớp Trạm 4
    fin_str = financial_data_to_string(financial_data)
    payload = {"input": f"Strongest bearish case against {ENTITY}?", "financial_data": fin_str}

    sub("Sơ đồ chuỗi")
    l1 = "  input ──► LUỒNG 1: itemgetter('input') | retriever | format_docs "
    l2 = "  fin_data ► LUỒNG 2: itemgetter('financial_data') (đi thẳng 100%) "
    col = max(len(l1), len(l2))
    l1 = l1 + "─" * (col - len(l1)) + "─┐"
    l2 = l2 + "─" * (col - len(l2)) + "─┴─► ChatPromptTemplate ─► (LLM)"
    row(C.B + l1.replace("LUỒNG 1", f"{C.CYN}LUỒNG 1{C.R}{C.B}") + C.R)
    row(C.B + l2.replace("LUỒNG 2", f"{C.GRN}LUỒNG 2{C.R}{C.B}") + C.R)
    row(f"{C.DIM}  (Luồng 2: không cắt nhỏ, không embedding, không lọc vector){C.R}")

    flow1 = itemgetter("input") | retriever | RunnableLambda(format_docs_with_citations)
    # itemgetter thuần là hàm Python, không có .invoke() -> phải bọc RunnableLambda (cùng bản chất với bug Trạm 6)
    flow2 = RunnableLambda(itemgetter("financial_data"))

    sub("LUỒNG 1 — Tin tức báo chí (đi qua retriever)", C.CYN)
    out1 = flow1.invoke(payload)
    retrieved = retriever.invoke(payload["input"])
    row(f"  Retriever lọc từ {vs.index.ntotal} chunks còn {C.B}{len(retrieved)}{C.R}: "
        + ", ".join(d.metadata["chunk_id"] for d in retrieved)
        + f"   → đóng dấu [Source 1..{len(retrieved)}]  ({len(out1):,} ký tự)")

    sub("LUỒNG 2 — Con số tài chính (đi thẳng)", C.GRN)
    out2 = flow2.invoke(payload)
    row(f"  Đầu ra Luồng 2 giống hệt financial_data đầu vào: {C.B}{C.GRN}{out2 == fin_str}{C.R}"
        f"   ({len(out2)} ký tự, 0 lần cắt, 0 lần embedding)")

    # Chuỗi LCEL đúng như research_agent._create_rag_chain, chỉ thay '| self.llm' bằng dừng ở prompt
    prompt_tpl = ChatPromptTemplate.from_template(DEVIL_PROMPT)
    rag_chain = (
        {
            "context_docs": itemgetter("input") | retriever,
            "input": itemgetter("input"),
            "financial_data": itemgetter("financial_data"),
        }
        | RunnablePassthrough.assign(context_formatted=lambda x: format_docs_with_citations(x["context_docs"]))
        | prompt_tpl
    )
    prompt_value = rag_chain.invoke(payload)
    final_prompt = prompt_value.to_string()

    sub("TỜ PROMPT HOÀN CHỈNH — thứ mà LLM thực sự nhận được", C.YEL)
    row(f"  {C.DIM}({len(final_prompt):,} ký tự, ≈ {len(final_prompt) // 4:,} token ước lượng){C.R}")
    row(f"{C.GRY}  {'┄' * (W - 8)}{C.R}")
    for line in final_prompt.split("\n"):
        if not line.strip():
            row(); continue
        if line.startswith("[Source"):
            m = re.match(r"^(\[Source \d+\]): (.*)$", line)
            row(f"  {C.B}{C.YEL}{m.group(1)}{C.R}: {C.CYN}{m.group(2)}{C.R}" if m else "  " + line)
        elif re.match(r"^[A-Z][A-Za-z /()0-9&-]+: \S", line) and len(line) < 70 and not line.startswith("Human"):
            k_, v_ = line.split(": ", 1)
            row(f"  {C.GRN}{k_}{C.R}: {v_}")
        else:
            rows_text(line, indent=2, color=C.GRY)
    row(f"{C.GRY}  {'┄' * (W - 8)}{C.R}")
    station_close()


# ======================================================================
#  TRẠM 6
# ======================================================================
class _AgentStub:
    """Mô phỏng ResearchAgent: chỉ giữ self.llm và hai phiên bản _create_rag_chain."""

    def __init__(self, llm):
        self.llm = llm

    # ---- CODE GỐC của tác giả (research_agent.py, nhánh `if not source_documents`) ----
    def create_rag_chain_BUGGY(self, system_prompt, source_documents):
        if not source_documents:
            def no_rag_chain(input_data):
                prompt = ChatPromptTemplate.from_template(system_prompt).format(
                    financial_data=input_data.get("financial_data", "N/A"),
                    context_formatted="No web context was available for this analysis.",
                )
                response = self.llm.invoke(prompt)
                return {"answer": response, "sources": []}
            return no_rag_chain                      # ← trả về HÀM thường

    # ---- BẢN VÁ ----
    def create_rag_chain_FIXED(self, system_prompt, source_documents):
        if not source_documents:
            def no_rag_chain(input_data):
                prompt = ChatPromptTemplate.from_template(system_prompt).format(
                    financial_data=input_data.get("financial_data", "N/A"),
                    context_formatted="No web context was available for this analysis.",
                )
                response = self.llm.invoke(prompt)
                return {"answer": response, "sources": []}
            return RunnableLambda(no_rag_chain)      # ← bọc thành Runnable


def station6():
    station_open(6, "TÁI HIỆN & VÁ BUG no_rag_chain (research_agent.py ~ dòng 80)",
                 "source_documents = []  →  Before vs. After   (LLM được thay bằng FakeListChatModel, không cần API key)", C.RED)
    fake_llm = FakeListChatModel(responses=[
        "[MOCK LLM] Không có ngữ cảnh web; phân tích chỉ dựa trên số liệu tài chính: "
        "P/E 54.2 cao hơn nhiều so với Forward P/E 36.8, nghĩa là thị trường đang kỳ vọng tăng trưởng lợi nhuận rất mạnh."])
    agent = _AgentStub(fake_llm)
    payload = {"input": f"Strongest bearish case against {ENTITY}?",
               "financial_data": financial_data_to_string(financial_data)}
    empty_docs = []  # kịch bản DuckDuckGo lỗi DNS / rate-limit  ⇒  không có tài liệu nào

    sub("Giả lập: source_documents = []   (web search thất bại hoàn toàn)")
    row(f"  len(source_documents) = {C.B}{C.RED}{len(empty_docs)}{C.R}  →  _create_rag_chain đi vào nhánh `if not source_documents:`")

    sub("BEFORE — Code gốc:  return no_rag_chain", C.RED)
    row(f"{C.GRY}    def no_rag_chain(input_data): ...{C.R}")
    row(f"{C.GRY}    return no_rag_chain        # hàm Python thuần, KHÔNG phải Runnable{C.R}")
    row(f"{C.GRY}  ...rồi _run_analysis gọi:  rag_chain.invoke({{...}}){C.R}")
    row()
    try:
        buggy_chain = agent.create_rag_chain_BUGGY(DEVIL_PROMPT, empty_docs)
        buggy_chain.invoke(payload)
        row(f"  {C.YEL}(không có lỗi?){C.R}")
    except Exception as e:
        row(f"  {C.B}{C.RED}✘ {type(e).__name__}: {e}{C.R}")
        row(f"{C.DIM}    → App Streamlit sẽ rơi vào except trong run_full_analysis và hiện 'A critical error occurred'.{C.R}")

    sub("AFTER — Bản vá:  return RunnableLambda(no_rag_chain)", C.GRN)
    row(f"{C.GRY}    return RunnableLambda(no_rag_chain)   # bọc hàm thành Runnable  ⇒  có .invoke(){C.R}")
    row()
    try:
        fixed_chain = agent.create_rag_chain_FIXED(DEVIL_PROMPT, empty_docs)
        result = fixed_chain.invoke(payload)
        row(f"  {C.B}{C.GRN}✔ .invoke() thành công{C.R}   kiểu trả về: {type(result).__name__}   sources = {result['sources']}")
        row(f"  answer.content (từ LLM giả):")
        rows_text(result["answer"].content, indent=4, color=C.CYN)
        row(f"{C.DIM}    → Khi web search sập, agent vẫn trả lời được (degraded mode) thay vì crash toàn bộ.{C.R}")
    except Exception as e:
        row(f"  {C.RED}✘ Bản vá vẫn lỗi: {type(e).__name__}: {e}{C.R}")
    station_close()


# ======================================================================
#  MAIN
# ======================================================================
def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Demo Phần 2: Dynamic RAG (mock NVDA, pipeline thật)")
    ap.add_argument("--auto", action="store_true", help="chạy một mạch, không dừng chờ Enter")
    ap.add_argument("--station", type=int, default=1, choices=range(1, 7), metavar="N",
                    help="bắt đầu hiển thị từ trạm N (1-6)")
    ARGS = ap.parse_args()

    print(f"\n{C.B}{C.CYN}{'█' * W}{C.R}")
    print(f"{C.B}{C.CYN}█{C.R}{C.B}{'DEMO PHẦN 2 — DYNAMIC RAG & TRUY XUẤT NGỮ CẢNH  |  AI Agent for Financial Moat Analysis'.center(W - 2)}{C.CYN}█{C.R}")
    print(f"{C.B}{C.CYN}█{C.R}{C.DIM}{'Input: MOCK NVDA (offline)   •   Pipeline: THẬT (Splitter · MiniLM · FAISS · LCEL)'.center(W - 2)}{C.R}{C.B}{C.CYN}█{C.R}")
    print(f"{C.B}{C.CYN}{'█' * W}{C.R}")

    stations = [station1, station2, station3, station4, station5, station6]
    names = ["Trạm 1", "Trạm 2", "Trạm 3", "Trạm 4", "Trạm 5", "Trạm 6"]

    # Các trạm sau cần trạng thái do trạm trước tạo ra: chạy "ngầm" nếu người dùng nhảy cóc.
    start = ARGS.station
    if start > 1:
        import io, contextlib
        for i in range(start - 1):
            if i + 1 in (2, 3, 4):  # các trạm tạo state (chunks, vector_store, top_docs)
                with contextlib.redirect_stdout(io.StringIO()):
                    stations[i]()

    for i in range(start - 1, 6):
        stations[i]()
        if i < 5:
            pause(names[i + 1])

    print(f"\n{C.B}{C.GRN}✔ Hoàn tất 6 trạm.{C.R} "
          f"{C.DIM}Dữ liệu đầu vào là mock; chunking, embedding, FAISS, LCEL đều chạy thật.{C.R}\n")


if __name__ == "__main__":
    main()