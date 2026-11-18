import yfinance as yf
from langchain.tools import tool
import requests  # Keep for potential future use, but not for yfinance session

# [FIXED] 12 trường số liệu dùng cho "Error Gate": nếu TẤT CẢ đều None thì coi như không có dữ liệu.
FINANCIAL_FIELDS = [
    "Current Price", "Previous Close", "Open", "Day Low", "Day High", "Market Cap",
    "Trailing P/E", "Forward P/E", "Trailing EPS", "Volume", "52 Week High", "52 Week Low",
]


@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetches key financial information for a given stock ticker using the yfinance library.
    Multi-step, resilient approach: fast_info -> history(5d) fallback -> get_info() enrichment.
    Each step is isolated, so one failing step never aborts the others.
    Returns a formatted string of the data, using 'N/A' for unavailable metrics, or
    "Error: No financial data available for ticker '<ticker>'." when nothing could be retrieved.
    """
    try:
        stock = yf.Ticker(ticker)
        data = {}

        # --- Bước 1: fast_info ---
        try:
            fi = stock.fast_info
        except Exception:
            fi = None

        # [FIXED] pick(*keys): đọc linh hoạt snake_case / camelCase. FastInfo tải "lười" nên lỗi
        # (vd. KeyError 'currentTradingPeriod') chỉ nổ ra khi đọc khóa -> bọc try/except từng khóa.
        def pick(*keys):
            if fi is None:
                return None
            for k in keys:
                try:
                    v = fi[k]
                except Exception:
                    continue
                if v is not None:
                    return v
            return None

        data["Current Price"]  = pick("last_price", "lastPrice")
        data["Previous Close"] = pick("previous_close", "previousClose")
        data["Open"]           = pick("open", "regularMarketOpen")
        data["Day Low"]        = pick("day_low", "dayLow")
        data["Day High"]       = pick("day_high", "dayHigh")
        data["Volume"]         = pick("last_volume", "volume")
        data["Market Cap"]     = pick("market_cap", "marketCap")
        data["Currency"]       = pick("currency")  # [FIXED] thu thập đơn vị tiền tệ từ fast_info

        # --- Bước 2: [FIXED] nếu Current Price / Open / Previous Close còn thiếu -> lấy từ history(5d) ---
        if any(data.get(k) is None for k in ("Current Price", "Open", "Previous Close")):
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    close = hist["Close"].dropna()
                    if data.get("Current Price") is None and not close.empty:
                        data["Current Price"] = float(close.iloc[-1])
                    if data.get("Previous Close") is None and len(close) > 1:
                        data["Previous Close"] = float(close.iloc[-2])
                    if data.get("Open") is None and "Open" in hist.columns:
                        opening = hist["Open"].dropna()
                        if not opening.empty:
                            data["Open"] = float(opening.iloc[-1])
                    if data.get("Day Low") is None and "Low" in hist.columns:
                        low = hist["Low"].dropna()
                        if not low.empty:
                            data["Day Low"] = float(low.iloc[-1])
                    if data.get("Day High") is None and "High" in hist.columns:
                        high = hist["High"].dropna()
                        if not high.empty:
                            data["Day High"] = float(high.iloc[-1])
            except Exception:
                pass  # history hỏng -> bỏ qua, các bước sau vẫn chạy

        # --- Bước 3: [FIXED] bổ sung P/E, EPS, 52-week High/Low (và các chỗ còn trống) từ get_info() ---
        try:
            info = stock.get_info() or {}
        except Exception:
            info = {}

        def from_info(*keys):
            for k in keys:
                v = info.get(k)
                if v is not None:
                    return v
            return None

        data["Company Name"] = from_info("longName", "shortName")
        data["Symbol"]       = from_info("symbol")
        data["Trailing P/E"] = from_info("trailingPE")
        data["Forward P/E"]  = from_info("forwardPE")
        data["Trailing EPS"] = from_info("trailingEps")
        data["52 Week High"] = from_info("fiftyTwoWeekHigh")
        data["52 Week Low"]  = from_info("fiftyTwoWeekLow")

        # Điền chỗ trống còn lại bằng info (dự phòng khi fast_info + history đều thất bại)
        info_fallbacks = {
            "Current Price":  ("currentPrice", "regularMarketPrice"),
            "Previous Close": ("previousClose", "regularMarketPreviousClose"),
            "Open":           ("open", "regularMarketOpen"),
            "Day Low":        ("dayLow", "regularMarketDayLow"),
            "Day High":       ("dayHigh", "regularMarketDayHigh"),
            "Volume":         ("volume", "regularMarketVolume"),
            "Market Cap":     ("marketCap",),
            "Currency":       ("currency", "financialCurrency"),  # [FIXED] dự phòng currency qua get_info()
        }
        for field, info_keys in info_fallbacks.items():
            if data.get(field) is None:
                data[field] = from_info(*info_keys)

        # --- [FIXED] Error Gate: kiểm tra TRƯỚC khi đổi None thành "N/A" ---
        if all(data.get(field) is None for field in FINANCIAL_FIELDS):
            return f"Error: No financial data available for ticker '{ticker}'."

        # [FIXED] "Currency" nằm ngay sau "Symbol"
        key_order = [
            "Company Name", "Symbol", "Currency", "Current Price", "Previous Close", "Open",
            "Day Low", "Day High", "Market Cap", "Trailing P/E", "Forward P/E",
            "Trailing EPS", "Volume", "52 Week High", "52 Week Low",
        ]

        report = []
        for key in key_order:
            value = data.get(key)
            report.append(f"{key}: {value if value is not None else 'N/A'}")

        return "\n".join(report)

    except Exception as e:
        return f"Error: An unexpected exception occurred while fetching data for ticker '{ticker}'. Details: {e}"


@tool
def scrape_website(url: str) -> str:
    """A placeholder for a website scraping tool."""
    return "Scraping functionality is not implemented in this demo."
