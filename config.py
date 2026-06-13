# =========================
# UNIVERSE (GLOBAL EQUITY + TECH + FINANCE + ENERGY)
# =========================

UNIVERSE = list(dict.fromkeys([
    "SNDK", "MU", "STX", "INTC", "AMD", "DELL", "LRCX", "ARM", "AMAT",
    "CAT", "TSM", "GOOG", "GEV", "ANET", "CSCO", "AVGO", "BHP", "GS",
    "C", "NVDA", "MS", "TXN", "QCOM", "PANW", "APP", "AAPL", "ORCL",
    "MRK", "LLY", "XOM", "JNJ", "CVX", "MUFG", "GE", "SHEL", "RTX",
    "AMZN", "UNH", "IBM", "PLTR", "TSLA", "BAC", "WMT", "ABBV", "JPM",
    "BABA", "KO", "AXP", "LIN", "WFC", "MSFT", "PM", "BRK-B", "META",
    "V", "MA", "HD", "PG", "TMUS", "SAP"
]))

# =========================
# BENCHMARK (GLOBAL EQUITY ETF)
# =========================

BENCHMARK = "VWCE.MI"

# =========================
# PORTFOLIO SETTINGS
# =========================

BUY_THRESHOLD = 40
SELL_THRESHOLD = 25

MAX_POSITIONS = 10