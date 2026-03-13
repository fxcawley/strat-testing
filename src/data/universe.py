"""
Universe definitions -- reusable lists of tickers to operate on.
"""

# S&P 500 subset -- top holdings / liquid names for quick testing
SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "XOM", "MA", "PG", "JNJ",
    "COST", "HD", "ABBV", "MRK", "CRM", "AMD", "BAC", "NFLX",
]

# Sector ETFs
SECTOR_ETFS = {
    "tech": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "consumer_disc": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communications": "XLC",
}
