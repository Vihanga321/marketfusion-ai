"""Static contract for MarketFusion V0.5B continuous intelligence."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "intelligence"
NEWS_FILE = DATA_ROOT / "news" / "news_items.parquet"
MACRO_FILE = DATA_ROOT / "macro" / "macro_observations.parquet"
CONTEXT_HISTORY_FILE = DATA_ROOT / "context" / "v05b_context_history.parquet"
STATUS_FILE = DATA_ROOT / "runtime_status.json"
QUALITY_REPORT = ROOT / "reports" / "v05b_continuous_intelligence_quality.txt"
CONTRACT_VERSION = "v0.5b-continuous-intelligence-v1"
DEFAULT_INTERVAL_SECONDS = 300

NEWS_SOURCES = {
    "FED_MONETARY": {
        "kind": "rss",
        "url": "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "authority": "OFFICIAL",
        "region": "USD",
    },
    "FED_SPEECHES": {
        "kind": "rss",
        "url": "https://www.federalreserve.gov/feeds/speeches.xml",
        "authority": "OFFICIAL",
        "region": "USD",
    },
    "ECB_PRESS": {
        "kind": "rss",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "authority": "OFFICIAL",
        "region": "EUR",
    },
    "ECB_STATPRESS": {
        "kind": "rss",
        "url": "https://www.ecb.europa.eu/rss/statpress.html",
        "authority": "OFFICIAL",
        "region": "EUR",
    },
    "BLS_EMPLOYMENT": {
        "kind": "rss",
        "url": "https://www.bls.gov/feed/empsit.rss",
        "authority": "OFFICIAL",
        "region": "USD",
    },
    "BLS_CPI": {
        "kind": "rss",
        "url": "https://www.bls.gov/feed/cpi.rss",
        "authority": "OFFICIAL",
        "region": "USD",
    },
    "GDELT_EURUSD": {
        "kind": "gdelt",
        "url": "https://api.gdeltproject.org/api/v2/doc/doc",
        "authority": "AGGREGATOR",
        "region": "GLOBAL",
    },
}

GDELT_QUERY = '("Federal Reserve" OR "European Central Bank" OR inflation OR unemployment OR payrolls OR "interest rates" OR euro OR dollar)'

MACRO_SOURCES = {
    "us_effective_fed_funds": {
        "provider": "FRED_GRAPH",
        "series": "DFF",
        "unit": "percent",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF",
    },
    "us_2y_yield": {
        "provider": "FRED_GRAPH",
        "series": "DGS2",
        "unit": "percent",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
    },
    "us_10y_yield": {
        "provider": "FRED_GRAPH",
        "series": "DGS10",
        "unit": "percent",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    },
    "us_10y_minus_2y": {
        "provider": "FRED_GRAPH",
        "series": "T10Y2Y",
        "unit": "percentage_points",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
    },
    "ecb_main_refinancing_rate": {
        "provider": "ECB_SDMX",
        "series": "FM.D.U2.EUR.4F.KR.MRR_RT.LEV",
        "unit": "percent",
        "url": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.MRR_RT.LEV",
    },
}

TOPIC_KEYWORDS = {
    "central_bank": ("federal reserve", "fomc", "ecb", "central bank", "monetary policy", "policy rate"),
    "rates": ("interest rate", "interest rates", "rate cut", "rate hike", "yield", "treasury", "bond"),
    "inflation": ("inflation", "cpi", "consumer price", "core price", "ppi", "price pressures"),
    "labor": ("employment", "unemployment", "payroll", "jobs", "wages", "labor market", "labour market"),
    "growth": ("gdp", "growth", "recession", "activity", "output", "pmi"),
    "risk": ("war", "tariff", "sanction", "geopolitical", "bank crisis", "default", "risk-off", "risk on"),
}
USD_KEYWORDS = ("usd", "dollar", "u.s.", "united states", "federal reserve", "fomc", "treasury", "bls")
EUR_KEYWORDS = ("eur", "euro", "euro area", "eurozone", "ecb", "european central bank")
HIGH_IMPACT_KEYWORDS = (
    "fomc", "federal reserve", "ecb", "monetary policy", "interest rate", "rate decision",
    "cpi", "inflation", "employment situation", "nonfarm", "payroll", "unemployment", "gdp",
)

NEWS_WINDOWS_MINUTES = (15, 60, 240, 1440)
MACRO_FEATURE_SERIES = tuple(MACRO_SOURCES)
