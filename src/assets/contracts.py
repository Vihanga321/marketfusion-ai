"""V1.0A asset identity, broker contract, and path isolation contracts.

Static asset identity is deliberately separate from broker-supplied contract
details.  The latter must be read from the connected MT5 terminal and is never
guessed from Forex conventions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ASSETS = ("EURUSD", "XAUUSD")
ALIASES = {
    "EURUSD": "EURUSD", "EUR/USD": "EURUSD", "EUR_USD": "EURUSD",
    "XAUUSD": "XAUUSD", "XAU/USD": "XAUUSD", "GOLD": "XAUUSD",
}


@dataclass(frozen=True)
class AssetConfig:
    asset_id: str
    asset_class: str
    display_name: str
    display_symbol: str
    quote_currency: str
    broker_search_terms: tuple[str, ...]
    enabled_by_default: bool
    risk_profile: str


ASSETS = {
    "EURUSD": AssetConfig(
        "EURUSD", "FOREX", "Euro / U.S. Dollar", "EUR/USD", "USD",
        ("EURUSD",), True, "FX_MAJOR",
    ),
    "XAUUSD": AssetConfig(
        "XAUUSD", "PRECIOUS_METAL", "Gold / U.S. Dollar", "XAU/USD", "USD",
        ("XAU", "GOLD"), False, "GOLD_SPOT",
    ),
}


def normalize_asset_id(value: str) -> str:
    key = str(value).strip().upper().replace(" ", "")
    try:
        return ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported asset {value!r}; choose EURUSD or XAUUSD") from exc


def get_asset(value: str) -> AssetConfig:
    return ASSETS[normalize_asset_id(value)]


@dataclass(frozen=True)
class BrokerSymbolSpec:
    asset_id: str
    broker_symbol: str
    description: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread: int
    trade_mode: int
    currency_base: str
    currency_profit: str
    currency_margin: str

    def __post_init__(self) -> None:
        if self.asset_id not in SUPPORTED_ASSETS:
            raise ValueError("Unknown asset_id")
        if not self.broker_symbol or self.digits < 0:
            raise ValueError("Invalid broker symbol identity")
        if self.point <= 0 or self.trade_tick_size <= 0 or self.contract_size <= 0:
            raise ValueError("Broker point, tick size, and contract size must be positive")
        if self.volume_min <= 0 or self.volume_max < self.volume_min or self.volume_step <= 0:
            raise ValueError("Invalid broker volume contract")

    @classmethod
    def from_mt5(cls, asset_id: str, info: Any) -> "BrokerSymbolSpec":
        asset = normalize_asset_id(asset_id)
        return cls(
            asset_id=asset,
            broker_symbol=str(info.name),
            description=str(getattr(info, "description", "")),
            digits=int(info.digits), point=float(info.point),
            trade_tick_size=float(info.trade_tick_size),
            trade_tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min), volume_max=float(info.volume_max),
            volume_step=float(info.volume_step), spread=int(info.spread),
            trade_mode=int(info.trade_mode),
            currency_base=str(getattr(info, "currency_base", "")),
            currency_profit=str(getattr(info, "currency_profit", "")),
            currency_margin=str(getattr(info, "currency_margin", "")),
        )

    def price_to_points(self, price_move: float) -> float:
        return float(price_move) / self.point

    def price_to_ticks(self, price_move: float) -> float:
        return float(price_move) / self.trade_tick_size

    def points_to_price(self, points: float) -> float:
        return float(points) * self.point

    def ticks_to_price(self, ticks: float) -> float:
        return float(ticks) * self.trade_tick_size

    def relative_move(self, price_move: float, reference_price: float) -> float:
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        return float(price_move) / float(reference_price)

    def atr_fraction(self, price_move: float, atr: float) -> float:
        if atr <= 0:
            raise ValueError("atr must be positive")
        return float(price_move) / float(atr)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _gold_candidates(symbols: Iterable[Any]) -> list[Any]:
    candidates = []
    for symbol in symbols:
        name = str(getattr(symbol, "name", ""))
        description = str(getattr(symbol, "description", ""))
        upper = f"{name} {description}".upper()
        if "XAU" not in upper and "GOLD" not in upper:
            continue
        # Exclude equities/ETFs called GOLD. Spot gold must have XAU base and USD profit.
        if str(getattr(symbol, "currency_base", "")).upper() != "XAU":
            continue
        if str(getattr(symbol, "currency_profit", "")).upper() != "USD":
            continue
        candidates.append(symbol)
    return sorted(candidates, key=lambda item: (str(item.name).upper() != "XAUUSD", len(str(item.name)), str(item.name)))


def discover_broker_symbol(mt5: Any, asset_id: str) -> BrokerSymbolSpec:
    """Discover a supported broker symbol from an initialized MT5 adapter."""
    asset = normalize_asset_id(asset_id)
    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"MT5 symbols_get failed: {mt5.last_error()}")
    if asset == "XAUUSD":
        candidates = _gold_candidates(symbols)
    else:
        candidates = [item for item in symbols if str(getattr(item, "name", "")).upper() == asset]
    if not candidates:
        raise RuntimeError(f"No supported broker symbol found for {asset}")
    info = mt5.symbol_info(str(candidates[0].name))
    if info is None:
        raise RuntimeError(f"MT5 symbol_info failed for {candidates[0].name}: {mt5.last_error()}")
    spec = BrokerSymbolSpec.from_mt5(asset, info)
    if asset == "XAUUSD" and (spec.currency_base.upper(), spec.currency_profit.upper()) != ("XAU", "USD"):
        raise RuntimeError("Discovered gold symbol is not XAU quoted in USD")
    return spec


@dataclass(frozen=True)
class AssetPaths:
    asset_id: str
    market: Path
    bars: Path
    quotes: Path
    features: Path
    evaluation: Path
    models: Path
    reports: Path
    runtime: Path
    model_registry: Path
    shadow_predictions: Path
    shadow_outcomes: Path

    def bar_file(self, timeframe: str, broker_symbol: str | None = None) -> Path:
        symbol = broker_symbol or self.asset_id
        return self.bars / f"{symbol}_{timeframe.upper()}.parquet"


def asset_paths(asset_id: str, root: Path = ROOT) -> AssetPaths:
    asset = normalize_asset_id(asset_id)
    market = root / "data" / "market" / asset
    evaluation = root / "data" / "evaluation" / asset
    models = root / "models" / asset
    return AssetPaths(
        asset_id=asset, market=market, bars=market / "bars", quotes=market / "quotes",
        features=root / "data" / "features" / asset, evaluation=evaluation,
        models=models, reports=root / "reports" / asset, runtime=root / "data" / "runtime" / asset,
        model_registry=models / "registry.json",
        shadow_predictions=evaluation / "shadow_predictions.parquet",
        shadow_outcomes=evaluation / "shadow_outcomes.parquet",
    )


def legacy_eurusd_paths(root: Path = ROOT) -> dict[str, Path]:
    """Documented compatibility map; existing EURUSD artifacts are not moved."""
    return {
        "market": root / "data" / "mt5" / "continuous",
        "features": root / "data" / "processed" / "v05a_eurusd_continuous_features.parquet",
        "models": root / "models" / "v05c",
        "evaluation": root / "data" / "evaluation" / "v08",
        "reports": root / "reports",
    }
