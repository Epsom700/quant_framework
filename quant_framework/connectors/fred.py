"""FRED (Federal Reserve Economic Data) connector implementation."""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

from quant_framework.connectors.connectors import BaseConnector, ConnectorRegistry

load_dotenv()
# ── File-based cache with 24-hour TTL ────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "quant_framework" / "fred"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class _FileCache:
    """Simple file-based cache using pickle with a configurable TTL."""

    def __init__(
        self,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        ttl: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._cache_dir = cache_dir
        self._ttl = ttl
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────

    def _key_path(self, key: str) -> Path:
        safe = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{safe}.pkl"

    # ── public API ────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if it exists and hasn't expired, else None."""
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                entry = pickle.load(f)
            if time.time() - entry["ts"] > self._ttl:
                path.unlink(missing_ok=True)
                return None
            return entry["data"]
        except (pickle.UnpicklingError, KeyError, EOFError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Persist *value* under *key* with the current timestamp."""
        path = self._key_path(key)
        with open(path, "wb") as f:
            pickle.dump({"ts": time.time(), "data": value}, f)

    def clear(self) -> None:
        """Remove every cached file."""
        for p in self._cache_dir.glob("*.pkl"):
            p.unlink(missing_ok=True)


# ── Top-20 popular FRED series (used by get_schema) ─────────────────────────

POPULAR_SERIES: List[Dict[str, str]] = [
    {"id": "GDP",            "title": "Gross Domestic Product"},
    {"id": "UNRATE",         "title": "Unemployment Rate"},
    {"id": "CPIAUCSL",       "title": "Consumer Price Index for All Urban Consumers"},
    {"id": "FEDFUNDS",       "title": "Federal Funds Effective Rate"},
    {"id": "DGS10",          "title": "10-Year Treasury Constant Maturity Rate"},
    {"id": "SP500",          "title": "S&P 500"},
    {"id": "M2SL",           "title": "M2 Money Stock"},
    {"id": "PAYEMS",         "title": "All Employees, Total Nonfarm"},
    {"id": "HOUST",          "title": "New Privately-Owned Housing Units Started"},
    {"id": "RSAFS",          "title": "Advance Retail Sales: Retail and Food Services"},
    {"id": "INDPRO",         "title": "Industrial Production: Total Index"},
    {"id": "PCE",            "title": "Personal Consumption Expenditures"},
    {"id": "UMCSENT",        "title": "University of Michigan Consumer Sentiment"},
    {"id": "T10YIE",         "title": "10-Year Breakeven Inflation Rate"},
    {"id": "VIXCLS",         "title": "CBOE Volatility Index: VIX"},
    {"id": "DEXUSEU",        "title": "US / Euro Foreign Exchange Rate"},
    {"id": "DCOILWTICO",     "title": "Crude Oil Prices: WTI"},
    {"id": "BAMLH0A0HYM2",   "title": "ICE BofA US High Yield Index OAS"},
    {"id": "WALCL",          "title": "Federal Reserve Total Assets"},
    {"id": "CPILFESL",       "title": "CPI Less Food and Energy (Core CPI)"},
]


# ── FREDConnector ────────────────────────────────────────────────────────────

@ConnectorRegistry.register("fred")
class FREDConnector(BaseConnector):
    """
    Connector for the Federal Reserve Economic Data (FRED) API.

    Usage::

        conn = FREDConnector()
        conn.connect({"api_key": "YOUR_API_KEY"})
        df = conn.query("GDP", observation_start="2020-01-01")
    """

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._fred: Optional[Fred] = None
        self._cache = _FileCache(
            cache_dir=cache_dir or _DEFAULT_CACHE_DIR,
            ttl=cache_ttl,
        )

    # ── BaseConnector interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fred"

    def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to the FRED API.

        Args:
            config: Must contain ``api_key``.
        """
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("config must contain 'api_key'")
        self._fred = Fred(api_key=api_key)

    def query(self, request: str, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch a FRED series by its series ID.

        Args:
            request: The FRED series ID (e.g. ``"GDP"``).
            **kwargs: Forwarded to ``fredapi.Fred.get_series`` (e.g. ``observation_start``, ``observation_end``).

        Returns:
            A DataFrame with a ``DatetimeIndex`` named ``"date"`` and a single
            column named after the series ID.
        """
        self._ensure_connected()

        # Build a deterministic cache key from the series id + kwargs
        cache_key = f"series:{request}:{kwargs}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        series: pd.Series = self._fred.get_series(request, **kwargs)  # type: ignore[union-attr]
        df = series.to_frame(name=request)
        df.index.name = "date"

        self._cache.set(cache_key, df)
        return df

    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema dict listing the top-20 popular FRED series.

        If connected, each entry is enriched with live metadata from FRED;
        otherwise the static ``POPULAR_SERIES`` list is returned.
        """
        schema: Dict[str, Any] = {"source": "FRED", "series": []}

        for entry in POPULAR_SERIES:
            info: Dict[str, Any] = {"id": entry["id"], "title": entry["title"]}

            if self._fred is not None:
                cache_key = f"info:{entry['id']}"
                cached = self._cache.get(cache_key)
                if cached is not None:
                    info.update(cached)
                else:
                    try:
                        meta = self._fred.get_series_info(entry["id"])
                        live = {
                            "frequency": meta.get("frequency", ""),
                            "units": meta.get("units", ""),
                            "seasonal_adjustment": meta.get("seasonal_adjustment", ""),
                            "last_updated": str(meta.get("last_updated", "")),
                        }
                        info.update(live)
                        self._cache.set(cache_key, live)
                    except Exception:
                        pass  # fall through with static info only

            schema["series"].append(info)

        return schema

    def health_check(self) -> bool:
        """Return ``True`` if the FRED API is reachable."""
        if self._fred is None:
            return False
        try:
            self._fred.get_series_info("GDP")
            return True
        except Exception:
            return False

    # ── helpers ───────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._fred is None:
            raise RuntimeError("Not connected. Call connect(config) first.")

    def clear_cache(self) -> None:
        """Remove all cached data."""
        self._cache.clear()