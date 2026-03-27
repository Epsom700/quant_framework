"""Yahoo Finance connector implementation."""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from quant_framework.connectors.connectors import BaseConnector, ConnectorRegistry

logger = logging.getLogger(__name__)

# ── File-based cache with 24-hour TTL ────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "quant_framework" / "yfinance"
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


# ── Popular tickers by category (used by get_schema) ────────────────────────

POPULAR_TICKERS: Dict[str, List[str]] = {
    "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
    "etfs": ["SPY", "QQQ", "IWM", "DIA"],
    "sectors": ["XLF", "XLK", "XLE", "XLV", "XLI"],
}


# ── YFinanceConnector ────────────────────────────────────────────────────────

@ConnectorRegistry.register("yfinance")
class YFinanceConnector(BaseConnector):
    """
    Connector for Yahoo Finance market data via the ``yfinance`` library.

    Usage::

        conn = YFinanceConnector()
        conn.connect({})
        df = conn.query("SPY", period="1y")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._config: Dict[str, Any] = {}
        self._cache = _FileCache(
            cache_dir=cache_dir or _DEFAULT_CACHE_DIR,
            ttl=cache_ttl,
        )

    # ── BaseConnector interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "yfinance"

    def connect(self, config: Dict[str, Any]) -> None:
        """
        No-op — yfinance needs no authentication. Stores config for reference.

        Args:
            config: Optional configuration dict (kept for interface consistency).
        """
        self._config = config or {}

    def query(self, request: str, **kwargs: Any) -> pd.DataFrame:
        """
        Fetch Yahoo Finance data for a ticker.

        Args:
            request: The ticker symbol (e.g. ``"SPY"``, ``"^GSPC"``).
            **kwargs: Forwarded to ``yfinance.download``
                      (e.g. ``period``, ``interval``, ``start``, ``end``).

        Returns:
            A DataFrame with a ``DatetimeIndex`` named ``"date"`` and
            OHLCV columns (Open, High, Low, Close, Volume).
        """
        # Build a deterministic cache key from the ticker + kwargs
        cache_key = f"series:{request}:{kwargs}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        df: pd.DataFrame = yf.download(request, **kwargs)
        # yf.download may return multi-level columns for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "date"

        self._cache.set(cache_key, df)
        return df

    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema dict listing popular tickers by category.

        Each entry contains the category, ticker ID, and (if available)
        live metadata fetched via ``yfinance.Ticker``.
        """
        schema: Dict[str, Any] = {"source": "yFinance", "series": []}

        for category, tickers in POPULAR_TICKERS.items():
            for ticker in tickers:
                info: Dict[str, Any] = {"category": category, "id": ticker}

                # Try to enrich with live metadata
                cache_key = f"info:{ticker}"
                cached = self._cache.get(cache_key)
                if cached is not None:
                    info.update(cached)
                else:
                    try:
                        meta = yf.Ticker(ticker)
                        live = {
                            "name": meta.info.get("shortName"),
                            "exchange": meta.info.get("exchange"),
                            "type": meta.info.get("quoteType"),
                            "currency": meta.info.get("currency"),
                            "last_price": meta.info.get("regularMarketPrice"),
                            "previous_close": meta.info.get("regularMarketPreviousClose"),
                            "market_cap": meta.info.get("marketCap"),
                            "sector": meta.info.get("sector"),
                            "industry": meta.info.get("industry"),
                        }
                        info.update(live)
                        self._cache.set(cache_key, live)
                    except Exception:
                        pass  # fall through with static info only

                schema["series"].append(info)

        return schema

    def health_check(self) -> bool:
        """Return ``True`` if yfinance can fetch data successfully."""
        try:
            df = yf.download("SPY", period="1d")
            return not df.empty
        except Exception:
            return False

    # ── helpers ───────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Remove all cached data."""
        self._cache.clear()
