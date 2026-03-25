"""
prepare_snapshot.py — One-time data preparation for the autoresearch loop.

Downloads FRED series and caches them as JSON files in data_snapshot/.
This ensures every experiment runs against identical data — the agent
never touches this file or the snapshot directory.

Usage:
    uv run python experiments/prepare_snapshot.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_framework.connectors.fred import FREDConnector

# ── Series to cache ───────────────────────────────────────────────────────────
# The agent can USE any of these but cannot ADD to the snapshot.
# To expand the available data, the human edits this list and reruns.

SERIES = {
    "GDP":       {"observation_start": "2000-01-01", "frequency": "q"},
    "UNRATE":    {"observation_start": "2000-01-01", "frequency": "m"},
    "FEDFUNDS":  {"observation_start": "2000-01-01", "frequency": "m"},
    "GS10":      {"observation_start": "2000-01-01", "frequency": "m"},
    "GS2":       {"observation_start": "2000-01-01", "frequency": "m"},
    "CPIAUCSL":  {"observation_start": "2000-01-01", "frequency": "m"},
    "T10Y2Y":    {"observation_start": "2000-01-01", "frequency": "d"},
    "BAMLH0A0HYM2":  {"observation_start": "2000-01-01", "frequency": "d"},  # HY spread
    "DCOILWTICO": {"observation_start": "2000-01-01", "frequency": "d"},       # WTI crude
    "VIXCLS":    {"observation_start": "2000-01-01", "frequency": "d"},        # VIX
    "UMCSENT":   {"observation_start": "2000-01-01", "frequency": "m"},        # Consumer sentiment
    "HOUST":     {"observation_start": "2000-01-01", "frequency": "m"},        # Housing starts
    "INDPRO":    {"observation_start": "2000-01-01", "frequency": "m"},        # Industrial production
    "M2SL":      {"observation_start": "2000-01-01", "frequency": "m"},        # M2 money supply
}


def main():
    snapshot_dir = Path(__file__).resolve().parent / "data_snapshot"
    snapshot_dir.mkdir(exist_ok=True)

    fred = FREDConnector()
    fred.connect()  # uses FRED_API_KEY from env

    print(f"Caching {len(SERIES)} FRED series to {snapshot_dir}/")

    for series_id, params in SERIES.items():
        try:
            df = fred.query(series_id, **params)
            records = [
                {"date": row.Index.isoformat(), "value": row.value}
                for row in df.itertuples()
                if row.value is not None
            ]
            out_path = snapshot_dir / f"{series_id}.json"
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"  ✓ {series_id}: {len(records)} observations")
        except Exception as e:
            print(f"  ✗ {series_id}: {e}")

    print("\nSnapshot complete. The agent will use these files for all experiments.")
    print("To add more series, edit SERIES in this file and rerun.")


if __name__ == "__main__":
    main()
