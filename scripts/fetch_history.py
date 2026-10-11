"""Download hourly weather history from Open-Meteo and build the TabPFN training table.

Run on your own computer (needs internet):

    python scripts/fetch_history.py --lat 19.076 --lon 72.8777 --years 2

Writes data/raw_hourly.csv and data/training.csv. Commit data/training.csv so the
deployed app can train TabPFN without calling Open-Meteo at startup.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.features import FEATURES, HOURLY_VARS, LABEL, compute_features  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    r = httpx.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    return pd.DataFrame(hourly)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, default=config.DEFAULT_LAT)
    p.add_argument("--lon", type=float, default=config.DEFAULT_LON)
    p.add_argument("--years", type=int, default=2)
    p.add_argument("--out", default=str(config.TRAINING_CSV))
    args = p.parse_args()

    # The archive lags a few days behind today.
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=365 * args.years)

    print(f"Fetching {start} .. {end} for ({args.lat}, {args.lon}) ...")
    raw = fetch(args.lat, args.lon, start, end)
    raw_path = Path(args.out).with_name("raw_hourly.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    print(f"  {len(raw):,} hourly rows -> {raw_path}")

    feats = compute_features(raw, with_label=True)
    feats = feats.dropna(subset=FEATURES + [LABEL])
    feats[LABEL] = feats[LABEL].astype(int)
    feats.to_csv(args.out, index=False)

    rate = feats[LABEL].mean()
    print(f"  {len(feats):,} training rows -> {args.out}")
    print(f"  rain-in-next-{config.HORIZON_HOURS}h rate: {rate:.1%}")


if __name__ == "__main__":
    main()
