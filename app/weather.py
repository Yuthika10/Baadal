"""Live conditions from Open-Meteo (free, no key). Uses the same features as training."""

from __future__ import annotations

import httpx
import pandas as pd

from .features import FEATURES, HOURLY_VARS, compute_features

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def live_features(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "current": "temperature_2m",
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "auto",
    }
    r = httpx.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    hourly = pd.DataFrame(data["hourly"])
    feats = compute_features(hourly, with_label=False)

    now = pd.to_datetime(data["current"]["time"]).floor("h")
    idx = feats.index[feats["time"] == now]
    if len(idx) == 0:
        raise RuntimeError("Current hour not found in Open-Meteo response")
    i = int(idx[0])
    row = {f: float(feats.loc[i, f]) for f in FEATURES}

    # What the weather model itself forecasts for the next 3 hours (for comparison only)
    precip = hourly["precipitation"].astype(float)
    model_next_3h = float(precip.iloc[i + 1 : i + 4].sum())

    return {
        "row": row,
        "local_time": str(now),
        "timezone": data.get("timezone"),
        "model_forecast_mm_3h": round(model_next_3h, 2),
    }
