"""Live conditions from Open-Meteo (free, no key). Uses the same features as training.

Open-Meteo rate-limits per IP address, and cloud hosts like Render share outbound IPs
with many other apps. So the browser fetches the forecast itself (from the user's own
IP) and sends the JSON along with the photo. The server only calls Open-Meteo as a
fallback, with a short cache and one retry.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from .features import FEATURES, HOURLY_VARS, compute_features

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_SECONDS = 20 * 60
_cache: dict[tuple, tuple[float, dict]] = {}


def forecast_params(lat: float, lon: float) -> dict:
    return {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": ",".join(HOURLY_VARS),
        "current": "temperature_2m",
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "auto",
    }


def features_from_payload(data: dict) -> dict:
    """Turn an Open-Meteo forecast response into one feature row for the current hour."""
    hourly_raw = data["hourly"]
    missing = [v for v in HOURLY_VARS + ["time"] if v not in hourly_raw]
    if missing:
        raise ValueError(f"weather data missing {missing}")
    hourly = pd.DataFrame(hourly_raw)
    feats = compute_features(hourly, with_label=False)

    now = pd.to_datetime(data["current"]["time"]).floor("h")
    idx = feats.index[feats["time"] == now]
    if len(idx) == 0:
        raise ValueError("current hour not found in weather data")
    i = int(idx[0])
    row = {f: float(feats.loc[i, f]) for f in FEATURES}
    if any(pd.isna(v) for v in row.values()):
        raise ValueError("weather data has gaps for the current hour")

    # What the weather model itself forecasts for the next 3 hours (for comparison only)
    precip = hourly["precipitation"].astype(float)
    model_next_3h = float(precip.iloc[i + 1 : i + 4].sum())

    return {
        "row": row,
        "local_time": str(now),
        "timezone": data.get("timezone"),
        "model_forecast_mm_3h": round(model_next_3h, 2),
    }


def _fetch(lat: float, lon: float) -> dict:
    params = forecast_params(lat, lon)
    last_exc = None
    for attempt in range(2):
        try:
            r = httpx.get(FORECAST_URL, params=params, timeout=20)
            if r.status_code == 429 and attempt == 0:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(1)
    raise RuntimeError(f"Open-Meteo unavailable from the server ({last_exc or 'rate limited'})")


def live_features(lat: float, lon: float, payload: dict | None = None) -> dict:
    # 1. Weather the browser already fetched (preferred: the user's own IP, no shared limit)
    if payload:
        try:
            out = features_from_payload(payload)
            out["source"] = "browser"
            return out
        except Exception:
            pass  # malformed or stale: fall back to the server fetch

    # 2. Server fetch, cached per ~1 km cell for 20 minutes
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        data = hit[1]
    else:
        data = _fetch(lat, lon)
        _cache[key] = (time.time(), data)
    out = features_from_payload(data)
    out["source"] = "server"
    return out