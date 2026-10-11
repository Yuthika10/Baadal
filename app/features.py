"""Feature engineering shared by training (historical data) and serving (live data).

Both paths call `compute_features` on an hourly Open-Meteo dataframe, so the model
always sees features built exactly the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Hourly variables requested from Open-Meteo (archive and forecast APIs share names).
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "precipitation",
]

FEATURES = [
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "humidity",
    "temp",
    "dew_spread",
    "pressure",
    "pressure_change_3h",
    "wind_speed",
    "rain_last_3h",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

LABEL = "rain_next"


def compute_features(df: pd.DataFrame, with_label: bool = True) -> pd.DataFrame:
    """Turn an hourly dataframe (columns = HOURLY_VARS + 'time') into model features."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    out = pd.DataFrame({"time": df["time"]})
    out["cloud_low"] = df["cloud_cover_low"]
    out["cloud_mid"] = df["cloud_cover_mid"]
    out["cloud_high"] = df["cloud_cover_high"]
    out["humidity"] = df["relative_humidity_2m"]
    out["temp"] = df["temperature_2m"]
    out["dew_spread"] = df["temperature_2m"] - df["dew_point_2m"]
    out["pressure"] = df["surface_pressure"]
    out["pressure_change_3h"] = df["surface_pressure"] - df["surface_pressure"].shift(3)
    out["wind_speed"] = df["wind_speed_10m"]
    # rain over the 3 hours up to and including now
    out["rain_last_3h"] = df["precipitation"].rolling(3, min_periods=3).sum()

    hour = out["time"].dt.hour
    month = out["time"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    if with_label:
        h = config.HORIZON_HOURS
        # precipitation in the next h hours (t+1 .. t+h)
        future = sum(df["precipitation"].shift(-k) for k in range(1, h + 1))
        out["rain_next_mm"] = future
        out[LABEL] = (future >= config.RAIN_THRESHOLD_MM).astype("float")
        out.loc[future.isna(), LABEL] = np.nan

    return out


# ---------------------------------------------------------------------------
# Turning what Gemma saw in the photo into cloud-layer features
# ---------------------------------------------------------------------------

CLOUD_GENERA = [
    "clear",
    "cirrus",
    "cirrocumulus",
    "cirrostratus",
    "altocumulus",
    "altostratus",
    "nimbostratus",
    "stratocumulus",
    "stratus",
    "cumulus",
    "cumulonimbus",
]

GENUS_LAYER = {
    "cirrus": "high",
    "cirrocumulus": "high",
    "cirrostratus": "high",
    "altocumulus": "mid",
    "altostratus": "mid",
    "nimbostratus": "mid",   # thick, spans low-mid; treated as mid + low below
    "stratocumulus": "low",
    "stratus": "low",
    "cumulus": "low",
    "cumulonimbus": "low",   # base is low, tower reaches high
}


def apply_sky_observation(row: dict, sky: dict) -> dict:
    """Overwrite model-estimated cloud layers with what the observer actually saw.

    `row` holds live features (from weather data). `sky` is the validated vision output:
    {"genus": str, "coverage_oktas": int, "vertical_development": bool}.
    The photo is treated as ground truth for the layer it shows. Layers it cannot
    show (hidden above a thick low deck) keep the weather-model estimate.
    """
    row = dict(row)
    genus = sky.get("genus", "clear")
    pct = max(0, min(8, int(sky.get("coverage_oktas", 0)))) / 8 * 100

    if genus == "clear" or pct == 0:
        row["cloud_low"] = min(row["cloud_low"], 10)
        row["cloud_mid"] = min(row["cloud_mid"], 10)
        row["cloud_high"] = min(row["cloud_high"], 10)
        return row

    layer = GENUS_LAYER.get(genus, "low")
    row[f"cloud_{layer}"] = pct

    if genus == "nimbostratus":
        row["cloud_low"] = max(row["cloud_low"], pct)
    if genus == "cumulonimbus" or sky.get("vertical_development"):
        row["cloud_low"] = max(row["cloud_low"], pct)
        row["cloud_mid"] = max(row["cloud_mid"], 60.0)
    # If you can see high cloud through the gaps, the low deck can't be thick.
    if layer == "high" and pct >= 50:
        row["cloud_low"] = min(row["cloud_low"], 100 - pct)
    return row
