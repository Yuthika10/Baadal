"""Baadal API + static frontend.

Run locally:   uvicorn app.main:app --reload
On Render:     uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, gemma, weather
from .features import CLOUD_GENERA, HOURLY_VARS, apply_sky_observation
from .predictor import predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("cloud-oracle")

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fit TabPFN in the background so the server answers health checks immediately.
    def _fit():
        try:
            predictor.ensure_ready()
        except Exception as exc:
            log.error("predictor failed to start: %s", exc)

    threading.Thread(target=_fit, daemon=True).start()
    yield


app = FastAPI(title="Baadal", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "predictor": predictor.status(),
        "gemma": {"backend": config.GEMMA_BACKEND, "model": gemma.model_label()},
        "horizon_hours": config.HORIZON_HOURS,
        "threshold_mm": config.RAIN_THRESHOLD_MM,
    }


@app.get("/api/config")
def client_config() -> dict:
    return {
        "default_lat": config.DEFAULT_LAT,
        "default_lon": config.DEFAULT_LON,
        "default_place": config.DEFAULT_PLACE,
        "genera": CLOUD_GENERA,
        "weather_hourly": HOURLY_VARS,
        "horizon_hours": config.HORIZON_HOURS,
        "gemma": gemma.model_label(),
    }


def _save_reading(record: dict) -> None:
    config.READINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.READINGS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _load_readings() -> list[dict]:
    if not config.READINGS_PATH.exists():
        return []
    rows = {}
    for line in config.READINGS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "outcome":
            if rec["reading_id"] in rows:
                rows[rec["reading_id"]]["outcome"] = rec["rained"]
        else:
            rows[rec["id"]] = rec
    return sorted(rows.values(), key=lambda r: r["created_at"], reverse=True)


@app.post("/api/read-sky")
def read_sky(
    image: UploadFile | None = File(default=None),
    lat: float = Form(default=None),
    lon: float = Form(default=None),
    manual_genus: str | None = Form(default=None),
    manual_oktas: int | None = Form(default=None),
    manual_vertical: bool = Form(default=False),
    weather_json: str | None = Form(default=None),
) -> dict:
    t0 = time.time()
    lat = config.DEFAULT_LAT if lat is None else lat
    lon = config.DEFAULT_LON if lon is None else lon

    # 1. Live atmosphere (the browser usually sends Open-Meteo data it fetched itself)
    payload = None
    if weather_json and len(weather_json) < 600_000:
        try:
            payload = json.loads(weather_json)
        except json.JSONDecodeError:
            payload = None
    try:
        live = weather.live_features(lat, lon, payload)
    except Exception as exc:
        raise HTTPException(
            502,
            "Couldn't get live weather right now (the free weather service is busy). "
            f"Please try again in a minute. [{exc}]",
        )
    row = live["row"]

    # 2. What the sky looks like: Gemma, or the user's own call
    sky, sky_error = None, None
    if manual_genus:
        if manual_genus not in CLOUD_GENERA:
            raise HTTPException(400, f"manual_genus must be one of {CLOUD_GENERA}")
        sky = {
            "is_sky": True,
            "genus": manual_genus,
            "coverage_oktas": 0 if manual_genus == "clear" else (manual_oktas if manual_oktas is not None else 5),
            "vertical_development": manual_vertical or manual_genus == "cumulonimbus",
            "darkness": "medium",
            "confidence": 1.0,
            "visual_clues": "chosen by the observer",
            "source": "you",
        }
    elif image is not None:
        raw = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 12 MB)")
        try:
            sky = gemma.read_sky(raw)
            sky["source"] = gemma.model_label()
        except gemma.GemmaError as exc:
            sky_error = str(exc)
            log.warning("vision failed: %s", exc)

    # 3. Predict: weather data alone vs. weather data + what you saw
    try:
        rows = [row]
        if sky and sky.get("is_sky", True):
            rows.append(apply_sky_observation(row, sky))
        probs = predictor.predict(rows)
    except Exception as exc:
        raise HTTPException(503, f"Prediction model not ready: {exc}")
    p_data = probs[0]
    p_eyes = probs[1] if len(probs) > 1 else None
    headline = p_eyes if p_eyes is not None else p_data

    # 4. Explain
    facts = {
        "cloud_type": sky["genus"] if sky else "unknown (no photo)",
        "cloud_cover_oktas": sky["coverage_oktas"] if sky else None,
        "towering_growth": sky["vertical_development"] if sky else None,
        "rain_chance_pct": round(headline * 100),
        "rain_chance_from_weather_data_only_pct": round(p_data * 100),
        "humidity_pct": round(row["humidity"]),
        "pressure_change_3h_hpa": round(row["pressure_change_3h"], 1),
        "temp_minus_dewpoint_c": round(row["dew_spread"], 1),
        "rain_last_3h_mm": round(row["rain_last_3h"], 1),
        "horizon_hours": config.HORIZON_HOURS,
    }
    lesson = gemma.explain(facts) if sky else gemma.template_explanation({**facts, "cloud_type": "clear"})

    reading = {
        "id": uuid.uuid4().hex[:10],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "local_time": live["local_time"],
        # rounded to ~1 km: enough for weather, not enough to pinpoint a home
        "lat": round(lat, 2),
        "lon": round(lon, 2),
        "sky": sky,
        "p_with_eyes": p_eyes,
        "p_data_only": p_data,
        "model_forecast_mm_3h": live["model_forecast_mm_3h"],
        "predictor": predictor.backend,
    }
    _save_reading(reading)

    return {
        "reading_id": reading["id"],
        "rain_probability": headline,
        "p_with_eyes": p_eyes,
        "p_data_only": p_data,
        "sky": sky,
        "sky_error": sky_error,
        "lesson": lesson,
        "conditions": {k: round(v, 2) for k, v in row.items()},
        "model_forecast_mm_3h": live["model_forecast_mm_3h"],
        "local_time": live["local_time"],
        "timezone": live["timezone"],
        "weather_source": live.get("source"),
        "horizon_hours": config.HORIZON_HOURS,
        "backends": {"predictor": predictor.status(), "gemma": gemma.model_label()},
        "elapsed_s": round(time.time() - t0, 2),
    }


class Outcome(BaseModel):
    reading_id: str
    rained: bool


@app.post("/api/log")
def log_outcome(outcome: Outcome) -> dict:
    ids = {r["id"] for r in _load_readings()}
    if outcome.reading_id not in ids:
        raise HTTPException(404, "Unknown reading id")
    _save_reading({
        "type": "outcome",
        "reading_id": outcome.reading_id,
        "rained": outcome.rained,
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return {"ok": True}


@app.get("/api/readings")
def readings(limit: int = 50) -> dict:
    rows = _load_readings()[:limit]
    scored = [r for r in rows if "outcome" in r and r.get("p_with_eyes") is not None]
    hits = sum(1 for r in scored if (r["p_with_eyes"] >= 0.5) == r["outcome"])
    return {"readings": rows, "scored": len(scored), "correct": hits}