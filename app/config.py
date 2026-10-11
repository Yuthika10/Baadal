"""Central configuration. Every setting can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# --- Location used when the browser does not share GPS (change to your own) ---
DEFAULT_LAT = float(_env("DEFAULT_LAT", "19.0760"))
DEFAULT_LON = float(_env("DEFAULT_LON", "72.8777"))
DEFAULT_PLACE = _env("DEFAULT_PLACE", "Mumbai")

# --- Prediction target ---
HORIZON_HOURS = int(_env("HORIZON_HOURS", "3"))          # look-ahead window
RAIN_THRESHOLD_MM = float(_env("RAIN_THRESHOLD_MM", "0.2"))  # "it rained" if total >= this

# --- TabPFN ---
# auto -> tabpfn-client if TABPFN_TOKEN is set, else local tabpfn if installed, else baseline
PREDICTOR_BACKEND = _env("PREDICTOR_BACKEND", "auto")
TABPFN_TOKEN = os.environ.get("TABPFN_TOKEN", "")
MAX_TRAIN_ROWS = int(_env("MAX_TRAIN_ROWS", "3000"))
TRAINING_CSV = Path(_env("TRAINING_CSV", str(ROOT / "data" / "training.csv")))

# --- Gemma ---
# ollama  -> local Gemma through Ollama (private, offline-capable)
# google  -> Gemma served by the Gemini API (used for the hosted Render demo)
# off     -> no vision; the user picks the cloud type by hand
GEMMA_BACKEND = _env("GEMMA_BACKEND", "ollama")
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "gemma4:e4b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMMA_API_MODEL = _env("GEMMA_API_MODEL", "gemma-4-26b-a4b-it")
GEMMA_TIMEOUT_S = float(_env("GEMMA_TIMEOUT_S", "90"))

# --- Field log (JSON lines). On Render's free tier this file resets on redeploy. ---
READINGS_PATH = Path(_env("READINGS_PATH", str(ROOT / "data" / "readings.jsonl")))
