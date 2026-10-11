"""Gemma's two jobs: read the sky from a photo, and explain the forecast in plain words.

Backends:
  ollama - local Gemma 4 (default model gemma4:e4b). Photo never leaves your machine.
  google - Gemma 4 served by the Gemini API (for the hosted demo).
  off    - no model; the UI asks the user to pick the cloud type.

Every vision answer is validated against a fixed schema. A malformed answer gets one
repair attempt; if that fails we return an explicit error instead of guessing.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re

import httpx
from PIL import Image, ImageOps

from . import config
from .features import CLOUD_GENERA

log = logging.getLogger("cloud-oracle.gemma")


class GemmaError(RuntimeError):
    pass


VISION_PROMPT = f"""You are a meteorological observer. Look at this photo of the sky.
Reply with ONLY a JSON object, no prose, no code fences, using exactly these keys:

{{
  "is_sky": true or false,            // false if the photo does not mainly show the sky
  "genus": one of {json.dumps(CLOUD_GENERA)},   // the DOMINANT cloud type, "clear" if none
  "coverage_oktas": integer 0-8,      // how much of the visible sky is covered (8 = overcast)
  "vertical_development": true or false,  // tall, towering, cauliflower-like growth?
  "darkness": "light" | "medium" | "dark", // how dark the cloud bases look
  "confidence": number 0-1,           // how sure you are about the genus
  "visual_clues": short string        // under 20 words: what you saw that decided it
}}

Use WMO cloud genera. Prefer "cumulus" for puffy separate clouds with flat bases,
"cumulonimbus" only for very tall towers or anvil tops, "stratus" for a featureless
grey sheet, "altostratus" when the sun looks like it is behind frosted glass,
"cirrus" for thin wispy streaks high up."""

EXPLAIN_PROMPT = """You are a friendly sky-reading teacher. A person just looked up at the sky.
Facts:
{facts}

Reply with ONLY a JSON object:
{{
  "explanation": "max 2 short sentences: why the rain chance is what it is, naming the cloud and one weather clue",
  "spot_it_tip": "1 sentence: one visual sign this person can look for themselves next time, so they need the app less"
}}
Do not invent numbers that are not in the facts. Plain words, no jargon without explaining it."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def prepare_image(raw: bytes, max_side: int = 1024) -> bytes:
    """Re-encode as a small JPEG. This also strips EXIF (GPS, device) before any model sees it."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise GemmaError(f"Could not read the image: {exc}") from exc
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def extract_json(text: str, required: tuple[str, ...] = ()) -> dict:
    """Pull the JSON object out of a model reply.

    Models often wrap the object in prose, code fences, a repeated draft, or a
    trailing note. We scan every '{', decode the first complete object from there,
    and return the first one that has the keys we need.
    """
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    found = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if all(k in obj for k in required):
                return obj
            found.append(obj)
    if found and not required:
        return max(found, key=len)
    raise ValueError(f"no JSON object with keys {list(required)} in model reply")


def validate_sky(d: dict) -> dict:
    genus = str(d.get("genus", "")).strip().lower()
    if genus not in CLOUD_GENERA:
        raise ValueError(f"genus '{genus}' is not a WMO genus")
    oktas = int(round(float(d.get("coverage_oktas"))))
    if not 0 <= oktas <= 8:
        raise ValueError("coverage_oktas out of range")
    conf = float(d.get("confidence", 0.5))
    darkness = str(d.get("darkness", "medium")).lower()
    if darkness not in ("light", "medium", "dark"):
        darkness = "medium"
    if genus == "clear":
        oktas = min(oktas, 1)
    return {
        "is_sky": bool(d.get("is_sky", True)),
        "genus": genus,
        "coverage_oktas": oktas,
        "vertical_development": bool(d.get("vertical_development", False)),
        "darkness": darkness,
        "confidence": max(0.0, min(1.0, conf)),
        "visual_clues": str(d.get("visual_clues", ""))[:160],
    }


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _ollama(prompt: str, image_jpeg: bytes | None = None) -> str:
    body = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    if image_jpeg:
        body["images"] = [base64.b64encode(image_jpeg).decode()]
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/generate", json=body, timeout=config.GEMMA_TIMEOUT_S)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise GemmaError(f"Ollama call failed ({exc}). Is `ollama serve` running and "
                         f"`ollama pull {config.OLLAMA_MODEL}` done?") from exc
    return r.json().get("response", "")


GEMINI_BASE = "https://generativelanguage.googleapis.com"


def _google_upload(image_jpeg: bytes) -> str:
    """Files API upload, used if inline image data is rejected. Returns file_uri."""
    headers = {
        "x-goog-api-key": config.GEMINI_API_KEY,
        "X-Goog-Upload-Protocol": "raw",
        "Content-Type": "image/jpeg",
    }
    r = httpx.post(f"{GEMINI_BASE}/upload/v1beta/files", headers=headers,
                   content=image_jpeg, timeout=config.GEMMA_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["file"]["uri"]


def _google(prompt: str, image_jpeg: bytes | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise GemmaError("GEMINI_API_KEY is not set")
    url = f"{GEMINI_BASE}/v1beta/models/{config.GEMMA_API_MODEL}:generateContent"
    headers = {"x-goog-api-key": config.GEMINI_API_KEY}

    def call(image_part):
        parts = ([image_part] if image_part else []) + [{"text": prompt}]
        body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.1}}
        return httpx.post(url, headers=headers, json=body, timeout=config.GEMMA_TIMEOUT_S)

    image_part = None
    if image_jpeg:
        image_part = {"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(image_jpeg).decode()}}
    try:
        r = call(image_part)
        if r.status_code == 400 and image_jpeg:
            # Some models only accept uploaded files - retry through the Files API.
            uri = _google_upload(image_jpeg)
            r = call({"file_data": {"mime_type": "image/jpeg", "file_uri": uri}})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")[:300]
        raise GemmaError(f"Gemini API call failed: {exc} {detail}") from exc
    data = r.json()
    try:
        # Skip "thought" parts: thinking models may include draft JSON there.
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError) as exc:
        raise GemmaError(f"Unexpected Gemini API response: {str(data)[:300]}") from exc


def _generate(prompt: str, image_jpeg: bytes | None = None) -> str:
    backend = config.GEMMA_BACKEND
    if backend == "ollama":
        return _ollama(prompt, image_jpeg)
    if backend == "google":
        return _google(prompt, image_jpeg)
    raise GemmaError("Gemma is switched off (GEMMA_BACKEND=off)")


def model_label() -> str:
    if config.GEMMA_BACKEND == "ollama":
        return f"{config.OLLAMA_MODEL} via Ollama (local)"
    if config.GEMMA_BACKEND == "google":
        return f"{config.GEMMA_API_MODEL} via Gemini API"
    return "off"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def read_sky(image_raw: bytes) -> dict:
    jpeg = prepare_image(image_raw)
    text = _generate(VISION_PROMPT, jpeg)
    try:
        return validate_sky(extract_json(text, ("genus",)))
    except Exception as first_error:
        log.warning("vision output invalid (%s); attempting one repair", first_error)
        repair = (
            "Rewrite the following into ONLY a valid JSON object with keys is_sky, genus, "
            f"coverage_oktas, vertical_development, darkness, confidence, visual_clues. genus must be one of "
            f"{json.dumps(CLOUD_GENERA)}.\n\n{text[:2000]}"
        )
        try:
            return validate_sky(extract_json(_generate(repair), ("genus",)))
        except Exception as exc:
            raise GemmaError(f"Gemma's answer could not be validated: {exc}") from exc


def explain(facts: dict) -> dict:
    """Two-sentence explanation. Falls back to a plain template (and says so) on failure."""
    try:
        text = _generate(EXPLAIN_PROMPT.format(facts=json.dumps(facts, indent=1)))
        d = extract_json(text, ("explanation", "spot_it_tip"))
        return {
            "explanation": str(d["explanation"])[:400],
            "spot_it_tip": str(d["spot_it_tip"])[:250],
            "source": "gemma",
        }
    except Exception as exc:
        log.warning("explanation fell back to template: %s", exc)
        return template_explanation(facts)


TIPS = {
    "cumulus": "Watch the tops: if puffy clouds keep growing taller through the afternoon, showers are getting closer.",
    "cumulonimbus": "A flat, spreading 'anvil' top means the storm is mature - rain is usually close behind.",
    "stratocumulus": "Lumpy grey rolls with blue gaps rarely rain much; if the gaps close and bases darken, take an umbrella.",
    "stratus": "A low featureless grey sheet brings drizzle at most - real rain needs thicker cloud.",
    "nimbostratus": "If you can't find the sun's position at all through a thick grey sky, steady rain is likely.",
    "altostratus": "A 'frosted-glass' sun that keeps fading is often the first sign of rain several hours out.",
    "altocumulus": "Mid-level rows of small cloudlets on a humid morning can mean storms later in the day.",
    "cirrus": "Wispy high streaks that thicken into a milky sheet over hours often signal weather arriving later.",
    "cirrocumulus": "Tiny high ripples ('mackerel sky') mean moisture aloft - watch whether they thicken.",
    "cirrostratus": "A halo ring around the sun or moon comes from high ice cloud and often precedes rain within a day.",
    "clear": "Clear skies with high humidity can still build afternoon clouds - look up again after lunch.",
}


def template_explanation(facts: dict) -> dict:
    genus = facts.get("cloud_type", "clear")
    pct = facts.get("rain_chance_pct", 0)
    level = "high" if pct >= 60 else "moderate" if pct >= 30 else "low"
    return {
        "explanation": f"The chance of rain looks {level} ({pct}%). It is based on {genus} cloud, "
                       f"{facts.get('humidity_pct')}% humidity and a pressure change of "
                       f"{facts.get('pressure_change_3h_hpa')} hPa over 3 hours.",
        "spot_it_tip": TIPS.get(genus, TIPS["clear"]),
        "source": "template (Gemma unavailable)",
    }