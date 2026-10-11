# Cloud Oracle ☁️

**Look up, take one photo, learn to read the sky.**
Gemma 4 (open-weight vision) reads the clouds, TabPFN (tabular foundation model) turns
what you saw plus live humidity and pressure into a rain probability for the next 3 hours,
and Gemma explains it in two sentences with one "spot it yourself next time" tip.

```
Sky photo ──► resized on phone (EXIF/GPS stripped) ──► Gemma 4 vision
                                                        → {genus, oktas, towering, confidence}
Live air   ──► Open-Meteo (humidity, pressure, 3h pressure change, dew point, wind)
                                │
             same feature code as training (app/features.py)
                                ▼
           TabPFN ──► P(rain in next 3 h)  [with your eyes]  vs  [weather data only]
                                ▼
           Gemma ──► 2-sentence explanation + 1 sky-reading tip
                                ▼
           You log "it rained / stayed dry" later → honest field score
```

## Project layout

| Path | What it does |
|---|---|
| `scripts/fetch_history.py` | Downloads 2 years of hourly weather for your location, writes `data/training.csv` |
| `scripts/evaluate.py` | TabPFN vs climatology, logistic regression, gradient boosting on a time split → `results/comparison.png` |
| `app/features.py` | One feature function used for both training and live data; maps cloud type → cloud-layer cover |
| `app/predictor.py` | TabPFN via `tabpfn-client` (hosted) or `tabpfn` (local). Falls back to a **clearly labelled** baseline only if TabPFN can't start |
| `app/gemma.py` | Gemma via Ollama (local) or Gemini API (hosted). Strict JSON schema, one repair attempt, explicit error otherwise |
| `app/weather.py` | Live Open-Meteo conditions (free, no key) |
| `app/main.py` | FastAPI: `/api/read-sky`, `/api/log`, `/api/readings`, `/api/health` |
| `app/static/index.html` | Mobile web app (camera, manual cloud picker, result, lesson, field log) |
| `render.yaml` | One-click Render Blueprint |

---

## Step 0 — Accounts and keys (10 min)

1. **TabPFN token**: sign in at https://platform.priorlabs.ai/account/api-keys and create an API key.
   Also claim the Hacktoberfest partner credits at https://hacktoberfest.com/my/promos.
2. **Gemini API key** (for the hosted demo's Gemma): https://aistudio.google.com/apikey
3. **Render** account (sign in with GitHub): https://render.com
4. **Ollama** on your laptop: https://ollama.com/download, then:
   ```bash
   ollama pull gemma4:e4b
   ```
5. **Python 3.11** and Git.

## Step 1 — Set up the project locally

```bash
cd cloud-oracle
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
```

## Step 2 — Build the training data

```bash
python scripts/fetch_history.py --lat 19.076 --lon 72.8777 --years 2
```

Use your own coordinates. This writes `data/training.csv` (~17,000 hourly rows) and prints
the rain rate. **Commit `data/training.csv`**: the deployed app trains TabPFN from it at startup.

How the label works: a row is "rain" if Open-Meteo's precipitation over the **next 3 hours**
is at least 0.2 mm. Features are cloud cover at low/mid/high levels, humidity, temperature,
temperature minus dew point, pressure, **3-hour pressure change**, wind, rain in the last
3 hours, and hour/month (cyclic).

## Step 3 — Prove TabPFN works (your partner-tech evidence)

```bash
export TABPFN_TOKEN=your_token        # Windows PowerShell: $env:TABPFN_TOKEN="your_token"
python scripts/evaluate.py
```

Trains on the oldest 80% of hours and tests on the most recent 20% (no peeking at the
future). You get `results/metrics.json` and `results/comparison.png`: put the chart in your
DEV post. Report the honest result even if gradient boosting is close; the "zero tuning,
one-line fit" angle is still a fair point for TabPFN.

## Step 4 — Run it locally with private, local Gemma

```bash
ollama serve                          # if not already running
export GEMMA_BACKEND=ollama
export TABPFN_TOKEN=your_token
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Laptop: open http://localhost:8000
- Check http://localhost:8000/api/health: `predictor.is_tabpfn` must be `true`.
- Phone on the same Wi-Fi: open `http://<your-laptop-ip>:8000`. Photo upload works; GPS
  won't (browsers only share location over HTTPS), so it uses `DEFAULT_LAT/LON`. That's fine
  for testing. The Render URL is HTTPS, so GPS works there.

Test with a few sky photos from your gallery first. If Gemma mislabels often, read
`visual_clues` in the response and tighten `VISION_PROMPT` in `app/gemma.py`.

## Step 5 — Deploy to Render

1. Create a **new GitHub repo** (must be new this week: challenge rule) and push:
   ```bash
   git init && git add . && git commit -m "Cloud Oracle"
   git branch -M main
   git remote add origin https://github.com/<you>/cloud-oracle.git
   git push -u origin main
   ```
2. Render dashboard → **New + → Blueprint** → pick the repo. It reads `render.yaml`.
3. When asked, paste **GEMINI_API_KEY** and **TABPFN_TOKEN**.
4. Wait for the build (2–4 min). Open `https://<name>.onrender.com/api/health` and confirm:
   - `predictor.backend` is `tabpfn-client` and `is_tabpfn` is `true`
   - `gemma.model` is `gemma-4-26b-a4b-it via Gemini API`
5. Open the site on your phone, allow location and camera, take a sky photo.

Why this split: Render's free tier can't host a vision model, so the hosted demo calls
Gemma through the Gemini API and TabPFN through Prior Labs' API. The **local mode** (Step 4)
is the private version: photo never leaves your laptop. Say this plainly in the post.

Free-tier notes: the service sleeps after inactivity (first load ~50 s, so open it before you
demo). `data/readings.jsonl` resets on redeploy, so screenshot your field log for the post.

## Step 6 — Take it outside (bonus points)

Do 3–4 readings at different times (ideally one cloudy afternoon). For each one:
photo of the sky → screenshot of the result → after 3 hours, tap **It rained / Stayed dry**.
The field log shows your score. Include misses: honesty reads well.

## Step 7 — Write the DEV post (weighted most)

Use the challenge template. Suggested beats:
1. **Hook**: your sky photo + "Cloud Oracle said 68%. Here's what happened."
2. **What I Built**: look up, one tap, learn to read the sky; screen time under 30 s.
3. **Demo**: Render link + 30–60 s screen recording.
4. **How I Built It**: diagram above; the proxy-label approach (photo → cloud layers,
   because no public "photo → rain" dataset exists); `comparison.png`; the
   "with your eyes vs weather data only" comparison.
5. **Why Open Innovation Matters**: local Gemma = photo stays on your device and works with
   bad signal; open weights = swap models, tune the prompt, run at $0; TabPFN = a strong
   tabular model with no training loop.
6. **Field test**: your logged readings, including wrong ones.
7. **Limitations**: Gemma can confuse similar clouds; reanalysis cloud layers are
   estimates; 3-hour rain is hard.
8. **Prize Categories**: Best Use of Gemma, Best Use of TabPFN, Best Use of Render.
   Add Best Use of Entire if you embed your agent session.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Footer says "baseline model — TabPFN unavailable" | `TABPFN_TOKEN` missing/invalid. `/api/health` → `predictor.error` says why |
| "Gemma couldn't read this photo" | Ollama not running / model not pulled, or `GEMINI_API_KEY` wrong. Use "Tell it what you see" meanwhile |
| Gemini API 404 for the model | Model ID changed: set `GEMMA_API_MODEL` to a current Gemma ID from https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api |
| Build fails on Render installing `antlr4`/`omegaconf` | Make sure `PYTHON_VERSION=3.11.11` is set (it is in `render.yaml`) |
| Slow first prediction | TabPFN fits in the background at startup; wait until the footer shows "TabPFN" |
| "Could not fetch live weather" | Open-Meteo briefly down; retry |

Commits after the deadline must be listed in this README (challenge rule).

## License
MIT. Weather data © Open-Meteo (CC BY 4.0).
