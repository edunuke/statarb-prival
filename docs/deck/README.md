# Stat-Arb Presentation Deck — FastAPI microservice

A slim FastAPI microservice that serves a **single-file, self-contained Bespoke.js
presentation deck** summarising the `statarb_1.0.ipynb` pipeline: the Ghost →
Eligible → Live architecture, a Mermaid diagram of the full pipeline, the live
rebalance log, the diagnostic tearsheet image and the OOS walk-forward results
(2026-02-25 → 2026-08-25).

## Layout

```
docs/
├── imgs/results.png            # Diagnostic tearsheet rendered by the notebook
└── deck/
    ├── main.py                 # FastAPI app (root route serves the deck)
    ├── static/presentation.html# Bespoke.js deck (Mermaid via CDN)
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
docker-compose.yml           # Compose file: build + run the deck service
```

The deck HTML is a single file: Bespoke.js and Mermaid are loaded from
jsDelivr, and the tearsheet PNG is **base64-inlined** by `main.py` at request
time, so the served page has no extra image requests.

## Run locally

```
cd dev/docs
pip install -r deck/requirements.txt
uvicorn deck.main:app --port 8000
```

Open http://localhost:8000 — navigate with arrow keys / swipe. The progress
bar and slide counter are built in.

## Run with Docker Compose

Build and start the deck server with a single command (run from the repo root or
from `docs/` — the compose file sets the repo root as the build context so the
image picks up `results/7/` chart PNGs):

```
# from docs/
docker compose up --build -d
```

Open http://localhost:8080. The container publishes port 8080, restarts
automatically, and serves the deck on `/` with a `/health` liveness probe.

Stop and remove the container with `docker compose down`.

## Endpoints

| Route        | Response                                            |
|--------------|-----------------------------------------------------|
| `/`          | The presentation deck (single HTML file)            |
| `/api/info`  | JSON metadata used by the deck (run window, trades) |
| `/health`    | Liveness probe                                      |