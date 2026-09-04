import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent
DECK_HTML = BASE_DIR / "static" / "presentation.html"

CHART_FILES = {
    "__CHART_EXPECTATION__": "dev_expectation_vs_realized.png",
    "__CHART_PERFORMANCE__": "dev_cumulative_performance.png",
    "__CHART_FUNNEL__": "dev_funnel_sizing.png",
}

_CANDIDATES = (
    BASE_DIR / "results" / "7",
    BASE_DIR.parent.parent / "results" / "7",
)
RESULTS_DIR = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])

app = FastAPI(title="Stat-Arb Presentation Deck")


def build_deck() -> str:
    html = DECK_HTML.read_text(encoding="utf-8")
    for placeholder, filename in CHART_FILES.items():
        path = RESULTS_DIR / filename
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            html = html.replace(placeholder, f"data:image/png;base64,{b64}")
        else:
            html = html.replace(placeholder, "")
    return html


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return build_deck()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
