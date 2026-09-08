import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent
DECK_HTML = BASE_DIR / "static" / "presentation_v1.html"
DECK_V2_HTML = BASE_DIR / "static" / "presentation_v2.html"
DECK_V3_HTML = BASE_DIR / "static" / "presentation_v3.html"

CHART_FILES = {
    "__CHART_EXPECTATION__": "dev_expectation_vs_realized.png",
    "__CHART_PERFORMANCE__": "dev_cumulative_performance.png",
    "__CHART_FUNNEL__": "dev_funnel_sizing.png",
}

_CANDIDATES = (
    BASE_DIR.parent.parent / "experiments" / "results" / "7",
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
    return DECK_V3_HTML.read_text(encoding="utf-8")


@app.get("/v1", response_class=HTMLResponse)
def deck_v1() -> str:
    return build_deck()


@app.get("/v2", response_class=HTMLResponse)
def deck_v2() -> str:
    return DECK_V2_HTML.read_text(encoding="utf-8")


@app.get("/v3", response_class=HTMLResponse)
def deck_v3() -> str:
    return DECK_V3_HTML.read_text(encoding="utf-8")


@app.get("/checklist", response_class=HTMLResponse)
def checklist() -> str:
    return (BASE_DIR / "static" / "statarb_checklist.html").read_text(encoding="utf-8")


@app.get("/report/v3", response_class=HTMLResponse)
def report_v3() -> str:
    return (BASE_DIR / "static" / "report_v3.html").read_text(encoding="utf-8")


@app.get("/report/v3.2", response_class=HTMLResponse)
def report_v32() -> str:
    return (BASE_DIR / "static" / "report_v3.2.html").read_text(encoding="utf-8")


@app.get("/distress/v3", response_class=HTMLResponse)
def distress_v3() -> str:
    return (BASE_DIR / "static" / "distress_report.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
