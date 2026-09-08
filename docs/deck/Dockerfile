# Build context: repo root (see docs/docker-compose.yml; run via `docker compose up --build`).
FROM python:3.11-slim

WORKDIR /app

COPY docs/deck/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs/deck/main.py .
COPY docs/deck/static ./static

# Independent chart images and CSV data from experiments/results/7/
COPY experiments/results/7/ ./results/7/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]