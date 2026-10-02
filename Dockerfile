FROM python:3.13-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY web/ web/
COPY models/ models/
COPY reports/ reports/
COPY data/raw/ data/raw/
# Run the package from /app/src (not site-packages) so it finds models/, reports/ and data/ under /app.
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

EXPOSE 8000
# Serves the web dashboard at / and the REST API at /docs. Hosts like Render set $PORT.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
