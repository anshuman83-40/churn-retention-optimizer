FROM python:3.13-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY models/ models/
COPY reports/ reports/
COPY data/raw/ data/raw/
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
# Default: REST API. For the dashboard:
#   docker run -p 8501:8501 churn-optimizer streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
