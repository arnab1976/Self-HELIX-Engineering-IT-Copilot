FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY common ./common
COPY lexora ./lexora
COPY nexus ./nexus
COPY configs ./configs
COPY data/sample ./data/sample
COPY scripts ./scripts
COPY clients/web-portal ./clients/web-portal

# Build synthetic data, train CPU baselines and the index at image build time (seconds)
RUN python scripts/generate_sample_data.py && python scripts/bootstrap.py

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
