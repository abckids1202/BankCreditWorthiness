FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY credit_simulator ./credit_simulator
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir .
COPY dashboard.py ./dashboard.py
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

