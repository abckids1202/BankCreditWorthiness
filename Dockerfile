FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY credit_simulator ./credit_simulator
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir .
COPY dashboard.py ./dashboard.py
EXPOSE 8000 8501

