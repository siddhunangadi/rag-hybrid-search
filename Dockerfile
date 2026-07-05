FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

EXPOSE 8000

CMD uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
