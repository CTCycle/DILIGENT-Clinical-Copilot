FROM ghcr.io/astral-sh/uv:0.8.22-python3.14-bookworm

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY DILIGENT ./DILIGENT

RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=3)"

CMD ["uv", "run", "python", "-m", "uvicorn", "DILIGENT.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
