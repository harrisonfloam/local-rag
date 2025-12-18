FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /local-rag

COPY . .

# ENV UV_COMPILE_BYTECODE=1
RUN uv sync
ENV PATH="/local-rag/.venv/bin:$PATH"

CMD ["python", "-m", "app.api.main"]
