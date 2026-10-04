# Linux is required by shockbench-flow 0.1.2's subprocess runner.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    SBF_CACHE_DIR=/cache/shockbench-flow \
    PYTHONUNBUFFERED=1
WORKDIR /workspace

# Keep dependency downloads cached when only source or agents change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked
COPY agents ./agents
COPY examples ./examples
COPY tests ./tests

ENTRYPOINT ["uv", "run", "--no-sync", "sbf"]
CMD ["--help"]
