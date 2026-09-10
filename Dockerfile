# syntax=docker/dockerfile:1
#
# The search server. Not the builder: that wants a GPU, Docling's model
# downloads and several gigabytes of VRAM, and it runs once per corpus rather
# than continuously.
#
# Two runtime targets, because what the server needs depends on where it
# embeds queries:
#
#   server-remote  EMBEDDING_BACKEND=openai. No torch, no transformers, no
#                  model weights -- the endpoint owns all of that.
#   server         EMBEDDING_BACKEND=local. Loads Qwen3-Embedding-0.6B in
#                  process, so it carries the CPU torch build and downloads
#                  the weights on first use.
#
# `server-remote` is the default target because it is the smaller of the two
# by a factor of 2.7 (905 MB against 2.43 GB) and holds no model. It is not the
# default because it is faster: the two do the same work in different places,
# so which one answers sooner is a question about the endpoint's machine and
# this one, not about the images.

ARG PYTHON_VERSION=3.13
# Pinned rather than floating: this decides how the lock file is interpreted.
ARG UV_VERSION=0.11.6

# --- a base with uv, shared by both dependency stages ----------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ARG UV_VERSION
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# --- dependencies ----------------------------------------------------------
#
# Each stage installs the manifest before the source, so editing a module does
# not re-resolve or re-download the tree. --no-install-project is what keeps
# the project itself out of that first layer.

FROM base AS deps-remote
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra qdrant
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra qdrant

FROM base AS deps-local
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra cpu --extra qdrant
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra cpu --extra qdrant

# --- runtime ---------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
WORKDIR /app

# Nothing writes inside the image. The two things that do change -- the
# database and, for the local target, the downloaded model -- are volumes.
#
# Both mount points are created here rather than left to Docker. A named
# volume inherits the ownership of the directory it covers *if that directory
# exists in the image*; if it does not, Docker creates it owned by root and
# the unprivileged process cannot write to it. That is how the model cache
# fails, and it fails at first download rather than at startup.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    REST_HOST=0.0.0.0 \
    REST_PORT=8001 \
    DB_FILE_PATH=/app/data/mcp_manual_walker.db \
    CHROMADB_PATH=/app/data/db/chroma_db \
    PDF_ROOT_DIR=/app/data/pdfs \
    HF_HOME=/home/app/.cache/huggingface

# 127.0.0.1 is right on a laptop and useless here: a server bound to loopback
# inside its own network namespace is reachable by nothing. HOST and REST_HOST
# above are the only settings this image overrides for that reason rather than
# for a path. Which of the two ports is published, and to what address, is the
# compose file's decision.

EXPOSE 8000 8001
USER app
VOLUME ["/app/data"]

# Proves the port is accepting connections. Deliberately not a request to
# /mcp: that transport expects a session, and a health check that has to
# speak the protocol correctly is a health check that breaks when the
# protocol moves. start-period covers loading the embedding model, which the
# local target does before it listens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os,socket; socket.create_connection(('127.0.0.1', int(os.environ['PORT'])), 4).close()"]

CMD ["python", "-m", "mcp_manual_walker.main"]

# --- the two published targets ---------------------------------------------

FROM runtime AS server-remote
COPY --from=deps-remote --chown=app:app /app/.venv /app/.venv
ENV EMBEDDING_BACKEND=openai

FROM runtime AS server
COPY --from=deps-local --chown=app:app /app/.venv /app/.venv
# float32 rather than the checkpoint's bfloat16: bfloat16 has no fast CPU
# kernels, and one query measured 0.86 s under it against 0.43 s under
# float32. It doubles the resident weights (1.11 GiB -> 2.22 GiB) and halves
# the latency, which is the right trade for a server that holds one model and
# answers one query at a time.
ENV EMBEDDING_BACKEND=local \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_DTYPE=float32
