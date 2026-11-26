FROM python:3.13-slim

WORKDIR /app

# Install uv. Pinned by digest, so a rebuild gets exactly this image even if the tag is moved.
# To upgrade, change both the tag and the digest (`docker buildx imagetools inspect ghcr.io/astral-sh/uv:<tag>`).
COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /uvx /bin/

# Unprivileged user the app runs as. Its home (for library caches such as Hugging Face models)
# is outside /app, so the code directory can stay read-only for it.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application
COPY . .

# Sync again to install the project itself
RUN uv sync --frozen --no-dev

# Run from the virtual environment directly. The app user cannot write to it, so tell uv never to
# re-sync it: otherwise `docker compose exec backend uv run ...` would try to install dev packages.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1

# Set environment variable for persistent database storage
ENV DB_DATA_DIR=/app/db_data

# /app (the code and the virtual environment) stays root-owned, so the app cannot change or
# replace its own code. It writes only to the database and uploads directories, its home
# directory, and the system temp directory (downloaded documents).
# docker-compose.yaml hands a bind-mounted ./db_data to this user before start-up (the
# db-data-permissions service). Without Compose, run: sudo chown -R 10001:10001 db_data
ENV HOME=/home/app
RUN mkdir -p /app/db_data /app/uploads \
 && chown -R app:app /app/db_data /app/uploads

# Numeric IDs, so orchestrators can verify the container does not run as root
USER 10001:10001

EXPOSE 8000

# "/" needs no API key; the app does not start at all while MongoDB is unreachable
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
