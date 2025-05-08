FROM python:3.10-slim

# Install system dependencies for espeak, build tools, etc.
RUN apt-get update && apt-get install -y \
    espeak-ng \
    espeak-ng-data \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    g++ \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /usr/share/espeak-ng-data \
 && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/

# Install UV (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Install Rust
RUN apt-get update && apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# IMPORTANT: Add Rust's cargo bin to the PATH for subsequent RUN commands in this stage.
# This ensures 'rustc' is found when 'uv sync' builds Rust-dependent packages like sudachipy.
ENV PATH="/root/.cargo/bin:$PATH"

# Create non-root user and work directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files (pyproject.toml)
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install Python dependencies using uv.
# The PATH (set by the ENV instruction above) now includes /root/.cargo/bin.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra cpu

# Copy project files
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh

# Set final environment variables for the application.
# This will prepend /app/.venv/bin to the PATH that already includes /root/.cargo/bin from the ENV above.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=false \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    PATH="/app/.venv/bin:$PATH" \
    DEVICE="cpu" \
    DOWNLOAD_MODEL=true

# Download model (after all dependencies are installed and PATH includes .venv)
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Define the entrypoint
CMD ["./entrypoint.sh"]
