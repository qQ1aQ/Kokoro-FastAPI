FROM python:3.10-slim

# STAGE 1: Build dependencies as root

# Install system dependencies
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

# Add Rust's cargo bin to the PATH for this stage (while still root)
ENV PATH="/root/.cargo/bin:$PATH"

# Create application and model directory (still as root)
RUN mkdir -p /app/api/src/models/v1_0
WORKDIR /app

# Copy dependency file (pyproject.toml)
COPY pyproject.toml ./pyproject.toml

# Install Python dependencies using uv (as root)
# The PATH (set by the ENV instruction above) now includes /root/.cargo/bin,
# so sudachipy should compile correctly.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 && \
    uv sync --extra cpu

# STAGE 2: Setup appuser, copy application files, and run as appuser

# Create non-root user
# This is done AFTER installing dependencies that might need root or go into root's home.
RUN useradd -m -s /bin/bash -u 1000 appuser

# Change ownership of the /app directory (including .venv and models dir) to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app # Current directory is already /app, this reinforces context for appuser

# Copy project files (as appuser, using --chown for safety from build context)
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh

# Set final environment variables for the application (as appuser)
# This PATH prepends the virtual environment's bin to appuser's default PATH.
# Rust tools are not needed in the PATH for appuser at runtime.
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

# Download model (as appuser)
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Define the entrypoint (will run as appuser)
CMD ["./entrypoint.sh"]
