FROM python:3.10-slim

    # Install dependencies and check espeak location
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

    # Install UV using the installer script
    RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/ && \
        mv /root/.local/bin/uvx /usr/local/bin/

    # Install Rust for sudachipy and other potential Rust-based dependencies
    RUN apt-get update && apt-get install -y curl && \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

    # Add the Rust cargo bin directory to the PATH for future layers
    ENV PATH="/root/.cargo/bin:$PATH" 

    # Create non-root user and set up directories and permissions
    RUN useradd -m -u 1000 appuser && \
        mkdir -p /app/api/src/models/v1_0 && \
        chown -R appuser:appuser /app

    USER appuser
    WORKDIR /app

    # Copy dependency files
    COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

    # Install dependencies
    # The PATH now includes the Rust compiler directory when this RUN command executes
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv venv --python 3.10 && \
        uv sync --extra cpu

    # Copy project files including models
    COPY --chown=appuser:appuser api ./api
    COPY --chown=appuser:appuser web ./web
    COPY --chown=appuser:appuser docker/scripts/ ./
    RUN chmod +x ./entrypoint.sh

    # Set environment variables
    ENV PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app:/app/api \
        # You already have this below, make sure it's consistent
        # PATH="/app/.venv/bin:$PATH" \
        UV_LINK_MODE=copy \
        USE_GPU=false \
        PHONEMIZER_ESPEAK_PATH=/usr/bin \
        PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
        ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

    # Note: You have two ENV PATH lines. Let's combine them or ensure the later one correctly appends.
    # For simplicity, let's modify the existing main ENV block later to include the Rust path.
    # For now, adding this separate ENV should work for the build step.

    ENV DOWNLOAD_MODEL=true
    # Download model if enabled
    RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
        python download_model.py --output api/src/models/v1_0; \
        fi

    ENV DEVICE="cpu"
    # Run FastAPI server through entrypoint.sh
    CMD ["./entrypoint.sh"]
