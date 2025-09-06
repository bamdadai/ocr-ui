# ====================================================================
# Final, Production-Ready Dockerfile (CUDA-enabled, single-stage)
# ====================================================================

# Base image with CUDA 12.4 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Allow switching requirement set at build-time (api.txt, worker.txt, base.txt)
ARG REQ_FILE=requirements/base.txt

# 1) System dependencies
# Remove NVIDIA apt repos (can be blocked or cause 403 in some environments) before updating
RUN rm -f /etc/apt/sources.list.d/cuda* /etc/apt/sources.list.d/nvidia* || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    poppler-utils \
    libgl1-mesa-glx \
    fonts-dejavu-core \
    ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Create Python venv inside final image and install dependencies
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt
COPY requirements/ ./requirements/
RUN python3 -m venv "$VIRTUAL_ENV" \
    && . "$VIRTUAL_ENV/bin/activate" \
    && pip install --upgrade pip \
    && pip install --timeout=600 -r ${REQ_FILE} \
    && pip install --default-timeout=100 --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0

# 3) Create non-root user and copy application code
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser master_config.json ./master_config.json

# Ensure project root is importable for Celery/Gunicorn processes
ENV PYTHONPATH=/home/appuser

# 4) Switch to non-root user and expose port
USER appuser
EXPOSE 8084

# 5) Default command (overridden by compose for workers)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8084"]
