# ====================================================================
# Final, Production-Ready Dockerfile (CUDA-enabled, single-stage)
# ====================================================================

# Base image with CUDA 12.4 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

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
RUN python3 -m venv "$VIRTUAL_ENV" && . "$VIRTUAL_ENV/bin/activate" && pip install --upgrade pip
RUN . "$VIRTUAL_ENV/bin/activate" && python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
RUN . "$VIRTUAL_ENV/bin/activate" && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
COPY requirements/ ./requirements/
RUN . "$VIRTUAL_ENV/bin/activate" && pip install --timeout=600 -r ${REQ_FILE}
RUN . "$VIRTUAL_ENV/bin/activate" && pip install paddleocr==3.2.0
WORKDIR /app
COPY  assets/ ./assets/
COPY  app/ ./app/
COPY  static/ ./static/
COPY  templates/ ./templates/
COPY  master_config.json ./master_config.json

# Ensure project root is importable for Celery/Gunicorn processes
ENV PYTHONPATH=/app
EXPOSE 8084

# 5) Default command (overridden by compose for workers)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8084"]
