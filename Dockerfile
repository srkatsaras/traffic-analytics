FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# Libraries needed by the OpenCV wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Include weights in the image so jobs do not need to download them.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY api/ ./api/
COPY scripts/ ./scripts/


FROM base AS test

COPY requirements-dev.txt .
RUN python -m pip install --no-cache-dir -r requirements-dev.txt

COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-q"]


FROM base AS runtime

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
