# One container: React build + FastAPI (frontend, /api, /media) + the job worker. Built for Hugging Face Spaces
# (Docker SDK, port 7860, runs as uid 1000) but works anywhere Docker runs.

FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npx vite build

FROM python:3.12-slim
# libGL / glib for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/app/data FRONTEND_DIST=/home/user/app/frontend/dist PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user backend/requirements-runtime.txt backend/requirements-runtime.txt
RUN pip install --no-cache-dir --user -r backend/requirements-runtime.txt
COPY --chown=user backend/ backend/
COPY --chown=user catalog/ catalog/
COPY --chown=user samples/ samples/
COPY --chown=user scripts/ scripts/
COPY --chown=user --from=web /web/dist frontend/dist
EXPOSE 7860
CMD ["sh", "scripts/start.sh"]
