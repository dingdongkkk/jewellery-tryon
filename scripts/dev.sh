#!/bin/zsh
# Starts API (8000), worker and Vite (5173). Ctrl-C stops all three.
cd "$(dirname "$0")/.."
ROOT=$PWD
trap 'kill 0' EXIT INT TERM
(cd backend && "$ROOT/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app) &
(cd backend && "$ROOT/.venv/bin/python" -m app.worker) &
(cd frontend && npm run dev -- --host 127.0.0.1 --port 5188 --strictPort) &
wait
