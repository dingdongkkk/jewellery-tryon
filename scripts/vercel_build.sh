#!/bin/bash
# Vercel build: the static site (frontend/dist) plus the files the browser needs at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd frontend && VITE_STATELESS=1 npx vite build)
DIST=frontend/dist
mkdir -p "$DIST/catalog" "$DIST/samples" "$DIST/models" "$DIST/mediapipe-wasm"
cp -R catalog/prepared/cutouts catalog/prepared/products "$DIST/catalog/"
cp samples/*.jpg "$DIST/samples/"
python3 -c "import json,glob,os; print(json.dumps({'samples':['/samples/'+os.path.basename(p) for p in sorted(glob.glob('samples/*.jpg'))]}))" > "$DIST/samples/index.json"
cp backend/models/*.task "$DIST/models/"
cp frontend/node_modules/@mediapipe/tasks-vision/wasm/* "$DIST/mediapipe-wasm/"
