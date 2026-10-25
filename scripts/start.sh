#!/bin/sh
# Container start: seed the catalogue (no AI calls: the prepared cutouts are copied in), then run the
# worker and the web server (frontend + API) together.
set -e
cd /home/user/app
mkdir -p "$DATA_DIR/cutouts" "$DATA_DIR/masks" "$DATA_DIR/originals/products" "$DATA_DIR/originals/uploads" "$DATA_DIR/outputs"
cp -n catalog/prepared/cutouts/* "$DATA_DIR/cutouts/" 2>/dev/null || true
cp -n catalog/prepared/masks/* "$DATA_DIR/masks/" 2>/dev/null || true
cp -n catalog/prepared/products/* "$DATA_DIR/originals/products/" 2>/dev/null || true
python scripts/prepare_catalog.py
(cd backend && python -m app.worker) &
cd backend && exec uvicorn app.serve:root --host 0.0.0.0 --port "${PORT:-7860}"
