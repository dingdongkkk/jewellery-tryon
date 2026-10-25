"""Production entry point: one server for the built frontend, the API (under /api) and media files.

Locally the Vite dev server proxies /api to app.main; in the container this file does the same job.
If APP_PASSWORD is set, every request needs HTTP basic auth (any username), so a public link can't be used
to spend the AI credit.
"""
import base64
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import config
from .main import app as api

FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", config.ROOT / "frontend" / "dist"))
PASSWORD = os.getenv("APP_PASSWORD", "")

root = FastAPI(title="Jewellery try-on")


@root.middleware("http")
async def password_gate(request: Request, call_next):
    if PASSWORD:
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                ok = hmac.compare_digest(given, PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response("Password required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Jewellery try-on"'})
    return await call_next(request)


root.mount("/api", api)
root.mount("/media", StaticFiles(directory=config.DATA_DIR), name="media")
root.mount("/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")
if FRONTEND_DIST.exists():
    root.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
