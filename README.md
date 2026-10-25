# Jewellery Try-On

Photo-based earrings, pendants and rings with natural AI placement and optional close-up detail enhancement.
React/Vite frontend, FastAPI API, SQLite queue and a separate Python worker.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
(cd frontend && npm install)
cp backend/.env.example backend/.env
.venv/bin/python scripts/prepare_catalog.py
./scripts/dev.sh
```

Open http://localhost:5188. Existing catalogue assets can be reused; catalogue preparation only needs to run when adding products. Restart the worker after Python changes; the API and frontend reload automatically.

Set `HF_TOKEN` for the free Hugging Face Spaces, or `FAL_KEY` for paid fal.ai generation. Keys stay on the backend. With a fal key the existing default is `fal-nano-banana`; select a different provider in Advanced or set `TRYON_PROVIDER`. `none` uses local lighting and makes no inference requests. Free Spaces have queues and quota limits; paid inference is billed per request. The worn pipeline uses a context generation, with possible retries, escalation and wornness checks. Optional detail enhancement adds one generation per detected piece. No external service is needed for local lighting.

## Detail and realism pipeline

1. Preserve the uploaded photo and use landmarks to estimate the jewellery region. If detection fails, give the model the whole photo.
2. Nano Banana receives the untouched person crop and product reference, generating jewellery with natural attachment, perspective, shadows and occlusion. Reject changed framing and failed wornness checks; do not silently substitute a pasted product image.
3. Preserve the accepted generated jewellery. **Enhance jewellery detail** optionally generates a tight 1024-pixel close-up for each detected piece, using the product reference. Register the close-up and reject it if alignment or geometry changes too much. A failed detail pass keeps the accepted main result.
4. Blend accepted detail into the master and export a lossless PNG at the selected size, bounded by source sampling and 64 megapixels. Increasing export dimensions alone does not create texture.
5. Inspection crops follow the detected jewellery location. They are exact crops of the downloadable master, so the preview and download show the same detail.

AI-generated stones and engraving can differ from the product reference. Compare the close-up against the original product image before using a result as an accurate product representation. Legacy providers and explicit no-AI preview still use the local compositing pipeline.

## Controls and outputs

Upload a photo, choose a product, select optional detail enhancement and export size, then generate. Detail enhancement costs additional AI calls and is off by default.

Each job directory contains `tryon.png` (the master), `preview.jpg`, exact master crops `detail_0.png` / `detail_1.png`, and the original product reference. `tryon_ai.png` shows the accepted worn result before optional detail enhancement. Diagnostic crops and alignment information record which enhancement passes were accepted. Restart the worker after Python changes.

## API

| Endpoint | Purpose |
|---|---|
| `POST /uploads` | Multipart `file`, orientation correction and validation |
| `DELETE /uploads/{id}` | Delete upload and its jobs |
| `GET /products` | Catalogue and category availability |
| `POST /autoplace` | `{upload_id, product_id}` → estimated placements |
| `POST /tryons` | `{upload_id, product_id, placement, provider?, seed?, settings?}` |
| `GET /tryons/{id}` | Status, assets, crop regions and quality metadata |
| `DELETE /tryons/{id}` | Delete result and unshared upload |

Placement items use uploaded-photo pixels: `{cx, cy, width, rotation, flip, occlude?}`.
Output size is requested through `settings: {"output_long_side": 6144}` (1024–8192).
States: `queued → preparing → generating → compositing → complete`, or `failed` / `quota_exceeded`.
The browser offers local lighting when inference quota is unavailable.

## Catalogue

`catalog/catalog.json` holds product sources, dimensions and attachment anchors. BiRefNet runs during preparation and caches original image, alpha mask and cutout separately. Inspect thin chains, prongs, openings and reflective edges manually. Sample photos and catalogue credits remain in their respective directories.

## Checks

```bash
PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests -v
(cd frontend && npm run build && npm run lint)
```

Image regressions check texture in the larger master, exact crop/download agreement, strip seams, highlight retention, occlusion consistency, output-size validation and rejection of displaced AI images. Live provider availability and image quality require separate sample runs; offline tests do not verify a remote model.

## Current limits

- A single product photograph cannot supply unseen surfaces or physically correct reflections from a new angle. Strong rotations and wraparound pieces need multiple photographs or accurate 3D assets.
- Face and finger masks handle some occlusion, but hair and clothing segmentation are not automatic. Placement may need manual correction. The opposite earring is a mirrored view.
- Pendant chains are procedural approximations, not a photographed catalogue chain. Bracelets remain unsupported.
- Approximate physical scale is inferred from landmarks; this is a visual preview, not a size measurement.
- Photos and results remain in local storage until deleted. AI mode sends context crops to the configured service. The prototype's `/media` endpoint has no authentication; production needs access controls.

## Deployment (Docker / Hugging Face Space)
`Dockerfile` builds one container: the React build, the API (served under `/api`) and the job worker, on port 7860.
`scripts/start.sh` copies the prepared catalogue (`catalog/prepared/`) into `DATA_DIR`, so the server never needs
the background-removal model. Environment variables / Space secrets:

| Variable | Needed | Purpose |
|---|---|---|
| `FAL_KEY` | yes | AI try-on |
| `APP_PASSWORD` | recommended | browser password prompt (any username), so a public link can't spend the AI credit |
| `HF_TOKEN` | optional | free fallback model |

`scripts/deploy_space.py` uploads the repo to a Hugging Face Docker Space and sets these secrets. The camera button
needs https, which the Space URL provides. Uploaded photos live in the container and are wiped on restart.

## Deployment (Vercel, serverless)

`vercel.json` deploys a stateless version: the static site plus one Python function (`api/index.py` → `backend/app/stateless.py`).
MediaPipe can't run inside a Vercel function, so body landmarks are detected in the browser (`frontend/src/landmarks.ts`, same
models) and sent with the photo. Nothing is stored on the server: the browser keeps the photo, each request carries it, and results
come back in the response (2560 px JPEG, to fit the 4.5 MB response limit). Built with `VITE_STATELESS=1` (`scripts/vercel_build.sh`).

Set `FAL_KEY` and `APP_PASSWORD` as Vercel environment variables. With `APP_PASSWORD` set, the site asks for the password before
the first AI request. Differences from the full server: no 6K/8K export, no job history, one request per try-on (max 300 s).
