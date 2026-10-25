# Jewellery Try-On

Photo-based earrings, pendants and rings with source-image detail and optional AI lighting.
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

Set `HF_TOKEN` for the free Hugging Face Spaces, or `FAL_KEY` for paid fal.ai generation. Keys stay on the backend. With a fal key the existing default is `fal-nano-banana`; select a different provider in Advanced or set `TRYON_PROVIDER`. `none` uses local lighting and makes no inference requests. Free Spaces have queues and quota limits; paid inference is billed per request. **A pair of earrings now uses two close-crop requests, one per piece** (and retries can incur additional usage). No external service is needed for local lighting.

## Detail and realism pipeline

1. Preserve the uploaded photo (EXIF corrected, capped at 6000 pixels on its long side). Load the original transparent product cutout, not a thumbnail.
2. Estimate placement from local MediaPipe landmarks. Review the attachment point and adjust it before generating. Mirrored earrings use mirrored product anchors. Face/finger occlusion masks are included in the render and lighting masks.
3. Render each piece directly into its own 1024-pixel context crop for the AI lighting pass. This gives a small earring hundreds of model pixels instead of a few dozen in a full portrait. The selected placement is shared by the AI input and final result.
4. Check global and local alignment of every returned crop. If a model moves the piece or reframes the photograph, retain local lighting for that piece. From accepted crops borrow only smooth relighting and bounded contact darkening. Never paste the AI-redrawn jewellery into the detailed result. A separate AI draft is available for inspection when framing allows it.
5. Transform the original product texture with premultiplied alpha. An area prefilter handles large reductions; one final affine transform handles placement, mirroring and rotation. Bounded relighting preserves highlight variation rather than clipping bright metal to white. Avoid adding synthetic noise over the jewellery.
6. Render a single PNG master at the chosen 4K/6K/8K target, bounded by source sampling, 8× enlargement and 64 megapixels. Never reduce an already larger uploaded photo. Rendering uses overlapping strips to keep working memory manageable on a 16 GB Mac.
7. Show the master in the zoom viewer. “Inspect piece” focuses its actual output pixels; zoom beyond 100% is explicitly labelled as enlargement. Downloads contain the same detail. Close-ups are literal crops of the PNG, not independently sharpened or regenerated views. Before/after uses a smaller preview for speed.

A 6K portrait is not six thousand pixels of jewellery. Inspect the output metadata to see the rendered width of each piece. Large exports reveal more of the source texture, but cannot recover engraving absent from the product photo or make an unfocused source sharp. Background enlargement is ordinary interpolation; it adds no genuine skin detail.

## Controls and outputs

Upload a photo, choose a product, check placement, choose export detail, then generate. The initial result is **Detailed result**; **AI draft** is a separate optional tab whose design can differ from the source.

Each job directory contains:

- `tryon.png`: canonical full-resolution PNG.
- `preview.jpg`: smaller preview for comparison.
- `detail_0.png`, `detail_1.png`: exact master crops; `detail.png` aliases the first.
- `reference.*`: original product photograph.
- `lighting_input_*.png`, `lighting_output_*.png`: per-piece AI diagnostic crops.
- `tryon_ai.png`: optional AI draft, not the fidelity-preserving export.

`timings.info.quality` records output/upload/source sizes, scale, jewellery widths, source limits and whether the background was enlarged. `lighting_passes` records crop coordinates and whether each AI pass was accepted. Existing old jobs remain viewable with their legacy tile overlays; regenerate them for a unified detailed PNG.

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
