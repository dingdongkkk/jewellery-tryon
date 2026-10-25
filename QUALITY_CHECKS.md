# Detail pipeline verification — 25 October 2026

## Completed checks

- Ten backend image regressions pass (`PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests -v`). They cover exact master crops, source texture at export scale, fractional-scale strip/chain/shadow seams, highlight variation, mirrored attachment anchors, occlusion masks, invalid export settings, individual high-resolution AI crops, rejected images and no-image fallback.
- Frontend TypeScript/Vite production build and oxlint pass.
- Browser flow: bundled portrait → automatic earring placement → generate → local-lighting fallback → detailed result. Both piece-focus buttons, 100% view, Fit, before/after slider, texture/reference view and PNG download were exercised.
- Downloaded PNG was byte-for-byte identical to the server master (`cmp`), and both close-up files were pixel-for-pixel crops of that master.
- Desktop and 390px mobile view have no horizontal overflow. Screenshots: `output/playwright/inspect-jewellery.png`, `output/playwright/mobile-texture.png`.
- Earrings, pendant and ring jobs all completed locally, with their final texture crops visually inspected. 6K earring output: 4884×6144 with a 312px-wide piece. 8K output: 6512×8192 with 416px-wide pieces (the input placement was about 128px wide).

## Live inference evidence

- A broad-crop Nano Banana run moved the jewellery; the alignment gate rejected that lighting.
- A tight-crop Nano Banana run returned no image. This exposed the need for the now-tested local-lighting fallback and concise provider errors.
- Tight-crop Qwen generation changed anatomy/framing and was rejected. It is not evidence of faithful try-on.
- After simplifying the lighting-only prompt, a single-piece Nano Banana sample passed global/local alignment (1.000/0.999) and completed a 4884×6144 source-based export: `data/outputs/job_b474b91de2b0/`. Its close-up was verified against the master.
- The final 8K local-lighting sample is `data/outputs/job_fddd0cafe32a/`.

These checks demonstrate detail retention and guarded lighting integration, not physically exact 3D fitting. Hair occlusion, hidden surfaces, ring wraparound and the procedural pendant chain remain limitations of the existing photo-based approach. Full source detail cannot survive arbitrary magnification or a blurry source photograph. Free Hugging Face inference was not rerun in this change.
