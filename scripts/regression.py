"""Free regression check: replay the saved AI images of every batch-test job through the CURRENT pipeline code.

No AI calls are made: a stub provider returns the images that fal produced during scripts/batch_test.py.
Usage: python scripts/regression.py [photo-name ...]
Output: test-results/regression/<photo>.png, test-results/regression.json, test-results/regression_sheet.jpg
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import tryon  # noqa: E402
from app.db import db, row_to_dict  # noqa: E402
from app.providers import GenerationResult  # noqa: E402


def replay(rec: dict, out_root: Path) -> dict:
    with db() as c:
        j = row_to_dict(c.execute("SELECT * FROM tryon_jobs WHERE id=?", (rec["job_id"],)).fetchone())
        prod = row_to_dict(c.execute("SELECT * FROM products WHERE id=?", (j["product_id"],)).fetchone())
        up = row_to_dict(c.execute("SELECT * FROM uploads WHERE id=?", (j["upload_id"],)).fetchone())
    d = ROOT / "data" / "outputs" / rec["job_id"]
    # same order the pipeline asks for them: first attempt(s), then the escalation attempt, then detail passes
    seq = ([d / "ai_context.png"] + sorted(d.glob("ai_context_*.png")) + sorted(d.glob("esc_ai_context*.png"))
           + sorted(d.glob("detail_ai_raw_*.png")))
    seq = [Image.open(p).convert("RGB") for p in seq if p.exists()]
    if not seq:
        return {"photo": rec["photo"], "status": "no saved AI image"}

    class Replay:
        name = "replay"

        if VERIFY:
            def verify(self, image, category):  # the real vision check (about $0.0002 each)
                from app.providers.fal import verify_worn
                return verify_worn(image, category)

        def generate_tryon(self, *a, **k):
            if not seq:
                raise RuntimeError("no more saved AI images")
            return GenerationResult(seq.pop(0), j["seed"], {})

    tryon.get_provider = lambda n: Replay()
    job = dict(j)
    job["provider"] = "fal-nano-banana"
    out = out_root / rec["job_id"]
    try:
        r = tryon.run_tryon(job, prod, up, out, lambda s: None)
    except Exception as e:  # a failure is a result too
        return {"photo": rec["photo"], "status": "failed", "error": str(e)[:200]}
    final = out_root.parent / f"{rec['photo']}.png"
    im = Image.open(r["output_path"])
    im.thumbnail((1200, 1200))
    im.save(final)
    w = r["info"].get("worn", {})
    return {"photo": rec["photo"], "status": "complete", "pieces": w.get("pieces"), "note": w.get("note"),
            "checks": [(c.get("answer"), c.get("reason")) for c in w.get("worn_checks", [])], "image": str(final)}


VERIFY = "--verify" in sys.argv

if __name__ == "__main__":
    only = set(a for a in sys.argv[1:] if not a.startswith("--"))
    res = json.loads((ROOT / "test-results" / "results.json").read_text())
    out_root = ROOT / "test-results" / "regression" / "work"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in res:
        if not rec.get("job_id") or (only and rec["photo"] not in only):
            continue
        r = replay(rec, out_root)
        rows.append(r)
        print(f"{r['photo']:36s} {r['status']:10s} {json.dumps(r.get('checks'))[:200]} {r.get('error') or ''}", flush=True)
    (ROOT / "test-results" / "regression.json").write_text(json.dumps(rows, indent=1))
    tiles = []
    for r in rows:
        if r["status"] != "complete":
            continue
        im = Image.open(r["image"]).convert("RGB")
        im.thumbnail((320, 320))
        t = Image.new("RGB", (320, 340), "white")
        t.paste(im, (0, 20))
        ImageDraw.Draw(t).text((3, 3), r["photo"][:40], fill="black")
        tiles.append(t)
    if tiles:
        s = Image.new("RGB", (320 * 6, 340 * ((len(tiles) + 5) // 6)), "white")
        for k, t in enumerate(tiles):
            s.paste(t, (320 * (k % 6), 340 * (k // 6)))
        s.save(ROOT / "test-results" / "regression_sheet.jpg", quality=88)
