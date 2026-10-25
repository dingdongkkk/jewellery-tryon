"""Run every test photo through the live pipeline once and record the outcome.

Usage: python scripts/batch_test.py            (paid: one fal call per case)
Results: test-results/results.json + test-results/sheet.jpg. Each job's AI images stay in data/outputs/<job>, so
later pipeline changes can be checked for free with scripts/regression.py.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
CASES = [
    ("01-front-bw-lowkey", "ear-pearl-enamel"),
    ("02-profile-bw-hair", "ear-granulated-hoop"),
    ("03-hair-covering-ears", "ear-garnet-cones"),
    ("04-candid-indoor-short-hair", "ear-pearl-enamel"),
    ("05-looking-down-warm-dark", "ear-granulated-hoop"),
    ("06-profile-looking-up-studio", "ear-garnet-cones"),
    ("07-three-quarter-outdoor-curly", "ear-granulated-hoop"),
    ("08-profile-lowkey-black", "ear-pearl-enamel"),
    ("09-looking-down-blonde", "ear-garnet-cones"),
    ("10-profile-hat-outdoor", "ear-pearl-enamel"),
    ("11-smiling-already-wearing-hoops", "ear-granulated-hoop"),
    ("12-golden-backlight-tilted", "ear-pearl-enamel"),
    ("13-warm-beauty-hand-face", "ear-garnet-cones"),
    ("14-bw-off-shoulder-pendant", "pend-mistress-animals"),
    ("15-window-light-moody", "pend-filigree-cross"),
    ("16-hands-holding-brushes-ring", "ring-garnet-cloisonne"),
    ("17-hands-makeup-ring", "ring-granulated-bezel"),
]


def call(method, path, body=None, ctype="application/json"):
    req = urllib.request.Request(API + path, data=body, method=method, headers={"Content-Type": ctype} if body else {})
    return json.load(urllib.request.urlopen(req, timeout=120))


def upload(path: Path):
    bd = "----batch"
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{bd}--\r\n".encode()
    return call("POST", "/uploads", body, f"multipart/form-data; boundary={bd}")


def run_case(photo: str, product: str) -> dict:
    rec = {"photo": photo, "product": product}
    up = upload(ROOT / "test-photos" / f"{photo}.jpg")
    rec["upload_id"] = up["id"]
    try:
        pl = call("POST", "/autoplace", json.dumps({"upload_id": up["id"], "product_id": product}).encode())
    except urllib.error.HTTPError as e:
        rec.update(status="autoplace_failed", error=e.read().decode()[:300])
        return rec
    rec["detected"] = pl.get("detected")
    t = time.time()
    job = call("POST", "/tryons", json.dumps({"upload_id": up["id"], "product_id": product,
                                              "placement": {k: pl[k] for k in ("items", "chain") if k in pl}}).encode())
    while job["status"] not in ("complete", "failed", "quota_exceeded"):
        time.sleep(1)
        job = call("GET", f"/tryons/{job['id']}")
    worn = ((job.get("timings") or {}).get("info") or {}).get("worn") or {}
    rec.update(job_id=job["id"], status=job["status"], error=job.get("error"), seconds=round(time.time() - t, 1),
               attempts=worn.get("attempts"), pieces=worn.get("pieces"), note=worn.get("note"))
    return rec


def sheet(results, out: Path):
    tiles = []
    for r in results:
        p = ROOT / "data" / "outputs" / str(r.get("job_id")) / "tryon.png"
        src = p if p.exists() else ROOT / "test-photos" / f"{r['photo']}.jpg"
        im = Image.open(src).convert("RGB")
        im.thumbnail((320, 320))
        t = Image.new("RGB", (320, 352), "white")
        t.paste(im, (0, 32))
        d = ImageDraw.Draw(t)
        d.text((3, 2), f"{r['photo'][:30]}", fill="black")
        d.text((3, 16), f"{r['product']} | {r['status']}", fill="red" if r["status"] != "complete" else "green")
        tiles.append(t)
    cols = 6
    s = Image.new("RGB", (320 * cols, 352 * ((len(tiles) + cols - 1) // cols)), "white")
    for k, t in enumerate(tiles):
        s.paste(t, (320 * (k % cols), 352 * (k // cols)))
    s.save(out, quality=88)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    out_dir = ROOT / "test-results"
    out_dir.mkdir(exist_ok=True)
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {(r["photo"], r["product"]) for r in results if r.get("status") == "complete"}
    for photo, product in CASES:
        if only and photo not in only:
            continue
        if not only and (photo, product) in done:
            continue
        r = run_case(photo, product)
        results = [x for x in results if (x["photo"], x["product"]) != (photo, product)] + [r]
        res_path.write_text(json.dumps(results, indent=1))
        print(f"{photo:36s} {product:24s} {r['status']:16s} {r.get('seconds', '')}s {(r.get('error') or '')[:120]}", flush=True)
    order = {c: i for i, c in enumerate(CASES)}
    results.sort(key=lambda r: order.get((r["photo"], r["product"]), 99))
    sheet(results, out_dir / "sheet.jpg")
