"""End-to-end smoke test against the running API: upload sample, place earring on the ear, run a job.
Usage: python scripts/smoke_test.py [provider] [product_id]"""
import json, sys, time, urllib.request

API = "http://127.0.0.1:8000"
provider = sys.argv[1] if len(sys.argv) > 1 else "none"
pid = sys.argv[2] if len(sys.argv) > 2 else "ear-garnet-cones"


def call(method, path, body=None, ctype="application/json"):
    req = urllib.request.Request(API + path, data=body, method=method, headers={"Content-Type": ctype} if body else {})
    return json.load(urllib.request.urlopen(req, timeout=60))


boundary = "----smoke"
img = open("samples/portrait-blue.jpg", "rb").read()
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"p.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + img + f"\r\n--{boundary}--\r\n".encode()
up = call("POST", "/uploads", body, f"multipart/form-data; boundary={boundary}")
prods = {p["id"]: p for p in call("GET", "/products")["products"]}
p = prods[pid]
d = p["dimensions"]
ax, ay = p["anchors"]["attach"]
if p["category"] == "earring":
    lobe, width = (745, 1208), 150.0   # left earlobe in portrait-blue.jpg (2000x2501)
else:
    lobe, width = (1000, 1880), 220.0  # top of the pendant, below the chin
h = width * d["cutout_h"] / d["cutout_w"]
item = {"cx": lobe[0] + (0.5 - ax) * width, "cy": lobe[1] + (0.5 - ay) * h, "width": width, "rotation": 0, "flip": False}
t = time.time()
job = call("POST", "/tryons", json.dumps({"upload_id": up["id"], "product_id": pid, "placement": {"items": [item]}, "provider": provider, "seed": 1234}).encode())
last = None
while True:
    j = call("GET", f"/tryons/{job['id']}")
    if j["status"] != last:
        print(f"{time.time() - t:6.1f}s {j['status']} {j.get('error') or ''}")
        last = j["status"]
    if j["status"] in ("complete", "failed", "quota_exceeded"):
        break
    time.sleep(0.5)
print(json.dumps({k: j[k] for k in ("id", "timings", "assets", "debug", "seed", "provider")}, indent=1))
