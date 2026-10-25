"""Auto-place + AI try-on through the running API. Usage: python scripts/ai_run.py <sample> <product_id> [provider]"""
import json, sys, time, urllib.error, urllib.request

API = "http://127.0.0.1:8000"
sample, pid = sys.argv[1], sys.argv[2]
provider = sys.argv[3] if len(sys.argv) > 3 else None  # None: the server default (fal when FAL_KEY is set)


def call(m, p, b=None, ct="application/json"):
    r = urllib.request.Request(API + p, data=b, method=m, headers={"Content-Type": ct} if b else {})
    return json.load(urllib.request.urlopen(r, timeout=120))


bd = "----x"
import os
img = open(sample if os.path.exists(sample) else f"samples/{sample}.jpg", "rb").read()
body = f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"p.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + img + f"\r\n--{bd}--\r\n".encode()
up = call("POST", "/uploads", body, f"multipart/form-data; boundary={bd}")
try:
    pl = call("POST", "/autoplace", json.dumps({"upload_id": up["id"], "product_id": pid}).encode())
except urllib.error.HTTPError as e:
    print("autoplace failed:", e.code, e.read().decode()[:300])
    sys.exit(1)
print("placement:", json.dumps(pl.get("detected")))
t = time.time()
j = call("POST", "/tryons", json.dumps({"upload_id": up["id"], "product_id": pid, "placement": {k: pl[k] for k in ("items", "chain") if k in pl}, **({"provider": provider} if provider else {})}).encode())
last = None
while True:
    j = call("GET", f"/tryons/{j['id']}")
    if j["status"] != last:
        print(f"{time.time() - t:6.1f}s {j['status']} {(j.get('error') or '')[:300]}", flush=True)
        last = j["status"]
    if j["status"] in ("complete", "failed", "quota_exceeded"):
        break
    time.sleep(1)
print(json.dumps({k: j[k] for k in ("id", "provider", "timings", "assets", "debug")}, indent=1)[:2500])
