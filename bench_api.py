"""Latency of the PC inference API (serve_model.py), measured from a client on the same machine.

Starts the server, sends single-image and batched requests built from the deployment samples,
and writes results/api_latency.json. Each request opens a new HTTP connection, as a phone would.

  python bench_api.py [--port 8080] [--single 300] [--batches 30] [--batch-size 100]
"""
import argparse
import base64
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def post(url, images):
    data = json.dumps({"images": images}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    return (time.perf_counter() - t0) * 1000, out


def summary(ms):
    ms = sorted(ms)
    return dict(n=len(ms), median_ms=round(st.median(ms), 2), p95_ms=round(ms[int(0.95 * (len(ms) - 1))], 2),
                mean_ms=round(st.mean(ms), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--single", type=int, default=300)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=100)
    a = ap.parse_args()

    with open(os.path.join(ROOT, "deploy", "samples.json")) as f:
        s = json.load(f)
    images = [list(base64.b64decode(p)) for p in s["pixels_b64"]]

    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve_model.py"), "--port", str(a.port),
                               "--host", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(100):  # wait until the server answers
            try:
                urllib.request.urlopen(base + "/model", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        for i in range(10):  # warm-up
            post(base + "/predict", [images[i]])

        single, agree = [], 0
        for i in range(a.single):
            ms, out = post(base + "/predict", [images[i % len(images)]])
            single.append(ms)
            agree += out["predictions"][0]["label"] == s["pred_fp32"][i % len(images)]
        batched = []
        for b in range(a.batches):
            start = (b * a.batch_size) % len(images)
            batch = [images[(start + k) % len(images)] for k in range(a.batch_size)]
            ms, _ = post(base + "/predict", batch)
            batched.append(ms)
    finally:
        server.terminate()
        server.wait()

    b = summary(batched)
    res = dict(note="Client and server on the same laptop (loopback), CPU inference, new HTTP connection per request.",
               cpu=platform.processor(), python=platform.python_version(),
               single_image=summary(single), single_image_agree_with_reference=f"{agree}/{a.single}",
               batch_size=a.batch_size, batch=b,
               batch_images_per_s=round(a.batch_size / (b["median_ms"] / 1000), 1))
    with open(os.path.join(ROOT, "results", "api_latency.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
