"""Minimal inference API on the PC for the exported global model (standard library + PyTorch).

  python serve_model.py [--port 8080]

  GET  /model    model.json: version, source run, accuracy, SHA-256 of the weights
  POST /predict  {"images": [[784 pixel values 0-255, row-major 28x28], ...]}
                 -> {"version": ..., "predictions": [{"label": 9, "class": "Ankle boot", "probs": [...]}, ...]}

Run export_model.py first; it writes deploy/model/.
"""
import argparse
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from hfl.model import build_model

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "deploy", "model")


def load():
    with open(os.path.join(MODEL_DIR, "model.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(MODEL_DIR, meta["files"]["fp32"]["file"]), "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != meta["files"]["fp32"]["sha256"]:
            raise SystemExit("weights_fp32.bin does not match the SHA-256 in model.json")
    model = build_model("fmnist").eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_fp32.pt"), map_location="cpu"))
    return meta, model


def predict(meta, model, images):
    x = torch.tensor(images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    x = (x - meta["input"]["mean"]) / meta["input"]["std"]
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)
    return [dict(label=int(p.argmax()), **{"class": meta["classes"][int(p.argmax())]},
                 probs=[round(float(v), 4) for v in p]) for p in probs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    meta, model = load()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/model":
                self._send(200, meta)
            else:
                self._send(404, {"error": "use GET /model or POST /predict"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "use POST /predict"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                images = req["images"]
                if not images or any(len(im) != 784 for im in images) or len(images) > 1000:
                    raise ValueError("send 1-1000 images of 784 pixel values each")
                self._send(200, {"version": meta["version"], "predictions": predict(meta, model, images)})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    print(f"model v{meta['version']} ({meta['source_run']}) serving on http://{a.host}:{a.port}  "
          f"(GET /model, POST /predict)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
