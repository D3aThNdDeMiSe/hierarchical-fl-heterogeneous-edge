"""Deployment check: size and accuracy of the exported global model in float32 and int8.

Evaluates the model that export_model.py wrote to deploy/model/ (the proposed scheme in the main
configuration, alpha = 0.5, seed 0), so the paper, the phone page and the PC API all describe the
same weights. int8: per-tensor symmetric quantisation of weight tensors, biases kept in float32,
as in dynamic-range quantisation. Writes results/deploy_check.json.
"""
import json
import os

import torch

from hfl import data as D
from hfl.model import build_model

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "deploy", "model")


def evaluate(model, x, y, bs=2000):
    with torch.no_grad():
        return sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs)) / len(x)


def main():
    with open(os.path.join(MODEL_DIR, "model.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = build_model("fmnist").eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_fp32.pt"), map_location="cpu"))
    _, _, x_te, y_te = D.load("fmnist", os.path.join(ROOT, "data"), torch.device("cpu"))
    acc_fp32 = evaluate(model, x_te, y_te)

    int8_bytes = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() > 1:  # weight tensors: int8 values + one float32 scale
                scale = p.abs().max() / 127.0
                p.copy_(torch.clamp(torch.round(p / scale), -127, 127) * scale)
                int8_bytes += p.numel() + 4
            else:            # biases stay float32
                int8_bytes += p.numel() * 4
    acc_int8 = evaluate(model, x_te, y_te)

    n_params = sum(p.numel() for p in model.parameters())
    out = dict(model_version=meta["version"], config=meta["source_run"], params=n_params, fp32_bytes=n_params * 4,
               int8_bytes=int8_bytes, compression=n_params * 4 / int8_bytes, acc_fp32=acc_fp32, acc_int8=acc_int8,
               acc_change_points=100 * (acc_int8 - acc_fp32))
    with open(os.path.join(ROOT, "results", "deploy_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
