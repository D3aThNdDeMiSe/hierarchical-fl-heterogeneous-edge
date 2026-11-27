"""Deployment check: size and accuracy of the final global model in float32 and int8.

Trains the proposed scheme in the main configuration (alpha = 0.5, seed 0), then applies
post-training int8 weight quantisation (per-tensor symmetric, biases kept in float32, as in
dynamic-range quantisation) and evaluates both versions on the Fashion-MNIST test set.
Writes results/deploy_check.json.
"""
import json
import os

import torch
from torch.nn.utils import vector_to_parameters

from hfl import data as D
from hfl.model import build_model
from hfl.sim import Config, run

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, x, y, bs=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    cfg = Config(method="hier_proposed", alpha=0.5, seed=0)
    res = run(cfg, log=lambda s: None, return_weights=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, x_te, y_te = D.load(cfg.dataset, os.path.join(ROOT, "data"), dev)

    model = build_model(cfg.dataset).to(dev).eval()
    vector_to_parameters(res["weights"].to(dev), model.parameters())
    acc_fp32 = evaluate(model, x_te, y_te)

    int8_bytes = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:  # weight tensors: int8 values + one float32 scale
                scale = p.abs().max() / 127.0
                p.copy_(torch.clamp(torch.round(p / scale), -127, 127) * scale)
                int8_bytes += p.numel() + 4
            else:            # biases stay float32
                int8_bytes += p.numel() * 4
    acc_int8 = evaluate(model, x_te, y_te)

    n_params = sum(p.numel() for p in model.parameters())
    out = dict(config=cfg.name, params=n_params, fp32_bytes=n_params * 4, int8_bytes=int8_bytes,
               compression=n_params * 4 / int8_bytes, acc_fp32=acc_fp32, acc_int8=acc_int8,
               acc_change_points=100 * (acc_int8 - acc_fp32), run_final_acc=res["final"]["acc"])
    with open(os.path.join(ROOT, "results", "deploy_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
