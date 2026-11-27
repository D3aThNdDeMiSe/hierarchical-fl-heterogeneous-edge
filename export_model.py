"""Export the final global model for deployment: versioned weights for the phone page and the PC API.

Trains the proposed scheme in the main configuration (alpha = 0.5, seed 0; the run used by
deploy_check.py), then writes deploy/model/:

  model.json          version, source run, accuracy, SHA-256 of the weights, int8 scales, layout
  weights_fp32.bin    float32 weights in TensorFlow.js layer order (NHWC / HWIO), 322,408 bytes
  weights_int8.bin    int8 kernels (per-tensor symmetric) + float32 biases, about 81 KB
  model_fp32.pt       PyTorch state_dict, loaded by serve_model.py
  ../samples.json     300 test images (30 per class) with PyTorch's predictions, for the parity check

The layout conversion (PyTorch NCHW -> TF.js NHWC) is checked here against PyTorch with an
independent NumPy forward pass before anything is written.
"""
import base64
import datetime
import hashlib
import json
import os

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from hfl import data as D
from hfl.model import build_model
from hfl.sim import Config, run

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "deploy", "model")
VERSION = "1.0"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530


def to_tfjs(sd):
    """PyTorch state_dict -> list of (name, array) in TF.js layer order and layout."""
    c1, c2 = sd["conv1.weight"], sd["conv2.weight"]            # [out, in, kh, kw]
    f1, f2 = sd["fc1.weight"], sd["fc2.weight"]                # [out, in]
    hidden = f1.shape[0]
    f1 = f1.reshape(hidden, 32, 7, 7).transpose(0, 2, 3, 1).reshape(hidden, -1)  # flatten order C,H,W -> H,W,C
    return [("conv1/kernel", c1.transpose(2, 3, 1, 0)), ("conv1/bias", sd["conv1.bias"]),
            ("conv2/kernel", c2.transpose(2, 3, 1, 0)), ("conv2/bias", sd["conv2.bias"]),
            ("dense1/kernel", f1.T), ("dense1/bias", sd["fc1.bias"]),
            ("dense2/kernel", f2.T), ("dense2/bias", sd["fc2.bias"])]


def np_forward(tf_weights, x_nhwc):
    """Independent NHWC forward pass in NumPy using the TF.js-layout weights."""
    w = dict(tf_weights)

    def conv_same(x, k, b):                                    # x [N,H,W,C], k [3,3,C,O]
        n, h, wd, _ = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        out = np.zeros((n, h, wd, k.shape[3]), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                out += np.einsum("nhwc,co->nhwo", xp[:, i:i + h, j:j + wd, :], k[i, j])
        return out + b

    def pool(x):
        n, h, wd, c = x.shape
        return x.reshape(n, h // 2, 2, wd // 2, 2, c).max(axis=(2, 4))

    x = pool(np.maximum(conv_same(x_nhwc, w["conv1/kernel"], w["conv1/bias"]), 0))
    x = pool(np.maximum(conv_same(x, w["conv2/kernel"], w["conv2/bias"]), 0))
    x = np.maximum(x.reshape(len(x), -1) @ w["dense1/kernel"] + w["dense1/bias"], 0)
    return x @ w["dense2/kernel"] + w["dense2/bias"]


def quantise(tf_weights):
    """Per-tensor symmetric int8 for kernels; biases stay float32 (as in deploy_check.py)."""
    out, scales = [], {}
    for name, a in tf_weights:
        if name.endswith("kernel"):
            s = float(np.abs(a).max() / 127.0)
            q = np.clip(np.round(a / s), -127, 127).astype(np.int8)
            out.append((name, q))
            scales[name] = s
        else:
            out.append((name, a.astype(np.float32)))
    return out, scales


def dequantise(qw, scales):
    return [(n, a.astype(np.float32) * scales[n] if n in scales else a) for n, a in qw]


def main():
    cfg = Config(method="hier_proposed", alpha=0.5, seed=0, data_root=os.path.join(ROOT, "data"))
    print("training", cfg.name, "(about 7 minutes on a laptop GPU)")
    res = run(cfg, log=lambda s: None, return_weights=True)
    model = build_model(cfg.dataset).eval()
    vector_to_parameters(res["weights"].cpu(), model.parameters())
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}

    _, _, x_te, y_te = D.load(cfg.dataset, os.path.join(ROOT, "data"), torch.device("cpu"))
    with torch.no_grad():
        logits_pt = torch.cat([model(x_te[i:i + 2000]) for i in range(0, len(x_te), 2000)])
    acc_fp32 = (logits_pt.argmax(1) == y_te).float().mean().item()

    tfw = to_tfjs(sd)
    qw, scales = quantise(tfw)
    deq = dequantise(qw, scales)

    # parity check of the layout conversion on 2,000 test images (NumPy NHWC vs PyTorch NCHW)
    xs = x_te[:2000].numpy().transpose(0, 2, 3, 1).astype(np.float64)
    lg_np = np_forward([(n, a.astype(np.float64)) for n, a in tfw], xs)
    diff = float(np.abs(lg_np - logits_pt[:2000].numpy()).max())
    agree = int((lg_np.argmax(1) == logits_pt[:2000].numpy().argmax(1)).sum())
    print(f"layout check: max |logit diff| = {diff:.2e}, argmax agreement {agree}/2000")
    assert diff < 1e-3 and agree == 2000, "TF.js layout conversion does not match PyTorch"

    lg_int8 = np.concatenate([np_forward(deq, x_te[i:i + 2000].numpy().transpose(0, 2, 3, 1))
                              for i in range(0, len(x_te), 2000)])
    acc_int8 = float((lg_int8.argmax(1) == y_te.numpy()).mean())

    os.makedirs(OUT, exist_ok=True)
    fp32 = b"".join(a.astype("<f4").tobytes() for _, a in tfw)
    i8 = b"".join(a.tobytes() for n, a in qw if n.endswith("kernel")) + \
        b"".join(a.astype("<f4").tobytes() for n, a in qw if n.endswith("bias"))
    with open(os.path.join(OUT, "weights_fp32.bin"), "wb") as f:
        f.write(fp32)
    with open(os.path.join(OUT, "weights_int8.bin"), "wb") as f:
        f.write(i8)
    torch.save(model.state_dict(), os.path.join(OUT, "model_fp32.pt"))

    meta = dict(
        name="hfl-hetero-edge global model", version=VERSION,
        exported=datetime.date.today().isoformat(), source_run=cfg.name,
        training="proposed scheme (FedProx + deadline + adaptive workload), 4 gateways x 5 devices, "
                 "60 edge rounds, Dirichlet alpha = 0.5, seed 0, simulated",
        params=int(sum(a.size for _, a in tfw)), classes=CLASSES, input=dict(shape=[28, 28, 1], mean=MEAN, std=STD),
        test_acc_fp32=acc_fp32, test_acc_int8=acc_int8,
        files=dict(fp32=dict(file="weights_fp32.bin", bytes=len(fp32), sha256=hashlib.sha256(fp32).hexdigest()),
                   int8=dict(file="weights_int8.bin", bytes=len(i8), sha256=hashlib.sha256(i8).hexdigest())),
        layout=[dict(name=n, shape=list(a.shape)) for n, a in tfw],
        int8_scales=scales,
        int8_format="int8 kernels in layout order, then float32 little-endian biases in layout order",
    )
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # 300 test images (first 30 of each class) with PyTorch's predictions, for the phone parity check
    raw = __import__("torchvision").datasets.FashionMNIST(os.path.join(ROOT, "data"), train=False).data.numpy()
    idx = np.concatenate([np.where(y_te.numpy() == c)[0][:30] for c in range(10)])
    probs = torch.softmax(logits_pt[idx], 1).numpy()
    samples = dict(note="Fashion-MNIST test images (first 30 per class) and PyTorch reference predictions",
                   version=VERSION, n=len(idx), test_index=idx.tolist(), labels=y_te.numpy()[idx].tolist(),
                   pred_fp32=logits_pt[idx].argmax(1).tolist(), pred_int8=lg_int8[idx].argmax(1).tolist(),
                   prob_fp32=np.round(probs, 4).tolist(),
                   pixels_b64=[base64.b64encode(raw[i].tobytes()).decode() for i in idx])
    with open(os.path.join(ROOT, "deploy", "samples.json"), "w") as f:
        json.dump(samples, f)
    print(json.dumps({k: meta[k] for k in ("version", "source_run", "params", "test_acc_fp32", "test_acc_int8")}, indent=1))
    print("fp32", len(fp32), "bytes; int8", len(i8), "bytes; written to", OUT)


if __name__ == "__main__":
    main()
