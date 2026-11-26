"""The shared model and a fast flat-parameter trainer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SmallCNN(nn.Module):
    """Conv(16)-Pool-Conv(32)-Pool-FC(hidden)-FC(classes). No BatchNorm, so averaging is exact.

    Fashion-MNIST (1x28x28, hidden=48): 80,602 parameters, 315 KB in float32.
    Mirrored in phone_benchmark/index.html.
    """

    def __init__(self, in_ch: int = 1, img: int = 28, n_classes: int = 10, hidden: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * (img // 4) ** 2, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def build_model(dataset: str) -> SmallCNN:
    if dataset == "fmnist":
        return SmallCNN(in_ch=1, img=28, hidden=48)
    if dataset == "cifar10":
        return SmallCNN(in_ch=3, img=32, hidden=40)  # 87,458 params, still < 100k
    raise ValueError(dataset)


class Trainer:
    """Trains one shared model object for many clients without re-allocating.

    The model's parameters are views into one flat buffer, so loading a client's
    starting weights is a single copy and returning its result is a single clone.
    """

    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 batch_size: int, lr: float, seed: int):
        self.model = model
        self.params = list(model.parameters())
        self.flat = parameters_to_vector(self.params).detach().clone()
        vector_to_parameters(self.flat, self.params)  # params now view into self.flat
        self.x, self.y = x, y
        self.bs, self.lr = batch_size, lr
        self.gen = torch.Generator(device=x.device)
        self.gen.manual_seed(seed)
        self.sizes = [p.numel() for p in self.params]

    @property
    def n_params(self) -> int:
        return self.flat.numel()

    def _anchors(self, w: torch.Tensor):
        return [t.view_as(p) for t, p in zip(torch.split(w, self.sizes), self.params)]

    def train(self, w0: torch.Tensor, idx: torch.Tensor, steps: int, mu: float = 0.0) -> torch.Tensor:
        """Run `steps` SGD steps on samples `idx`, starting from flat weights `w0`.

        mu > 0 adds the FedProx proximal term (mu/2)||w - w0||^2.
        Batches are drawn from one shuffled pass; steps <= ceil(len(idx)/bs).
        """
        self.flat.copy_(w0)
        if steps <= 0:
            return self.flat.clone()
        anchors = self._anchors(w0) if mu > 0 else None
        perm = idx[torch.randperm(len(idx), device=idx.device, generator=self.gen)]
        bs, lr = self.bs, self.lr
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            if len(b) == 0:
                break
            loss = F.cross_entropy(self.model(self.x[b]), self.y[b])
            grads = torch.autograd.grad(loss, self.params)
            with torch.no_grad():
                if anchors is not None:
                    diff = torch._foreach_sub(self.params, anchors)
                    torch._foreach_add_(self.params, grads, alpha=-lr)
                    torch._foreach_add_(self.params, diff, alpha=-lr * mu)
                else:
                    torch._foreach_add_(self.params, grads, alpha=-lr)
        return self.flat.clone()

    @torch.no_grad()
    def evaluate(self, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_classes: int, bs: int = 1000):
        """Return (accuracy, mean loss, per-class accuracy tensor) of weights `w` on (x, y)."""
        self.flat.copy_(w)
        preds, loss_sum = [], 0.0
        for i in range(0, len(x), bs):
            out = self.model(x[i:i + bs])
            loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
            preds.append(out.argmax(1))
        pred = torch.cat(preds)
        correct = pred == y
        per_class = torch.bincount(y[correct], minlength=n_classes).float() / \
            torch.bincount(y, minlength=n_classes).clamp(min=1).float()
        return correct.float().mean().item(), loss_sum / len(x), per_class
