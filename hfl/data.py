"""Dataset loading (kept on the GPU) and federated partitioning."""
from __future__ import annotations

import numpy as np
import torch

_STATS = {
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def load(name: str, root: str, device: torch.device):
    """Return (x_train, y_train, x_test, y_test) as normalised tensors on `device`."""
    import torchvision

    if name == "fmnist":
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        to_x = lambda ds: ds.data.float().div(255).unsqueeze(1)  # N,1,28,28
        to_y = lambda ds: ds.targets.clone().long()
    elif name == "cifar10":
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        to_x = lambda ds: torch.from_numpy(ds.data).float().div(255).permute(0, 3, 1, 2)  # N,3,32,32
        to_y = lambda ds: torch.tensor(ds.targets, dtype=torch.long)
    else:
        raise ValueError(f"unknown dataset {name!r}")

    mean, std = (torch.tensor(v).view(1, -1, 1, 1) for v in _STATS[name])
    out = []
    for ds in (tr, te):
        x = (to_x(ds) - mean) / std
        out += [x.to(device), to_y(ds).to(device)]
    return tuple(out)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float | None,
                        rng: np.random.Generator, min_size: int = 10) -> list[np.ndarray]:
    """Split sample indices across clients.

    alpha=None -> IID (uniform random, equal sizes).
    Otherwise label skew: for each class, proportions ~ Dir(alpha) over clients, with the
    standard balancing step (clients already holding >= N/n samples get no more of the
    class), resampled until every client has at least `min_size` samples.
    """
    n = len(labels)
    if alpha is None:
        perm = rng.permutation(n)
        return [np.sort(p) for p in np.array_split(perm, n_clients)]

    n_classes = int(labels.max()) + 1
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            p = rng.dirichlet(np.full(n_clients, alpha))
            sizes = np.array([len(b) for b in buckets])
            p = p * (sizes < n / n_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [np.sort(np.array(b)) for b in buckets]
    raise RuntimeError("could not satisfy min_size; lower min_size or raise alpha")


def label_distribution(labels: np.ndarray, parts: list[np.ndarray], n_classes: int) -> np.ndarray:
    """Row-stochastic matrix P[k, c] = share of client k's samples with label c."""
    P = np.zeros((len(parts), n_classes))
    for k, idx in enumerate(parts):
        P[k] = np.bincount(labels[idx], minlength=n_classes) / max(len(idx), 1)
    return P
