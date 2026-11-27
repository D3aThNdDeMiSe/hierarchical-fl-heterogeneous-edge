"""Core invariants of the simulator. Run with `python -m pytest -q`.

The first tests need no data. The last one runs a two-round federation on Fashion-MNIST and is
skipped when the dataset has not been downloaded yet (python run.py downloads it on first use).
"""
import math
import os

import numpy as np
import pytest
import torch

from hfl import data as D
from hfl import devices as DV
from hfl.model import build_model
from hfl.sim import Config, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def synthetic_labels(per_class=600, n_classes=10):
    return np.repeat(np.arange(n_classes), per_class)


@pytest.mark.parametrize("alpha", [None, 0.1, 0.5, 1.0])
def test_partition_uses_every_sample_exactly_once(alpha):
    labels = synthetic_labels()
    parts = D.dirichlet_partition(labels, 20, alpha, np.random.default_rng([0, 0]))
    joined = np.concatenate(parts)
    assert len(parts) == 20
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)
    assert min(len(p) for p in parts) >= 10


def test_partition_is_deterministic_for_a_seed():
    labels = synthetic_labels()
    a = D.dirichlet_partition(labels, 20, 0.1, np.random.default_rng([3, 0]))
    b = D.dirichlet_partition(labels, 20, 0.1, np.random.default_rng([3, 0]))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_streaming_aggregation_equals_weighted_average():
    """Both streaming forms (running weighted sum, running mean) match the batch FedAvg average."""
    g = torch.Generator().manual_seed(0)
    updates = [torch.randn(1000, generator=g) for _ in range(7)]
    weights = [float(n) for n in (120, 3000, 45, 800, 800, 10, 2500)]
    batch = sum(n * w for n, w in zip(weights, updates)) / sum(weights)

    acc, m = torch.zeros(1000), 0.0                 # what hfl.sim does
    for n, w in zip(weights, updates):
        acc.add_(w, alpha=n)
        m += n
    assert torch.allclose(acc / m, batch, atol=1e-5)

    mean, m = torch.zeros(1000), 0.0                # the running mean written in the paper
    for n, w in zip(weights, updates):
        m += n
        mean += (n / m) * (w - mean)
    assert torch.allclose(mean, batch, atol=1e-5)


def test_model_sizes_match_the_paper():
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(build_model("fmnist")) == 80_602
    assert count(build_model("cifar10")) == 87_458


def test_run_names_stay_backward_compatible():
    """Default settings must keep the original file names, so new options never overwrite old runs."""
    base = Config(method="hier_fedprox", alpha=0.5, seed=1)
    assert base.name == "fmnist_hier_fedprox_a0.5_E4K5_mixed_s1"
    assert Config(method="hier_fedprox", alpha=0.5, seed=1, mu=0.1).name.endswith("_mu0.1_s1")
    assert Config(method="hier_proposed", alpha=0.1, deadline_quantile=0.75).name.endswith("_dq0.75_s0")
    assert Config(method="hier_fedavg", alpha=None, profile="measured").name == "fmnist_hier_fedavg_iid_E4K5_mixed_measured_s0"


def test_device_profiles():
    lit, _ = DV.step_times("colext")
    assert math.isclose(lit["low"] / lit["high"], 2.3)
    assert math.isclose(lit["mid"] / lit["high"], math.sqrt(2.3))
    meas, _ = DV.step_times("measured")
    assert 2.8 < meas["mid"] / meas["high"] < 3.0
    assert 5.0 < meas["low"] / meas["high"] < 5.1


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "FashionMNIST")), reason="Fashion-MNIST not downloaded")
def test_byte_accounting_of_a_tiny_federation():
    """Two edge rounds with kappa2 = 2 is one global round: the server sees 2E models, the LANs 2NR."""
    E, K, R = 2, 2, 2
    w_bytes = 4 * 80_602
    common = dict(alpha=0.5, gateways=E, clients_per_gateway=K, rounds=R, kappa2=2, data_root=DATA)
    hier = run(Config(method="hier_fedavg", **common), log=lambda s: None)["final"]
    flat = run(Config(method="flat_fedavg", **common), log=lambda s: None)["final"]
    assert hier["bytes_server"] == 2 * E * w_bytes
    assert hier["bytes_edge"] == 2 * E * K * R * w_bytes
    assert flat["bytes_server"] == 2 * E * K * R * w_bytes
    assert 0.0 < hier["acc"] <= 1.0
