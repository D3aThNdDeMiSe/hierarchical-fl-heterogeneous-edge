"""Run one experiment: centralised, local-only, flat FL or hierarchical FL.

Hierarchical round structure (HierFAVG-style, see docs/DESIGN.md Section 7):
  every edge round, each gateway trains its K clients from its edge model and
  aggregates the on-time updates with a streaming weighted sum; every kappa2 edge
  rounds the server averages the edge models and sends the result back down.
Time is simulated (nothing sleeps); bytes are counted per tier.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import torch

from . import data as D
from . import devices as DV
from .model import Trainer, build_model

METHODS = {
    "centralised": dict(topo="central"),
    "local": dict(topo="local"),
    "flat_fedavg": dict(topo="flat", prox=False, deadline=False, adaptive=False),
    "hier_fedavg": dict(topo="hier", prox=False, deadline=False, adaptive=False),
    "hier_fedprox": dict(topo="hier", prox=True, deadline=False, adaptive=False),
    "hier_fedprox_dl": dict(topo="hier", prox=True, deadline=True, adaptive=False),
    "hier_proposed": dict(topo="hier", prox=True, deadline=True, adaptive=True),
}


@dataclass
class Config:
    method: str = "hier_proposed"
    dataset: str = "fmnist"
    alpha: float | None = 0.5          # Dirichlet concentration; None = IID
    gateways: int = 4                  # E
    clients_per_gateway: int = 5       # K
    tiers: str = "mixed"               # see devices.TIER_MIXES
    seed: int = 0
    rounds: int = 60                   # R edge rounds (flat FL: R rounds)
    kappa2: int = 2                    # edge rounds per global sync
    batch_size: int = 32
    lr: float = 0.05
    mu: float = 0.01                   # FedProx
    tau_min_frac: float = 0.1          # adaptive workload lower bound, as a share of a full epoch
    safety_quantile: float = 0.9       # adaptive workload plans for this quantile of speed jitter
    deadline_quantile: float = 0.5     # D = this quantile of clients' expected full-epoch time
    data_root: str = "data"
    profile: str = "colext"            # device speed ratios: "colext" (literature) or "measured" (own phones)

    @property
    def name(self) -> str:
        a = "iid" if self.alpha is None else f"a{self.alpha:g}"
        p = "" if self.profile == "colext" else f"_{self.profile}"
        d = "" if self.deadline_quantile == 0.5 else f"_dq{self.deadline_quantile:g}"
        return (f"{self.dataset}_{self.method}_{a}_E{self.gateways}K{self.clients_per_gateway}"
                f"_{self.tiers}{p}{d}_s{self.seed}")


def run(cfg: Config, log=print, return_weights: bool = False, on_eval=None) -> dict:
    spec = METHODS[cfg.method]
    final_w = None  # final global weights (centralised / flat / hierarchical), for deployment checks
    wall0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    x_tr, y_tr, x_te, y_te = D.load(cfg.dataset, cfg.data_root, dev)
    n_classes = int(y_tr.max().item()) + 1
    E, K = cfg.gateways, cfg.clients_per_gateway
    N, R = E * K, cfg.rounds

    # Separate RNG streams so partition, tiers, gateway membership and jitter are
    # identical across methods for the same seed (paired comparisons).
    rng_part, rng_tier, rng_gw, rng_jit = (np.random.default_rng([cfg.seed, i]) for i in range(4))
    labels = y_tr.cpu().numpy()
    parts = D.dirichlet_partition(labels, N, cfg.alpha, rng_part)
    P = D.label_distribution(labels, parts, n_classes)
    P_t = torch.tensor(P, dtype=torch.float32, device=dev)
    idx = [torch.as_tensor(p, dtype=torch.long, device=dev) for p in parts]
    n_k = np.array([len(p) for p in parts], dtype=float)
    tiers = DV.assign_tiers(N, cfg.tiers, rng_tier)
    step_s, step_src = DV.step_times(cfg.profile)
    t_nom = np.array([step_s[t] for t in tiers])
    groups = [g for g in np.array_split(rng_gw.permutation(N), E)]
    net = DV.NetConfig()

    model = build_model(cfg.dataset).to(dev)
    trainer = Trainer(model, x_tr, y_tr, cfg.batch_size, cfg.lr, cfg.seed)
    w0 = trainer.flat.clone()
    model_bytes = trainer.n_params * 4
    tau_full = np.ceil(n_k / cfg.batch_size).astype(int)
    tau_min = np.maximum(1, np.ceil(cfg.tau_min_frac * tau_full)).astype(int)
    mu = cfg.mu if spec.get("prox") else 0.0
    # Adaptive workload budgets for a slow round: the gateway plans with the profiled speed
    # inflated to the q-quantile of the log-normal jitter, so ~10% of planned clients run late.
    jitter_margin = float(np.exp(NormalDist().inv_cdf(cfg.safety_quantile) * net.jitter_sigma))

    def evaluate(w):
        acc, loss, per_class = trainer.evaluate(w, x_te, y_te, n_classes)
        ca = (P_t @ per_class).cpu().numpy()
        return dict(acc=acc, loss=loss, client_acc_mean=float(ca.mean()),
                    client_acc_p10=float(np.percentile(ca, 10)), client_acc_min=float(ca.min()),
                    client_acc_std=float(ca.std()))

    history: list[dict] = []
    totals = dict(sim_time=0.0, bytes_server=0, bytes_edge=0, updates=0, dropped=0,
                  steps=0, wasted_steps=0)
    extra: dict = {}

    def record(r, w):
        history.append(dict(round=r, **totals, **evaluate(w)))
        if on_eval is not None:
            on_eval(history[-1])
        if r % 10 == 0 or r == R:
            h = history[-1]
            log(f"  [{cfg.name}] round {r:3d}  acc {h['acc']:.4f}  p10 {h['client_acc_p10']:.4f}"
                f"  sim_t {h['sim_time']:8.1f}s  dropped {h['dropped']}")

    def client_round(clients, w_start, comm, deadline):
        """One synchronous round of `clients` from `w_start`; streaming weighted aggregation."""
        acc_w = torch.zeros_like(w_start)
        n_acc, finishes, late, n_up = 0.0, [], 0, 0
        for k in clients:
            tau = int(tau_full[k])
            if spec["adaptive"]:
                budget = (deadline - comm[k]) / (t_nom[k] * jitter_margin)
                tau = int(np.clip(math.floor(budget), tau_min[k], tau_full[k]))
            finish = comm[k] + tau * t_nom[k] * J[k]
            finishes.append(finish)
            totals["steps"] += tau
            totals["updates"] += 1
            if spec["deadline"] and finish > deadline:
                late += 1
                totals["dropped"] += 1
                totals["wasted_steps"] += tau
                continue  # the update arrives after the deadline and is discarded
            acc_w.add_(trainer.train(w_start, idx[k], tau, mu), alpha=float(n_k[k]))
            n_acc += n_k[k]
            n_up += 1
        w_new = acc_w / n_acc if n_acc > 0 else None
        t_round = deadline if (spec["deadline"] and late > 0) else max(finishes)
        return w_new, t_round, n_up

    topo = spec["topo"]
    if topo == "central":
        all_idx = torch.cat(idx)
        steps = math.ceil(len(all_idx) / cfg.batch_size)
        w = w0.clone()
        for r in range(1, R + 1):
            w = trainer.train(w, all_idx, steps)
            totals["steps"] += steps
            if r % cfg.kappa2 == 0 or r == R:
                record(r, w)

    elif topo == "local":
        g_acc, own_acc = [], []
        for k in range(N):
            w = w0.clone()
            for _ in range(R):
                w = trainer.train(w, idx[k], int(tau_full[k]))
            acc, _, per_class = trainer.evaluate(w, x_te, y_te, n_classes)
            g_acc.append(acc)
            own_acc.append(float(P_t[k] @ per_class))
        own = np.array(own_acc)
        totals["steps"] = int(R * tau_full.sum())
        history.append(dict(round=R, **totals, acc=float(np.mean(g_acc)), loss=float("nan"),
                            client_acc_mean=float(own.mean()), client_acc_p10=float(np.percentile(own, 10)),
                            client_acc_min=float(own.min()), client_acc_std=float(own.std())))
        extra["local_global_acc_per_client"] = g_acc
        log(f"  [{cfg.name}] local-only: mean global acc {np.mean(g_acc):.4f}, mean own-dist acc {own.mean():.4f}")

    elif topo == "flat":
        wan = np.array([DV.mbps_to_bytes_per_s(DV.WAN_MBPS[t]) for t in tiers])
        comm = 2 * model_bytes / wan
        deadline = float(np.quantile(comm + tau_full * t_nom, cfg.deadline_quantile))
        extra["deadline_s"] = deadline
        w = w0.clone()
        for r in range(1, R + 1):
            J = rng_jit.lognormal(0.0, net.jitter_sigma, N)
            w_new, t_round, _ = client_round(range(N), w, comm, deadline)
            if w_new is not None:
                w = w_new
            totals["sim_time"] += t_round
            totals["bytes_server"] += 2 * N * model_bytes
            record(r, w)

    elif topo == "hier":
        lan = DV.mbps_to_bytes_per_s(net.lan_total_mbps) / K   # fair share of the gateway's Wi-Fi
        comm = np.full(N, 2 * model_bytes / lan)
        deadline = float(np.quantile(comm + tau_full * t_nom, cfg.deadline_quantile))
        gw_rate = DV.mbps_to_bytes_per_s(net.gw_wan_mbps)
        n_e = np.array([n_k[g].sum() for g in groups])
        w_g = w0.clone()
        w_e = [w_g.clone() for _ in range(E)]
        gw_clock = np.zeros(E)
        agg_times, edge_round_times = [], []
        for r in range(1, R + 1):
            J = rng_jit.lognormal(0.0, net.jitter_sigma, N)
            for e, g in enumerate(groups):
                w_new, t_round, n_up = client_round(g, w_e[e], comm, deadline)
                if w_new is not None:
                    w_e[e] = w_new  # else: no update arrived in time, keep the edge model
                agg = DV.gateway_agg_time(n_up, model_bytes, net)
                gw_clock[e] += t_round + agg
                agg_times.append(agg)
                edge_round_times.append(t_round + agg)
                totals["bytes_edge"] += 2 * len(g) * model_bytes
            if r % cfg.kappa2 == 0 or r == R:
                totals["sim_time"] += float(np.max(gw_clock + model_bytes / gw_rate) + model_bytes / gw_rate)
                gw_clock[:] = 0.0
                w_g = sum(float(n_e[e]) * w_e[e] for e in range(E)) / float(n_e.sum())
                w_e = [w_g.clone() for _ in range(E)]
                totals["bytes_server"] += 2 * E * model_bytes
                record(r, w_g)
        extra.update(deadline_s=deadline, jitter_margin=jitter_margin, gw_agg_time_mean_s=float(np.mean(agg_times)),
                     edge_round_time_mean_s=float(np.mean(edge_round_times)),
                     gw_mem_streaming_bytes=2 * model_bytes, gw_mem_buffered_bytes=K * model_bytes)
    else:
        raise ValueError(topo)

    if topo in ("central", "flat"):
        final_w = w
    elif topo == "hier":
        final_w = w_g
    final = history[-1]
    result = dict(
        name=cfg.name, config=asdict(cfg), method_spec=spec, history=history, final=final,
        best_acc=max(h["acc"] for h in history),
        setup=dict(n_clients=N, model_params=trainer.n_params, model_bytes=model_bytes,
                   tier_counts={t: tiers.count(t) for t in ("low", "mid", "high")},
                   samples_per_client=dict(min=int(n_k.min()), median=float(np.median(n_k)), max=int(n_k.max())),
                   step_source=step_src, step_s=step_s, device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                   **extra),
        wall_time_s=time.time() - wall0,
    )
    if return_weights:
        result["weights"] = None if final_w is None else final_w.detach().cpu()
    return result
