"""Live demo: a small three-tier federation trained in about a minute.

Shows the devices behind each emulated Pi-class gateway (tier, data size, dominant classes),
then trains and prints one line per global round: accuracy, worst-served clients, simulated
time, server traffic and dropped updates. Ends with the headline numbers from the full study.

  python demo.py                       # proposed scheme, 2 gateways x 3 devices, 10 rounds
  python demo.py --method hier_fedavg  # the synchronous baseline, for comparison
"""
import argparse
import json
import os
import time

import numpy as np

from hfl import data as D
from hfl import devices as DV
from hfl.sim import METHODS, Config, run

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def bar(x, width=24):
    n = int(round(x * width))
    return "#" * n + "." * (width - n)


def show_setup(cfg):
    """Rebuild the partition and tier assignment exactly as the run will (same RNG streams)."""
    import torchvision
    labels = torchvision.datasets.FashionMNIST(os.path.join(ROOT, "data"), train=True, download=True).targets.numpy()
    N = cfg.gateways * cfg.clients_per_gateway
    rng_part, rng_tier, rng_gw = (np.random.default_rng([cfg.seed, i]) for i in range(3))
    parts = D.dirichlet_partition(labels, N, cfg.alpha, rng_part)
    tiers = DV.assign_tiers(N, cfg.tiers, rng_tier)
    groups = np.array_split(rng_gw.permutation(N), cfg.gateways)
    step_s, _ = DV.step_times(cfg.profile)
    print(f"\n  {cfg.gateways} emulated Pi-class gateways x {cfg.clients_per_gateway} devices, "
          f"Dirichlet alpha = {cfg.alpha}, method = {cfg.method}\n")
    for e, g in enumerate(groups):
        print(f"  Gateway {e + 1}")
        for k in g:
            counts = np.bincount(labels[parts[k]], minlength=10)
            top = ", ".join(f"{CLASSES[c]} {100 * counts[c] / counts.sum():.0f}%" for c in np.argsort(-counts)[:2])
            print(f"    device {k:2d}  {tiers[k]:4s} tier  {1000 * step_s[tiers[k]]:4.0f} ms/step  "
                  f"{len(parts[k]):5d} samples  mostly {top}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="hier_proposed", choices=[m for m in METHODS if m != "local"])
    ap.add_argument("--E", type=int, default=2)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = Config(method=a.method, gateways=a.E, clients_per_gateway=a.K, rounds=a.rounds, alpha=a.alpha,
                 seed=a.seed, data_root=os.path.join(ROOT, "data"))
    show_setup(cfg)

    print("  round   sim time   accuracy                            worst 10%   server MB   dropped")
    t0 = time.time()

    def on_eval(h):
        print(f"  {h['round']:5d}   {h['sim_time']:7.1f} s  {bar(h['acc'])} {100 * h['acc']:5.1f}%"
              f"     {100 * h['client_acc_p10']:5.1f}%     {h['bytes_server'] / 1e6:7.1f}   {h['dropped']:4d}", flush=True)

    res = run(cfg, log=lambda s: None, on_eval=on_eval)
    f = res["final"]
    print(f"\n  Done in {time.time() - t0:.0f} s of real time ({f['sim_time']:.0f} s simulated). "
          f"Only model weights crossed the network: {f['bytes_server'] / 1e6:.1f} MB to the server, "
          f"{f['bytes_edge'] / 1e6:.1f} MB on the gateways' Wi-Fi.")

    stats_path = os.path.join(ROOT, "results", "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            s = json.load(fh)
        et = s.get("equal_time", {})
        print("\n  Full study (Fashion-MNIST, 4 gateways x 5 devices, 3 seeds):")
        if "server_traffic_reduction" in s:
            print(f"    - the edge tier cuts server traffic {s['server_traffic_reduction']:.0f}x with no loss of accuracy")
        if "hier_proposed_0.1" in et and "hier_fedavg_0.1" in et:
            print(f"    - at equal time, proposed vs H-FedAvg at alpha = 0.1: "
                  f"{100 * et['hier_proposed_0.1']:.1f}% vs {100 * et['hier_fedavg_0.1']:.1f}%")
    print()


if __name__ == "__main__":
    main()
