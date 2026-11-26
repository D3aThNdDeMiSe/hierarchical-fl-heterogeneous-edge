"""Run one experiment and save its result as JSON.

Example:
  python run.py --method hier_proposed --alpha 0.5 --E 4 --K 5 --tiers mixed --seed 0
"""
import argparse
import json
import os
import sys

from hfl.sim import METHODS, Config, run


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="hier_proposed", choices=list(METHODS))
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    ap.add_argument("--alpha", default="0.5", help="Dirichlet alpha, or 'iid'")
    ap.add_argument("--E", type=int, default=4, help="number of gateways")
    ap.add_argument("--K", type=int, default=5, help="clients per gateway")
    ap.add_argument("--tiers", default="mixed", choices=["mixed", "mid", "low_heavy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--kappa2", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join("results", "runs"))
    ap.add_argument("--force", action="store_true", help="rerun even if the result exists")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = Config(method=a.method, dataset=a.dataset, alpha=None if a.alpha == "iid" else float(a.alpha),
                 gateways=a.E, clients_per_gateway=a.K, tiers=a.tiers, seed=a.seed, rounds=a.rounds,
                 kappa2=a.kappa2, batch_size=a.bs, lr=a.lr, mu=a.mu)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, cfg.name + ".json")
    if os.path.exists(path) and not a.force:
        print(f"skip {cfg.name} (exists)")
        return 0
    print(f"run {cfg.name}", flush=True)
    res = run(cfg, log=lambda s: print(s, flush=True))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    fin = res["final"]
    print(f"done {cfg.name}: acc {fin['acc']:.4f}, p10 {fin['client_acc_p10']:.4f}, "
          f"sim_time {fin['sim_time']:.0f}s, wall {res['wall_time_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
