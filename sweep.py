"""Run the experiment grid (docs/DESIGN.md Section 9) with several worker processes.

Jobs are ordered seed-major, so after the first third every configuration has one
result. Finished runs are skipped, so the sweep can be stopped and restarted.
Progress is written to results/sweep_status.json.

  python sweep.py --workers 5            # everything
  python sweep.py --groups G1 --dry-run  # list the G1 jobs
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from hfl.sim import Config

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "results", "runs")
LOGS = os.path.join(ROOT, "results", "logs")
STATUS = os.path.join(ROOT, "results", "sweep_status.json")
FL_METHODS = ["flat_fedavg", "hier_fedavg", "hier_fedprox", "hier_fedprox_dl", "hier_proposed"]


def grid():
    jobs = []
    for seed in (0, 1, 2):
        # G1 main: RQ1 (statistical), RQ2, RQ3
        for alpha in ("0.1", "0.5", "1.0"):
            for m in FL_METHODS + ["centralised", "local"]:
                jobs.append(dict(group="G1", method=m, alpha=alpha, E=4, K=5, tiers="mixed", seed=seed))
        for m in ("flat_fedavg", "hier_fedavg"):  # IID sanity check
            jobs.append(dict(group="G1", method=m, alpha="iid", E=4, K=5, tiers="mixed", seed=seed))
        # G2 system heterogeneity: RQ1
        for tiers in ("mid", "low_heavy"):
            for m in ("hier_fedavg", "hier_fedprox_dl", "hier_proposed"):
                jobs.append(dict(group="G2", method=m, alpha="0.5", E=4, K=5, tiers=tiers, seed=seed))
        # G3 scaling: RQ4 (two seeds)
        if seed < 2:
            for E, K in ((4, 10), (4, 25), (2, 5), (10, 5)):
                for m in ("flat_fedavg", "hier_fedavg", "hier_proposed"):
                    jobs.append(dict(group="G3", method=m, alpha="0.5", E=E, K=K, tiers="mixed", seed=seed))
        # G4 measured device profile (speed ratios from our own phones): RQ1, robustness of RQ3
        for alpha in ("0.1", "0.5", "1.0"):
            for m in ("hier_fedavg", "hier_fedprox_dl", "hier_proposed"):
                jobs.append(dict(group="G4", method=m, alpha=alpha, E=4, K=5, tiers="mixed", seed=seed,
                                 profile="measured"))
        # G5 deadline sensitivity (RQ3): looser deadlines for the proposed scheme (q = 0.5 is in G1)
        for alpha in ("0.1", "0.5"):
            for dq in (0.75, 0.9):
                jobs.append(dict(group="G5", method="hier_proposed", alpha=alpha, E=4, K=5, tiers="mixed",
                                 seed=seed, dq=dq))
        # G6 CIFAR-10 (the other dataset shared by the base papers): main configuration
        for alpha in ("0.1", "0.5"):
            for m in ("centralised", "flat_fedavg", "hier_fedavg", "hier_fedprox_dl", "hier_proposed"):
                jobs.append(dict(group="G6", method=m, alpha=alpha, E=4, K=5, tiers="mixed", seed=seed,
                                 dataset="cifar10"))
        # G7 FedProx strength (RQ3): mu = 0.01 is in G1; stronger proximal terms for H-FedProx
        for alpha in ("0.1", "0.5"):
            for mu in (0.1, 1.0):
                jobs.append(dict(group="G7", method="hier_fedprox", alpha=alpha, E=4, K=5, tiers="mixed",
                                 seed=seed, mu=mu))
    for j in jobs:
        j.setdefault("profile", "colext")
        j.setdefault("dq", 0.5)
        j.setdefault("dataset", "fmnist")
        j.setdefault("mu", 0.01)
        j["name"] = Config(method=j["method"], alpha=None if j["alpha"] == "iid" else float(j["alpha"]),
                           gateways=j["E"], clients_per_gateway=j["K"], tiers=j["tiers"], seed=j["seed"],
                           profile=j["profile"], deadline_quantile=j["dq"], dataset=j["dataset"],
                           mu=j["mu"]).name
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--groups", default="G1,G2,G3,G4,G5")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    groups = set(a.groups.split(","))
    jobs = [j for j in grid() if j["group"] in groups]
    todo = [j for j in jobs if not os.path.exists(os.path.join(RUNS, j["name"] + ".json"))]
    print(f"{len(jobs)} jobs in {sorted(groups)}, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if a.dry_run:
        for j in todo:
            print(" ", j["name"])
        return 0

    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    lock = threading.Lock()
    state = dict(total=len(jobs), done=len(jobs) - len(todo), failed=[], running=[],
                 started=time.strftime("%Y-%m-%d %H:%M:%S"), durations=[])

    def write_status():
        d = dict(state)
        left = state["total"] - state["done"] - len(state["failed"])
        if state["durations"]:
            per_job = sum(state["durations"]) / len(state["durations"]) / a.workers
            d["eta_min"] = round(left * per_job / 60, 1)
        d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        d.pop("durations")
        tmp = STATUS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, STATUS)

    def work(item):
        i, j = item
        if i < a.workers:
            time.sleep(4 * i)  # stagger CUDA initialisation; simultaneous starts can fail on Windows
        args = [sys.executable, os.path.join(ROOT, "run.py"), "--method", j["method"], "--alpha", j["alpha"],
                "--E", str(j["E"]), "--K", str(j["K"]), "--tiers", j["tiers"], "--seed", str(j["seed"]),
                "--profile", j["profile"], "--deadline-q", str(j["dq"]), "--dataset", j["dataset"],
                "--mu", str(j["mu"])]
        with lock:
            state["running"].append(j["name"])
            write_status()
        t0 = time.time()
        for attempt in range(3):  # retry transient "CUDA device busy or unavailable" start-up errors
            with open(os.path.join(LOGS, j["name"] + ".log"), "w") as log:
                rc = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                break
            time.sleep(5 + 5 * attempt)
        with lock:
            state["running"].remove(j["name"])
            if rc == 0:
                state["done"] += 1
                state["durations"].append(time.time() - t0)
            else:
                state["failed"].append(j["name"])
            write_status()
            print(f"[{state['done']}/{state['total']}] {'ok ' if rc == 0 else 'FAIL'} {j['name']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    write_status()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, enumerate(todo)))
    print(f"finished: {state['done']}/{state['total']} done, {len(state['failed'])} failed", flush=True)
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
