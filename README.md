# Federated Learning with Heterogeneous Edge Devices

A three-tier hierarchical federated learning (HFL) study: heterogeneous devices train locally,
**emulated Raspberry Pi-class edge gateways** aggregate their updates, and a server combines the
gateway models. Only model weights move between tiers.

Course project for Edge Computing (R&D track), Department of Artificial Intelligence,
FAST-NUCES Islamabad, Fall 2026.
Authors: **Immad Shah** (I22-0395) and **Sarim Mansoor** (I22-0532).

![Architecture](paper/figures/fig1_architecture.png)

## What this repository contains

| Path | What it is |
|---|---|
| `hfl/` | The simulator: data partitioning, model, device/gateway profiles, FL algorithms |
| `run.py` | Runs one experiment and saves `results/runs/<name>.json` |
| `sweep.py` | Runs the full experiment grid in parallel (resumable) |
| `analyze.py` | Builds the paper's figures, tables and key numbers from the results |
| `phone_benchmark/` | Browser page that measures per-step training time of the same CNN on a real phone |
| `docs/DESIGN.md` | Design spec: research questions, every parameter with its source, cost model, grid |
| `paper/` | IEEE conference paper (LaTeX) |
| `results/` | Run results (JSON), `summary.txt`, `stats.json` |

## Research questions

| RQ | Question | Experiment |
|---|---|---|
| RQ1 | How much do statistical and system heterogeneity cost? | Dirichlet α ∈ {0.1, 0.5, 1.0}, IID; device-tier mix all-mid / mixed / low-heavy |
| RQ2 | Does a Pi-class edge tier reduce server traffic and round time without losing accuracy? | Flat vs hierarchical FedAvg |
| RQ3 | Do FedProx, deadline-based aggregation and adaptive local workload help? | Ablation: H-FedAvg → +FedProx → +deadline → +adaptive |
| RQ4 | How does the system scale with devices per gateway and number of gateways? | K ∈ {5, 10, 25} at E = 4; E ∈ {2, 4, 10} at K = 5 |

Main configuration: 4 gateways × 5 devices (as in FedAda), Fashion-MNIST, a CNN with 80,602
parameters (315 KB), 60 edge rounds with a global sync every 2.

## How the simulation works

- **Devices** belong to low, mid or high tiers. The compute gap (2.3×) and upload gap (3.2×)
  between tiers come from the CoLExT phone testbed; per-round speed varies with log-normal jitter.
- **Gateways** are emulated Raspberry Pi 4-class nodes: Wi-Fi shared by their K devices,
  a 4 GB RAM budget, a per-update aggregation cost, and a broadband uplink to the server.
  They aggregate with a streaming weighted average (memory 2|w| instead of K|w|).
- **Time is simulated** (nothing sleeps) and **bytes are counted** per tier, so runs are fast
  while still reporting round latency and communication.
- **Heterogeneity-aware mechanisms:** FedProx, a round deadline (late updates are dropped) and
  FedCompass-style adaptive local workload (slow devices get fewer local steps).

All parameters and their sources are in [`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

Requires Python 3.10+ and PyTorch (a CUDA GPU is optional but much faster).

```bash
pip install -r requirements.txt

# one experiment (downloads Fashion-MNIST into data/ on first use)
python run.py --method hier_proposed --alpha 0.5 --E 4 --K 5 --tiers mixed --seed 0

# the full grid (111 runs; about 3 hours on an RTX 5060 laptop GPU with 4 workers)
python sweep.py --workers 4

# figures, tables and statistics for the paper
python analyze.py
```

Methods: `centralised`, `local`, `flat_fedavg`, `hier_fedavg`, `hier_fedprox`,
`hier_fedprox_dl`, `hier_proposed`.

## Measuring a real phone

Serve `phone_benchmark/` on your network (`python -m http.server 8000 --directory phone_benchmark`),
open `http://<your-PC-IP>:8000` on the phone and press **Run benchmark**. It trains the same CNN
with TensorFlow.js and reports the median time per training step. To calibrate the simulator, save
the low-end and mid-range results as `phone_benchmark/calibration.json`:

```json
{"low_ms": 0.0, "mid_ms": 0.0}
```

## Building the paper

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

## Scope note

The project proposal planned a physical Raspberry Pi gateway and Android phones. Raspberry Pis were
not allotted, so the gateway tier is emulated with Pi-class resource profiles. Following proposal
feedback, the evaluation uses the base papers' dataset (Fashion-MNIST) and device counts, and is
organised around the research questions above.
