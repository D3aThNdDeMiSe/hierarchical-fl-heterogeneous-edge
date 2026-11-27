# Design: Hierarchical FL with Heterogeneous Devices and Pi-Class Gateways

This file is the single source of truth for the simulator and the paper's methodology.
Every simulation parameter is listed with its source; values marked **assumption** are
stated as such in the paper.

## 1. Scope (course version)

- Three tiers: **devices → Pi-class edge gateways (emulated) → server**. Only model weights move.
- Simulation in plain PyTorch, with a simulated clock and byte counters. Flower is not used,
  because it has no native multi-level aggregation (risk anticipated in proposal Table 3).
- Raspberry Pis were not allotted, so gateways are **emulated** with Pi-class resource
  profiles. Device speeds use published phone measurements (CoLExT [4]) and, when available,
  measurements of our own phones (`phone_benchmark/`).
- Dataset: **Fashion-MNIST** (used by FedAda [8], ESPerHFL [10], MTGC [7], HAF-Edge [13]),
  plus **CIFAR-10** as a harder second dataset (G6).

## 2. Research questions → experiments → metrics

| RQ | Question | Experiment | Metrics that answer it |
|---|---|---|---|
| RQ1 | How much do statistical and system heterogeneity degrade accuracy and convergence relative to centralised training? | α ∈ {0.1, 0.5, 1.0} (+ IID sanity) and tier mix ∈ {homogeneous-mid, mixed, low-heavy} | Gap to centralised accuracy; rounds and simulated time to target |
| RQ2 | Does a Pi-class edge tier reduce server-side communication and round latency compared with flat FL, without loss of accuracy? | Flat FedAvg vs. hierarchical FedAvg, same local computation | Bytes to/from server, simulated round time, final accuracy |
| RQ3 | Do FedProx, deadline-based partial aggregation and adaptive local workload improve accuracy, worst-device accuracy and time-to-accuracy over FedAvg? | Ablation: H-FedAvg → +FedProx → +deadline → +adaptive (= proposed) | Final accuracy, 10th-percentile client accuracy, time-to-target, dropped updates, wasted steps |
| RQ4 | How do accuracy, round time and gateway load change as the number of devices per gateway and the number of gateways grow? | Clients per gateway K ∈ {5, 10, 25} at E = 4; gateways E ∈ {2, 4, 10} at K = 5 | Final accuracy, simulated round time, gateway aggregation time and memory, bytes |

RQ5 (edge-level DP) remains a stretch goal / future work.

Primary predictive metric: **top-1 test accuracy** (as reported by the base papers).
Heterogeneity/fairness metric: **10th-percentile client accuracy** (global model evaluated on
each client's label distribution, see Section 4).

## 3. Device counts (professor feedback 3)

| Paper | Gateways | Clients |
|---|---|---|
| FedAda [8] | 4 | 20 |
| HAF-Edge [13] | up to 5 | 11 per gateway |
| ESPerHFL [10] | 10 | 50 |
| MTGC [7] | 10 | 100 |
| FedUC [9] | 16 | 100 |

**Main configuration (fixed):** E = 4 gateways × K = 5 clients = 20 clients (matches FedAda).
**Scaling sweep (gradually increasing):** K ∈ {5, 10, 25} → N up to 100 (MTGC/FedUC scale),
and E ∈ {2, 4, 10}. The full dataset is split across all clients, as in the base papers.

## 4. Data and partitioning

- Fashion-MNIST, 60k train / 10k test, normalised (mean 0.2860, std 0.3530).
- **Dirichlet label skew**: for each class, its samples are split across N clients with
  proportions ~ Dir(α), with the standard balancing step (a client already holding ≥ N_total/N
  samples receives no more of the class), resampled until every client has ≥ 10 samples
  (as in NIID-Bench). IID = uniform random split.
- Clients train on all their samples. **Per-client accuracy** is the global model's accuracy
  under that client's own label distribution: `acc_k = Σ_c p_{k,c} · acc_c`, where `acc_c` is
  the per-class accuracy on the official test set and `p_{k,c}` the client's label proportions.
  This is exact (no sampling noise) and captures how well the shared model serves each skewed
  client. Global accuracy uses the official 10k test set.
- Clients are assigned to gateways uniformly at random (K per gateway).

## 5. Model

SmallCNN: Conv(1→16, 3×3, same) → ReLU → MaxPool 2 → Conv(16→32, 3×3, same) → ReLU →
MaxPool 2 → FC(1568→48) → ReLU → FC(48→10). **80,602 parameters, 315 KB float32**
(meets the proposal's < 100k parameters / < 1 MB target). No BatchNorm, so weights are plain
tensors and averaging is exact. The same architecture is in `phone_benchmark/index.html`.

Local training: SGD, lr 0.05, batch 32, 1 local epoch per edge round (τ_full = ⌈n_k / 32⌉ steps).
FedProx: μ = 0.01.

## 6. Device and network profiles

| Parameter | Value | Source |
|---|---|---|
| Device tiers | low / mid / high | Proposal (budget / mid-range / flagship) |
| Time per step, high tier | t_high = 20 ms | **assumption** (sets the time scale only) |
| Low/high compute ratio | 2.3× (literature profile); 5.06× (measured profile) | CoLExT [4]: OnePlus Nord 2T vs Pixel 7; own phones: Vivo Y31 vs Xiaomi POCO F3 |
| Mid/high compute ratio | 1.52× (literature, geometric mean); 2.90× (measured) | **assumption**; own phones: Redmi Note 12 vs POCO F3 |
| Per-round speed jitter | multiplicative log-normal, σ = 0.2 | **assumption** (throttling, background load; CoLExT [4] reports run-to-run variation) |
| Phone WAN uplink (flat FL) | high 20, mid 12, low 6.25 Mbit/s | high/low ratio 3.2× from CoLExT [4] (Xiaomi 12 upload); absolute values **assumption** |
| Gateway Wi-Fi LAN (shared by its K clients) | 50 Mbit/s total, fair share 50/K per client | **assumption** (Pi 4 onboard 802.11ac, practical throughput) |
| Gateway WAN uplink to server | 20 Mbit/s | **assumption** (broadband) |
| Gateway aggregation | 5 ms per update + bytes / 400 MB/s | **assumption** (memory-bound weighted sum on Pi-class ARM; FLEdge [15] reports embedded boards ≈4× slower than desktop) |
| Gateway RAM | 4 GB | Raspberry Pi 4 Model B (4 GB) |

Tier mixes: `mixed` = 1/3 each (default); `mid` = all mid; `low_heavy` = 60% low, 30% mid, 10% high.

## 7. Cost model (simulated clock)

Notation: |w| = model size in bytes, t_k = client k's nominal time per step, j_k ~ LogN(0, 0.2²)
the round's jitter, τ_k local steps, b = link rate.

- Client finish time in an edge round: `T_k = |w|/b_down + τ_k · t_k · j_k + |w|/b_up`
  (LAN rates for hierarchical, WAN rates for flat).
- **Synchronous (no deadline):** gateway e waits for all clients: `T_e = max_k T_k + A_e`.
- **Deadline D:** updates with `T_k > D` are dropped from that round (their steps count as
  wasted); `T_e = (D if any client was late, else max_k T_k) + A_e`.
- **Adaptive local workload** (FedCompass-style [11]): the gateway plans with the profiled speed
  inflated by a safety margin m = exp(z₀.₉·σ) ≈ 1.29 (the 90th percentile of the jitter):
  `τ_k = clip(⌊(D − c_k) / (m·t_k)⌋, τ_min, τ_full)` with c_k the client's communication time
  and τ_min = ⌈0.1·τ_full⌉. Jitter can still make a client miss D (≈10% of planned clients).
- Deadline D = the median client's expected full-epoch finish time (computed once per run
  from nominal profiles), so roughly half the clients cannot finish a full epoch in time.
- Gateway aggregation: `A_e = n_updates · (5 ms + |w| / 400 MB/s)`. **Streaming** aggregation
  (running weighted sum) keeps gateway memory at 2·|w| regardless of K; a buffered gateway
  would need K·|w| (reported analytically for larger models, e.g. ResNet-18).
- **Global sync every κ₂ = 2 edge rounds:** `T_global = max_e (Σ_{r=1}^{κ₂} T_e^{(r)} + |w|/B_gw) + |w|/B_gw`.
- **Flat FedAvg:** each round, all N clients exchange weights with the server over WAN;
  `T_round = max_k T_k` (WAN rates); server aggregation cost ≈ 0 (PC).
- Bytes: server-side bytes (WAN, up + down) and edge bytes (LAN, up + down) are counted per round.

Aggregation weights: sample-weighted FedAvg at both tiers
(`w_e = Σ n_k w_k / Σ n_k` over on-time clients; `w = Σ n_e w_e / Σ n_e` with n_e = gateway's
total samples). If no client of a gateway arrives on time, the gateway keeps its previous model.

## 8. Methods

| Name | Topology | Local objective | Deadline | Adaptive τ |
|---|---|---|---|---|
| `centralised` | – (pooled data, upper bound) | CE | – | – |
| `local` | – (each client alone, lower bound) | CE | – | – |
| `flat_fedavg` | devices → server | CE | no | no |
| `hier_fedavg` | devices → gateway → server | CE | no | no |
| `hier_fedprox` | hierarchical | CE + FedProx | no | no |
| `hier_fedprox_dl` | hierarchical | CE + FedProx | yes | no |
| `hier_proposed` | hierarchical | CE + FedProx | yes | yes |

All methods get the same budget of R = 60 edge rounds (flat: 60 rounds), i.e. the same
maximum local computation. Evaluation happens at every global sync (flat: every round).

## 9. Experiment grid

- **G1 main** (RQ1, RQ2, RQ3): E=4, K=5, tiers `mixed`, α ∈ {0.1, 0.5, 1.0}, all FL methods +
  centralised + local, seeds {0, 1, 2}. IID sanity: flat_fedavg and hier_fedavg on IID.
- **G2 system heterogeneity** (RQ1): α=0.5, tiers ∈ {`mid`, `low_heavy`},
  methods {hier_fedavg, hier_fedprox_dl, hier_proposed}, seeds {0, 1, 2}.
- **G3 scaling** (RQ4): α=0.5, tiers `mixed`, (E, K) ∈ {(4,10), (4,25), (2,5), (10,5)},
  methods {flat_fedavg, hier_fedavg, hier_proposed}, seeds {0, 1}.
- **G4 measured profile** (RQ1, robustness of RQ3): as G1 (E=4, K=5, tiers `mixed`,
  α ∈ {0.1, 0.5, 1.0}) but with speed ratios measured on our phones
  (`phone_benchmark/measurements.json`: POCO F3 1.0, Redmi Note 12 2.90, Vivo Y31 5.06),
  methods {hier_fedavg, hier_fedprox_dl, hier_proposed}, seeds {0, 1, 2}. 27 runs; 138 in total.

- **G5 deadline sensitivity** (RQ3): hier_proposed with deadline quantile q ∈ {0.75, 0.9}
  (q = 0.5 is in G1), α ∈ {0.1, 0.5}, seeds {0, 1, 2}. 12 runs.
- **G6 CIFAR-10** (second dataset shared by FedAda, FedUC, ESPerHFL, MTGC): E=4, K=5, tiers `mixed`,
  α ∈ {0.1, 0.5}, methods {centralised, flat_fedavg, hier_fedavg, hier_fedprox_dl, hier_proposed},
  seeds {0, 1, 2}. SmallCNN with 3 input channels and hidden 40 (87,458 params). 30 runs.
- **G7 FedProx strength** (RQ3): hier_fedprox with μ ∈ {0.1, 1.0} (μ = 0.01 is in G1), E=4, K=5,
  tiers `mixed`, α ∈ {0.1, 0.5}, seeds {0, 1, 2}. 12 runs; 192 in total.

Phone measurements (TF.js CPU backend, batch 32, median ms/step; inference ms/sample):
POCO F3 1,607 / 3.8; Redmi Note 12 4,665 / 9.3; Vivo Y31 8,124 / 25.5.
14 more phones are in `phone_benchmark/measurements_extra.json` (17 in total; raw runs and the two
excluded ones in `phone_benchmark/raw/`). Across all 17: 1.0-13.9 s per step, a 6.0x spread between
the 10th and 90th percentile; fast/middle/slow groups of 5/7/5 phones have median ratios
1 : 2.2 : 5.5, close to the three-phone profile above, so G4 keeps that profile.

Results are reported as mean ± std over seeds; method differences are checked with paired
t-tests over seeds and α settings.
