"""Device tiers, emulated Pi-class gateways and the network/time model.

Every number here is listed, with its source, in docs/DESIGN.md (Section 6).
Values marked ASSUMPTION are stated as assumptions in the paper.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

# CoLExT [4]: OnePlus Nord 2T needed ~2.3x the round time of a Pixel 7 (CPU underclocking),
# and a Xiaomi 12 needed ~3.2x longer to transmit its update.
COLEXT_COMPUTE_GAP = 2.3
COLEXT_UPLOAD_GAP = 3.2

# ASSUMPTION: absolute time per training step (batch 32) on a high-tier phone.
# Only ratios between tiers affect the comparisons; this sets the time axis' scale.
HIGH_TIER_STEP_S = 0.020

# ASSUMPTION (absolute values); the high/low ratio is CoLExT's 3.2x.
WAN_MBPS = {"high": 20.0, "mid": 12.0, "low": 20.0 / COLEXT_UPLOAD_GAP}

TIER_MIXES = {
    "mixed": {"low": 1 / 3, "mid": 1 / 3, "high": 1 / 3},
    "mid": {"mid": 1.0},
    "low_heavy": {"low": 0.6, "mid": 0.3, "high": 0.1},
}

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "..", "phone_benchmark", "calibration.json")


@dataclass
class NetConfig:
    lan_total_mbps: float = 50.0     # ASSUMPTION: Pi 4 onboard Wi-Fi, shared by its K clients
    gw_wan_mbps: float = 20.0        # ASSUMPTION: gateway uplink/downlink to the server
    agg_overhead_s: float = 0.005    # ASSUMPTION: per-update receive/deserialise cost on the Pi
    agg_bytes_per_s: float = 400e6   # ASSUMPTION: memory-bound weighted sum on Pi-class ARM
    jitter_sigma: float = 0.2        # ASSUMPTION: per-round log-normal speed variation
    gw_ram_bytes: float = 4e9        # Raspberry Pi 4 Model B (4 GB)


def step_times(calibration_path: str | None = CALIBRATION_FILE) -> tuple[dict, str]:
    """Nominal seconds per local step for each tier, plus where the ratios came from.

    Without calibration: low = 2.3x high (CoLExT), mid = geometric mean of the two.
    With phone_benchmark/calibration.json ({"low_ms": .., "mid_ms": ..}) from our own
    phones: the measured low/mid ratio places the mid tier; low stays 2.3x high.
    """
    low = COLEXT_COMPUTE_GAP
    mid = math.sqrt(COLEXT_COMPUTE_GAP)
    source = "CoLExT ratio; mid tier = geometric mean (assumption)"
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path) as f:
            cal = json.load(f)
        mid = low / (cal["low_ms"] / cal["mid_ms"])
        source = f"CoLExT ratio; mid tier from own phones (low/mid = {cal['low_ms'] / cal['mid_ms']:.2f})"
    rel = {"high": 1.0, "mid": mid, "low": low}
    return {t: HIGH_TIER_STEP_S * r for t, r in rel.items()}, source


def assign_tiers(n: int, mix: str, rng: np.random.Generator) -> list[str]:
    """Deterministic tier counts (largest remainder), randomly shuffled over clients."""
    shares = TIER_MIXES[mix]
    raw = {t: s * n for t, s in shares.items()}
    counts = {t: int(math.floor(v)) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - counts[t], reverse=True)[: n - sum(counts.values())]:
        counts[t] += 1
    tiers = [t for t, c in counts.items() for _ in range(c)]
    return [tiers[i] for i in rng.permutation(n)]


def mbps_to_bytes_per_s(mbps: float) -> float:
    return mbps * 1e6 / 8


def gateway_agg_time(n_updates: int, model_bytes: int, net: NetConfig) -> float:
    return n_updates * (net.agg_overhead_s + model_bytes / net.agg_bytes_per_s)


def describe(net: NetConfig) -> dict:
    steps, source = step_times()
    return {"net": asdict(net), "step_s": steps, "step_source": source, "wan_mbps": WAN_MBPS,
            "colext_compute_gap": COLEXT_COMPUTE_GAP, "colext_upload_gap": COLEXT_UPLOAD_GAP}
