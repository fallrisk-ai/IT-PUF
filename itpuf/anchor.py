"""
IT-PUF Anchor — Identity Storage, Verification Math, and Serialization.

Public operations (no measurement engine required):
    anchor = Anchor.load("model_anchor.json")
    anchor.save("copy.json")
    distances = anchor.distance(other_anchor.responses)
    far_report = compute_far([anchor_a, anchor_b, anchor_c])

Enrollment and live verification require the measurement engine,
available through a commercial license or the fallrisk.ai API.

Patent Application 63/982,893 (February 13, 2026)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from itpuf.contract import CONTRACT, MeasurementContract


# ═══════════════════════════════════════════════════════════════════
# ANCHOR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Anchor:
    """
    Enrolled model identity.

    Contains response vectors for all seeds, architecture metadata,
    and the contract under which enrollment was performed.
    """
    model_name: str
    responses: Dict[int, np.ndarray]       # seed → τ vector
    arch_type: str
    arch_strategy: str
    n_layers: int
    contract_hash: str
    epsilon: float
    enrolled_at: str
    prompt_bank_hash: str
    validation: Dict[int, dict] = field(default_factory=dict)

    def distance(self, other_responses: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Compute L2 distance per seed against another set of responses."""
        distances = {}
        for seed in self.responses:
            if seed not in other_responses:
                distances[seed] = float("inf")
            else:
                distances[seed] = float(np.linalg.norm(
                    self.responses[seed] - other_responses[seed]
                ))
        return distances

    def verify_against(self, other: Anchor) -> VerifyResult:
        """
        Verify this anchor against another anchor (offline comparison).

        Useful for auditing: does anchor_a match anchor_b?
        For live model verification, use the measurement engine.
        """
        distances = self.distance(other.responses)
        epsilon = max(self.epsilon, other.epsilon)
        ratios = {s: d / epsilon for s, d in distances.items()}
        max_d = max(distances.values()) if distances else float("inf")
        min_d = min(distances.values()) if distances else float("inf")
        accepted = max_d <= epsilon

        return VerifyResult(
            accepted=accepted,
            distances=distances,
            ratios_to_epsilon=ratios,
            max_distance=max_d,
            min_distance=min_d,
            epsilon=epsilon,
            decision="ACCEPT" if accepted else "REJECT",
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "responses": {
                str(s): v.tolist() for s, v in self.responses.items()
            },
            "arch_type": self.arch_type,
            "arch_strategy": self.arch_strategy,
            "n_layers": self.n_layers,
            "contract_hash": self.contract_hash,
            "epsilon": self.epsilon,
            "enrolled_at": self.enrolled_at,
            "prompt_bank_hash": self.prompt_bank_hash,
            "validation": self.validation,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Serialize anchor to JSON."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Anchor:
        """Deserialize anchor from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        responses = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in data["responses"].items()
        }
        return cls(
            model_name=data["model_name"],
            responses=responses,
            arch_type=data["arch_type"],
            arch_strategy=data["arch_strategy"],
            n_layers=data["n_layers"],
            contract_hash=data["contract_hash"],
            epsilon=data["epsilon"],
            enrolled_at=data["enrolled_at"],
            prompt_bank_hash=data["prompt_bank_hash"],
            validation=data.get("validation", {}),
        )


# ═══════════════════════════════════════════════════════════════════
# VERIFY RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VerifyResult:
    """Result of verification against an anchor."""
    accepted: bool
    distances: Dict[int, float]           # per-seed L2 distance
    ratios_to_epsilon: Dict[int, float]   # per-seed distance / ε
    max_distance: float
    min_distance: float
    epsilon: float
    decision: str                          # "ACCEPT" or "REJECT"

    def __str__(self) -> str:
        seed_details = ", ".join(
            f"s{s}={d:.6f} ({r:.1f}×ε)"
            for (s, d), (_, r) in zip(
                sorted(self.distances.items()),
                sorted(self.ratios_to_epsilon.items()),
            )
        )
        return f"{self.decision} | max={self.max_distance:.6f} | {seed_details}"


# ═══════════════════════════════════════════════════════════════════
# CROSS-MODEL FAR (zoo evaluation)
# ═══════════════════════════════════════════════════════════════════

def compute_far(
    anchors: List[Anchor],
    epsilon: Optional[float] = None,
) -> dict:
    """
    Compute False Accept Rate across a zoo of enrolled models.

    For every pair of distinct models and every seed, computes
    L2 distance and checks against ε.

    Args:
        anchors: List of enrolled Anchor objects.
        epsilon: Override ε (default: use each anchor's enrolled ε).

    Returns:
        Dict with FAR, comparisons, min/max distances, and details.
    """
    n = len(anchors)
    details = []
    n_false_accept = 0
    n_total = 0

    for i in range(n):
        for j in range(i + 1, n):
            eps = epsilon or max(anchors[i].epsilon, anchors[j].epsilon)
            seeds_i = set(anchors[i].responses.keys())
            seeds_j = set(anchors[j].responses.keys())

            for seed in seeds_i & seeds_j:
                d = float(np.linalg.norm(
                    anchors[i].responses[seed] - anchors[j].responses[seed]
                ))
                is_accept = d <= eps
                if is_accept:
                    n_false_accept += 1
                n_total += 1

                details.append({
                    "model_a": anchors[i].model_name,
                    "model_b": anchors[j].model_name,
                    "arch_a": anchors[i].arch_type,
                    "arch_b": anchors[j].arch_type,
                    "cross_architecture": anchors[i].arch_type != anchors[j].arch_type,
                    "seed": seed,
                    "distance": d,
                    "ratio_to_epsilon": d / eps,
                    "accepted": is_accept,
                })

    far = n_false_accept / n_total if n_total > 0 else 0.0
    all_distances = [r["distance"] for r in details]

    return {
        "FAR": far,
        "n_models": n,
        "n_pairs": n_total,
        "n_false_accepts": n_false_accept,
        "min_distance": min(all_distances) if all_distances else float("inf"),
        "max_distance": max(all_distances) if all_distances else 0.0,
        "min_ratio": min(r["ratio_to_epsilon"] for r in details) if details else float("inf"),
        "details": details,
    }
