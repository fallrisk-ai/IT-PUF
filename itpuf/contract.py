"""
Measurement Contract — Frozen specification for IT-PUF.

INVARIANT: Nothing in this file changes after v0.1.0 without a
major version bump. Changing any parameter invalidates all existing
anchors.

RNG DECISION: We use np.random.default_rng (NumPy Generator API),
which is the canonical RNG in prompt_bank.py and coffin_experiment_v2.py.

NOTE: Sprint 2 scripts used np.random.RandomState (legacy API), which
produces DIFFERENT sequences for the same seed. All Sprint 2 data is
internally consistent but incompatible with this contract. Re-enrollment
is required when migrating from Sprint 2 JSONs to this library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (frozen)
# ═══════════════════════════════════════════════════════════════════

SEEDS: Tuple[int, ...] = (42, 123, 456, 789)
K: int = 32                   # prompts per seed
D: int = 64                   # response vector dimensionality (2 × K)
MAX_SEQ_LEN: int = 512        # tokenizer truncation limit
EPSILON_DEFAULT: float = 1.003e-4  # canonical ε from BF16↔FP16 genuine repeats
EPSILON_FLOOR: float = 1e-4   # minimum ε (principled floor for deterministic backends)
EPS_NUMERICAL: float = 1e-12  # numerical floor for division


@dataclass(frozen=True)
class MeasurementContract:
    """
    Immutable measurement specification.

    Every enrollment and verification must reference a contract.
    Changing any field invalidates all anchors created under the
    previous contract.
    """
    seeds: Tuple[int, ...] = SEEDS
    k: int = K
    d: int = D
    max_seq_len: int = MAX_SEQ_LEN
    epsilon: float = EPSILON_DEFAULT
    version: str = "0.1.0"

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "k": self.k,
            "d": self.d,
            "max_seq_len": self.max_seq_len,
            "epsilon": self.epsilon,
            "version": self.version,
        }

    def content_hash(self) -> str:
        """Deterministic hash of contract parameters."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


# The singleton contract — import this, don't construct your own.
CONTRACT = MeasurementContract()


# ═══════════════════════════════════════════════════════════════════
# PROMPT SELECTION (canonical RNG)
# ═══════════════════════════════════════════════════════════════════

def select_prompts(
    prompt_bank: List[str],
    seed: int,
    k: int = K,
) -> List[str]:
    """
    Deterministically select k prompts from the bank.

    Uses np.random.default_rng (NumPy Generator API) — the canonical
    RNG matching prompt_bank.py and coffin_experiment_v2.py.

    Args:
        prompt_bank: The full prompt bank (512 prompts canonical).
        seed: Random seed for selection.
        k: Number of prompts to select.

    Returns:
        List of k prompts in sorted-index order.

    Raises:
        ValueError: If k exceeds bank size.
    """
    if k > len(prompt_bank):
        raise ValueError(f"k={k} exceeds bank size {len(prompt_bank)}")

    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(prompt_bank), size=k, replace=False))
    return [prompt_bank[i] for i in indices]


def get_prompt_indices(
    bank_size: int,
    seed: int,
    k: int = K,
) -> List[int]:
    """Return sorted indices of selected prompts (for debugging)."""
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(bank_size, size=k, replace=False))
    return indices


def get_challenge_indices(
    bank_size: int,
    seeds: Tuple[int, ...] = SEEDS,
    k: int = K,
) -> set:
    """Return union of all prompt indices across all seeds."""
    all_idx: set = set()
    for s in seeds:
        rng = np.random.default_rng(s)
        all_idx.update(rng.choice(bank_size, size=k, replace=False))
    return all_idx
