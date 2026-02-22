"""
IT-PUF: Inference-Time Physical Unclonable Function
====================================================
Model identity verification via challenge-response fingerprinting.

Patent Application 63/982,893 (February 13, 2026)

Public API (no GPU required):
    anchor = itpuf.Anchor.load("model_anchor.json")
    result = anchor.verify_against(other_anchor)
    report = itpuf.compute_far([anchor_a, anchor_b])

Enrollment and live verification require the measurement engine,
available through a commercial license or the fallrisk.ai API.

    Contact: anthony@fallrisk.ai
"""

__version__ = "0.1.0"

from itpuf.contract import MeasurementContract, CONTRACT
from itpuf.anchor import Anchor, VerifyResult, compute_far

__all__ = [
    "Anchor",
    "VerifyResult",
    "compute_far",
    "MeasurementContract",
    "CONTRACT",
]
