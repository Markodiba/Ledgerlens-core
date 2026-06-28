"""Real-time adversarial red team loop.

A continuous, GAN-style hardening loop in which a genetic-algorithm *attacker*
(:mod:`detection.red_team.attacker`) repeatedly perturbs the feature vectors of
known wash trades to drive the current production model's score below an evasion
threshold.  Successful evasions are persisted
(:mod:`detection.red_team.evasion_logger`) and, once enough accumulate, trigger a
``MODEL_EVASION_DETECTED`` webhook and a retraining run that folds the hard
examples back into the training set as high-risk positives.

:func:`detection.red_team.runner.run_red_team_loop` drives this continuously and
is safe to run on a background thread.

Module-level tunables (overridable via environment variables):

``EVASION_THRESHOLD``
    Model score (0-100) below which an attack counts as a successful evasion.
``N_EVASION_TRIGGER``
    Number of accumulated evasions that triggers an automated hardening run.
"""

import os

EVASION_THRESHOLD: float = float(os.getenv("LEDGERLENS_EVASION_THRESHOLD", "30.0"))
N_EVASION_TRIGGER: int = int(os.getenv("LEDGERLENS_N_EVASION_TRIGGER", "100"))

__all__ = ["EVASION_THRESHOLD", "N_EVASION_TRIGGER"]
