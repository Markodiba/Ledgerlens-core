"""Benford's Law digit-distribution analysis for transaction amounts.

Computes the chi-square statistic, per-digit Z-scores, and Mean Absolute
Deviation (MAD) of the leading-digit distribution of a set of amounts,
relative to the theoretical Benford distribution.

Also provides an incremental O(1) engine (`IncrementalBenfordEngine`) for
real-time streaming pipelines (issue #128).
"""

from __future__ import annotations

import collections
import json
import math
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DIGITS = list(range(1, 10))

# P(d) = log10(1 + 1/d) for d in 1..9
BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in DIGITS}


def first_digit(value: float) -> int | None:
    """Return the leading (most significant) decimal digit of `value`.

    Returns None for zero, negative, or non-finite values, which are
    excluded from Benford analysis.
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    while value < 1:
        value *= 10
    while value >= 10:
        value /= 10
    return int(value)


def digit_distribution(amounts: list[float]) -> dict[int, float]:
    """Return the observed proportion of each leading digit 1-9 in `amounts`."""
    digits = [d for d in (first_digit(a) for a in amounts) if d is not None]
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    counts = {d: 0 for d in DIGITS}
    for d in digits:
        counts[d] += 1
    return {d: counts[d] / n for d in DIGITS}


def chi_square_statistic(observed: dict[int, float], n: int) -> float:
    """Chi-square goodness-of-fit statistic vs. the Benford distribution.

    `observed` is a digit -> proportion mapping (e.g. from `digit_distribution`).
    `n` is the number of observations the proportions were computed from.
    """
    if n == 0:
        return 0.0
    chi_sq = 0.0
    for d in DIGITS:
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed.get(d, 0.0) * n
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count
    return chi_sq


def z_scores(observed: dict[int, float], n: int) -> dict[int, float]:
    """Per-digit Z-score of the observed proportion vs. Benford's expectation."""
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    scores = {}
    for d in DIGITS:
        p = BENFORD_EXPECTED[d]
        observed_p = observed.get(d, 0.0)
        # continuity correction as commonly used in Benford forensic analysis
        numerator = abs(observed_p - p) - (1 / (2 * n))
        denominator = math.sqrt(p * (1 - p) / n)
        scores[d] = max(numerator, 0.0) / denominator if denominator > 0 else 0.0
    return scores


def mean_absolute_deviation(observed: dict[int, float]) -> float:
    """MAD between observed and expected digit distributions.

    Values above ~0.015 (for first-digit tests) are commonly treated as
    indicating non-conformity with Benford's Law.
    """
    deviations = [abs(observed.get(d, 0.0) - BENFORD_EXPECTED[d]) for d in DIGITS]
    return float(np.mean(deviations))


def compute_benford_metrics(amounts: list[float]) -> dict:
    """Compute the full set of Benford metrics for a list of transaction amounts.

    Returns a dict with `chi_square`, `mad`, `z_scores` (per digit), the
    `observed_distribution`, and `sample_size`.
    """
    observed = digit_distribution(amounts)
    n = sum(1 for a in amounts if first_digit(a) is not None)

    return {
        "chi_square": chi_square_statistic(observed, n),
        "mad": mean_absolute_deviation(observed),
        "z_scores": z_scores(observed, n),
        "observed_distribution": observed,
        "sample_size": n,
    }


def is_anomalous(metrics: dict, mad_threshold: float = 0.015) -> bool:
    """Whether a `compute_benford_metrics` result exceeds the MAD threshold."""
    return metrics["mad"] > mad_threshold


# ---------------------------------------------------------------------------
# Incremental Benford Engine (O(1) per new transaction, issue #128)
# ---------------------------------------------------------------------------

BENFORD_EXPECTED_ARR = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


@dataclass
class DigitCounter:
    """9-element leading-digit count accumulator."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.int64))

    def increment(self, amount: float) -> None:
        d = first_digit(amount)
        if d is not None:
            self.counts[d - 1] += 1

    def decrement(self, amount: float) -> None:
        d = first_digit(amount)
        if d is not None:
            self.counts[d - 1] = max(0, self.counts[d - 1] - 1)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def chi_square(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        observed_p = self.counts / n
        expected = BENFORD_EXPECTED_ARR * n
        return float(np.sum((self.counts - expected) ** 2 / np.where(expected > 0, expected, 1)))

    def z_score(self) -> dict[int, float]:
        n = self.total
        if n == 0:
            return {d: 0.0 for d in range(1, 10)}
        observed_p = self.counts / n
        scores = {}
        for i, d in enumerate(range(1, 10)):
            p = BENFORD_EXPECTED_ARR[i]
            numerator = max(abs(observed_p[i] - p) - 1 / (2 * n), 0.0)
            denom = math.sqrt(p * (1 - p) / n)
            scores[d] = numerator / denom if denom > 0 else 0.0
        return scores

    def mad(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        observed_p = self.counts / n
        return float(np.mean(np.abs(observed_p - BENFORD_EXPECTED_ARR)))

    def to_dict(self) -> dict:
        return {"counts": self.counts.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DigitCounter":
        counts = np.array(d["counts"], dtype=np.int64)
        return cls(counts=counts)


# In-memory expiry deque entry: (timestamp_unix_float, amount)
_Entry = tuple[float, float]

_FOUR_HOURS_S: float = 4 * 3600.0  # seconds; windows longer than this use SQLite


class IncrementalWindow:
    """Wraps a DigitCounter with a timestamped expiry queue.

    For windows <= 4h, a deque is used. For longer windows, a SQLite-backed
    queue bounds memory for wallets with millions of events.
    """

    def __init__(
        self,
        window_seconds: float,
        wallet: str,
        window_name: str,
        db_path: str | None = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.wallet = wallet
        self.window_name = window_name
        self.counter = DigitCounter()
        self._use_sqlite = window_seconds > _FOUR_HOURS_S
        self._db_path = db_path or ":memory:"
        if self._use_sqlite:
            self._init_sqlite()
        else:
            self._deque: collections.deque[_Entry] = collections.deque()

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS window_events "
            "(wallet TEXT, window_name TEXT, ts REAL, amount REAL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_we ON window_events(wallet, window_name, ts)"
        )
        self._conn.commit()

    def add_transaction(self, amount: float, ts: datetime) -> None:
        ts_f = ts.timestamp()
        self.counter.increment(amount)
        if self._use_sqlite:
            self._conn.execute(
                "INSERT INTO window_events VALUES (?,?,?,?)",
                (self.wallet, self.window_name, ts_f, amount),
            )
            self._conn.commit()
        else:
            self._deque.append((ts_f, amount))

    def expire_transactions(self, as_of: datetime) -> None:
        cutoff = as_of.timestamp() - self.window_seconds
        if self._use_sqlite:
            expired = self._conn.execute(
                "SELECT amount FROM window_events WHERE wallet=? AND window_name=? AND ts<?",
                (self.wallet, self.window_name, cutoff),
            ).fetchall()
            for (amt,) in expired:
                self.counter.decrement(amt)
            self._conn.execute(
                "DELETE FROM window_events WHERE wallet=? AND window_name=? AND ts<?",
                (self.wallet, self.window_name, cutoff),
            )
            self._conn.commit()
        else:
            while self._deque and self._deque[0][0] < cutoff:
                _, amt = self._deque.popleft()
                self.counter.decrement(amt)

    def get_features(self) -> dict:
        return {
            "chi_square": self.counter.chi_square(),
            "mad": self.counter.mad(),
            "max_zscore": max(self.counter.z_score().values(), default=0.0),
            "sample_size": self.counter.total,
        }


_WINDOW_SIZES_S: dict[str, float] = {
    "1h": 3600.0,
    "4h": 4 * 3600.0,
    "24h": 24 * 3600.0,
    "7d": 7 * 86400.0,
    "30d": 30 * 86400.0,
}


class IncrementalBenfordEngine:
    """Per-wallet incremental Benford feature engine.

    Maintains five rolling windows (1h, 4h, 24h, 7d, 30d) per wallet.
    Each new transaction is processed in O(1); feature retrieval is O(9).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._wallet_windows: dict[str, dict[str, IncrementalWindow]] = {}
        self._lock = threading.Lock()

    def _get_windows(self, wallet: str) -> dict[str, IncrementalWindow]:
        windows = self._wallet_windows.get(wallet)
        if windows is None:
            windows = {
                name: IncrementalWindow(size_s, wallet, name, self._db_path)
                for name, size_s in _WINDOW_SIZES_S.items()
            }
            self._wallet_windows[wallet] = windows
        return windows

    def update(self, wallet: str, amount: float, timestamp: datetime) -> None:
        """Process a new transaction for `wallet`."""
        with self._lock:
            windows = self._get_windows(wallet)
            for window in windows.values():
                window.expire_transactions(timestamp)
                window.add_transaction(amount, timestamp)

    def get_features(self, wallet: str) -> dict:
        """Return Benford feature dict for `wallet` (same keys as BENFORD_FEATURE_NAMES)."""
        with self._lock:
            windows = self._get_windows(wallet)
            features = {}
            for name, window in windows.items():
                wf = window.get_features()
                features[f"benford_chi_square_{name}"] = wf["chi_square"]
                features[f"benford_mad_{name}"] = wf["mad"]
                features[f"benford_max_zscore_{name}"] = wf["max_zscore"]
            return features
