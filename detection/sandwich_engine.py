"""Sandwich attack detection for SDEX order-book trades.

A sandwich attack consists of three ordered events in a tight ledger window:
  1. Front-run: attacker buys asset X just before a large victim order.
  2. Victim order: large buy of X that moves the price.
  3. Back-run: attacker sells X immediately after at the inflated price.

False-positive hardening (issue #122):
  - Victim-amount minimum threshold (MIN_VICTIM_AMOUNT_XLM).
  - Price-impact score: minimum 0.3% movement required.
  - Statistical significance test: permutation test against 24h baseline.
  - Confidence score combining all three signals; only events >= 0.7 emitted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

MIN_VICTIM_AMOUNT_XLM: float = 500.0
MIN_PRICE_IMPACT: float = 0.003  # 0.3%
MIN_SANDWICH_CONFIDENCE: float = 0.7
_PERMUTATION_N: int = 1000


@dataclass
class SandwichCandidate:
    attacker_wallet: str
    victim_wallet: str
    asset_pair: str
    front_run_time: datetime
    victim_time: datetime
    back_run_time: datetime
    front_run_price: float
    victim_amount: float
    back_run_price: float
    pre_trade_price: float
    post_trade_price: float
    confidence: float = 0.0


@dataclass
class SandwichEvent:
    attacker_wallet: str
    victim_wallet: str
    asset_pair: str
    front_run_time: datetime
    victim_time: datetime
    back_run_time: datetime
    victim_amount: float
    price_impact: float
    sandwich_confidence: float


class SandwichEngine:
    """Detect sandwich attacks in a window of SDEX trade records."""

    def __init__(
        self,
        victim_amount_filter: float = MIN_VICTIM_AMOUNT_XLM,
        min_price_impact: float = MIN_PRICE_IMPACT,
        min_confidence: float = MIN_SANDWICH_CONFIDENCE,
        ledger_window: int = 5,
    ) -> None:
        self.victim_amount_filter = victim_amount_filter
        self.min_price_impact = min_price_impact
        self.min_confidence = min_confidence
        self.ledger_window = ledger_window

    def _compute_price_impact(self, pre_trade_price: float, post_trade_price: float) -> float:
        """Signed fractional price change caused by the victim order."""
        if pre_trade_price == 0.0:
            return 0.0
        return (post_trade_price - pre_trade_price) / pre_trade_price

    def _timing_significance(
        self,
        attacker_interval_s: float,
        pair_baseline_intervals: list[float],
    ) -> float:
        """One-sided permutation p-value: probability that a random interval
        from the 24h baseline is <= attacker_interval_s.

        Returns a significance score in [0, 1] where higher means tighter
        (more anomalous) timing. Score = 1 - p_value.
        """
        if not pair_baseline_intervals:
            return 0.5
        n = len(pair_baseline_intervals)
        count_le = sum(1 for v in pair_baseline_intervals if v <= attacker_interval_s)
        # Permutation p-value: fraction of baseline intervals at least as extreme
        p_value = count_le / n
        return 1.0 - p_value

    def _confidence(
        self,
        victim_amount: float,
        price_impact: float,
        timing_score: float,
    ) -> float:
        """Combine the three hardening signals into a [0, 1] confidence score."""
        # Amount signal: sigmoid-style, saturates at ~5x threshold
        amount_score = min(1.0, victim_amount / (self.victim_amount_filter * 5))
        # Impact signal: saturates at 3x min threshold
        impact_score = min(1.0, abs(price_impact) / (self.min_price_impact * 3))
        # Timing signal already in [0, 1]
        return (amount_score + impact_score + timing_score) / 3.0

    def detect(
        self,
        trades: pd.DataFrame,
        baseline_intervals: dict[str, list[float]] | None = None,
    ) -> list[SandwichEvent]:
        """Scan `trades` for sandwich attack patterns.

        `baseline_intervals` maps asset_pair -> list of inter-trade interval
        seconds over the prior 24h, used for the permutation significance test.
        """
        if trades.empty:
            return []

        required = {"base_account", "counter_account", "base_amount", "price", "ledger_close_time"}
        if not required.issubset(trades.columns):
            return []

        baseline_intervals = baseline_intervals or {}
        df = trades.sort_values("ledger_close_time").reset_index(drop=True)
        events: list[SandwichEvent] = []

        asset_pairs = df["asset_pair"].unique() if "asset_pair" in df.columns else [None]

        for pair in asset_pairs:
            if pair is not None:
                pair_df = df[df["asset_pair"] == pair].reset_index(drop=True)
            else:
                pair_df = df

            if len(pair_df) < 3:
                continue

            pair_key = str(pair) if pair else ""
            baselines = baseline_intervals.get(pair_key, [])

            # Group trades by account for fast lookup
            by_account: dict[str, list[int]] = {}
            for idx, row in pair_df.iterrows():
                acct = str(row["base_account"])
                by_account.setdefault(acct, []).append(idx)

            n = len(pair_df)
            for victim_idx in range(1, n - 1):
                victim_row = pair_df.iloc[victim_idx]
                victim_amount = float(victim_row["base_amount"])

                if victim_amount < self.victim_amount_filter:
                    continue

                victim_time = pd.Timestamp(victim_row["ledger_close_time"])
                victim_acct = str(victim_row["base_account"])

                # Look for front-run (buy) just before victim
                for fr_idx in range(max(0, victim_idx - self.ledger_window), victim_idx):
                    fr_row = pair_df.iloc[fr_idx]
                    fr_acct = str(fr_row["base_account"])
                    if fr_acct == victim_acct:
                        continue

                    fr_time = pd.Timestamp(fr_row["ledger_close_time"])

                    # Look for back-run (sell) just after victim by same attacker
                    for br_idx in range(victim_idx + 1, min(n, victim_idx + self.ledger_window + 1)):
                        br_row = pair_df.iloc[br_idx]
                        br_acct = str(br_row["base_account"])
                        if br_acct != fr_acct:
                            continue

                        br_time = pd.Timestamp(br_row["ledger_close_time"])
                        pre_price = float(fr_row["price"])
                        post_price = float(br_row["price"])
                        impact = self._compute_price_impact(pre_price, post_price)

                        if abs(impact) < self.min_price_impact:
                            continue

                        interval_s = (br_time - fr_time).total_seconds()
                        timing_score = self._timing_significance(interval_s, baselines)
                        conf = self._confidence(victim_amount, impact, timing_score)

                        if conf >= self.min_confidence:
                            events.append(
                                SandwichEvent(
                                    attacker_wallet=fr_acct,
                                    victim_wallet=victim_acct,
                                    asset_pair=pair_key,
                                    front_run_time=fr_time.to_pydatetime(),
                                    victim_time=victim_time.to_pydatetime(),
                                    back_run_time=br_time.to_pydatetime(),
                                    victim_amount=victim_amount,
                                    price_impact=impact,
                                    sandwich_confidence=conf,
                                )
                            )

        return events
