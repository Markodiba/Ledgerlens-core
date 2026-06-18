"""Atomic path-payment circularity detection.

A single signed transaction can route `XLM -> A -> B -> XLM` through several
order books and/or pools in one atomic operation. That ingests as a sequence
of unrelated `Trade` rows with no link back to the parent transaction, so a
wallet that round-trips its own funds through a multi-hop path payment is
invisible to single-account, consecutive-trade detectors. This module flags
that pattern directly from `ingestion.data_models.PathPayment` records.
"""

from ingestion.data_models import PathPayment


def detect_atomic_circular_routes(path_payments: list[PathPayment]) -> list[dict]:
    """Flag path payments where:

    - `source_account == destination_account` (atomic self-payment loop), or
    - `destination_asset == source_asset` (round-trips back to the same asset
      even when `destination_account` differs — still manufactures volume
      with no net economic position change).

    A legitimate non-cyclic multi-hop payment to a different destination in a
    different asset is not flagged.
    """
    routes = []
    for payment in path_payments:
        is_self_payment = payment.source_account == payment.destination_account
        is_same_asset_cycle = payment.source_asset.pair_symbol == payment.destination_asset.pair_symbol
        if not is_self_payment and not is_same_asset_cycle:
            continue

        routes.append(
            {
                "transaction_hash": payment.transaction_hash,
                "accounts": sorted({payment.source_account, payment.destination_account}),
                "hop_count": len(payment.path) + 1,
                "cycle_volume": min(payment.source_amount, payment.destination_amount),
                "is_atomic_self_payment": is_self_payment,
                "touches_pool": False,
            }
        )
    return routes
