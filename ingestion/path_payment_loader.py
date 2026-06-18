"""Path payment ingestion from Horizon.

`path_payment_strict_send` and `path_payment_strict_receive` are distinct
Horizon operation types that both route a source asset to a destination
asset across an intermediate path — the difference is which end is pinned
(send amount vs. receive amount). A wash trader can manufacture self-funded
circular volume with either, so both must be ingested.

Mirrors `ingestion/operations_loader.py`'s structure: all Horizon calls go
through `AsyncHorizonClient` / `get_with_retry`, both sync and async entry
points are provided.
"""

import logging
from datetime import datetime

import httpx

from ingestion.data_models import Asset, PathPayment
from ingestion.http_client import AsyncHorizonClient, get_with_retry
from ingestion.operations_loader import _horizon_url, _parse_datetime, _parse_float

logger = logging.getLogger("ledgerlens.path_payment_loader")

PAGE_LIMIT = 200
MAX_PATH_HOPS = 8
PATH_PAYMENT_OPERATION_TYPES = ("path_payment_strict_send", "path_payment_strict_receive")


def _is_path_payment_operation(record: dict) -> bool:
    return record.get("type") in PATH_PAYMENT_OPERATION_TYPES


def _asset_with_prefix(record: dict, prefix: str) -> Asset:
    key_prefix = f"{prefix}_" if prefix else ""
    asset_type = record.get(f"{key_prefix}asset_type")
    code = record.get(f"{key_prefix}asset_code")
    issuer = record.get(f"{key_prefix}asset_issuer")
    if asset_type == "native" or not code:
        return Asset(code="XLM", issuer=None)
    return Asset(code=code, issuer=issuer)


def _parse_path(record: dict) -> list[Asset]:
    """Parse the intermediate hop assets, defensively bounding the array length.

    Horizon caps a real path at 5 hops, but upstream array length is never
    trusted blindly — anything longer than `MAX_PATH_HOPS` is logged and
    truncated rather than raised.
    """
    raw_path = record.get("path") or []
    if len(raw_path) > MAX_PATH_HOPS:
        logger.warning(
            "Path payment %s has %d hops, exceeding the defensive bound of %d; truncating",
            record.get("id"),
            len(raw_path),
            MAX_PATH_HOPS,
        )
        raw_path = raw_path[:MAX_PATH_HOPS]

    hops = []
    for hop in raw_path:
        asset_type = hop.get("asset_type")
        code = hop.get("asset_code")
        issuer = hop.get("asset_issuer")
        if asset_type == "native" or not code:
            hops.append(Asset(code="XLM", issuer=None))
        else:
            hops.append(Asset(code=code, issuer=issuer))
    return hops


def _parse_path_payment(record: dict) -> PathPayment:
    return PathPayment(
        id=str(record.get("id") or ""),
        transaction_hash=str(record.get("transaction_hash") or ""),
        timestamp=_parse_datetime(record.get("created_at")),
        source_account=str(record.get("from") or record.get("source_account") or ""),
        destination_account=str(record.get("to") or ""),
        source_asset=_asset_with_prefix(record, "source"),
        destination_asset=_asset_with_prefix(record, ""),
        source_amount=_parse_float(record.get("source_amount")),
        destination_amount=_parse_float(record.get("amount")),
        path=_parse_path(record),
        strict_send=record.get("type") == "path_payment_strict_send",
    )


def load_path_payments(account: str, since: datetime, limit: int = PAGE_LIMIT) -> list[PathPayment]:
    """GET /accounts/{account}/operations filtered to path-payment operations since `since`."""
    cutoff = _parse_datetime(since)
    url = _horizon_url(f"/accounts/{account}/operations")
    with httpx.Client(timeout=30.0) as client:
        response = get_with_retry(client, url, params={"limit": limit, "order": "desc"})
        records = response.json().get("_embedded", {}).get("records", [])
    payments = [_parse_path_payment(r) for r in records if _is_path_payment_operation(r)]
    return [p for p in payments if p.timestamp >= cutoff]


def load_path_payments_for_accounts(
    accounts: list[str],
    since: datetime,
    limit: int = PAGE_LIMIT,
) -> list[PathPayment]:
    """Fetch path payments for each account in `accounts` and concatenate the results."""
    payments: list[PathPayment] = []
    for account in accounts:
        payments.extend(load_path_payments(account, since, limit))
    return payments


async def async_load_path_payments(
    account: str,
    since: datetime,
    client: AsyncHorizonClient,
    limit: int = PAGE_LIMIT,
) -> list[PathPayment]:
    """Async variant of `load_path_payments` using `AsyncHorizonClient`."""
    cutoff = _parse_datetime(since)
    data = await client.get(f"/accounts/{account}/operations", params={"limit": limit, "order": "desc"})
    records = data.get("_embedded", {}).get("records", [])
    payments = [_parse_path_payment(r) for r in records if _is_path_payment_operation(r)]
    return [p for p in payments if p.timestamp >= cutoff]
