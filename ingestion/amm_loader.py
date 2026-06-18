"""AMM liquidity pool ingestion from Horizon (CAP-38).

Mirrors `ingestion/operations_loader.py`'s structure: all Horizon calls go
through `AsyncHorizonClient` / `get_with_retry` rather than raw `httpx`, and
both sync and async entry points are provided.
"""

from datetime import datetime

import httpx

from ingestion.data_models import Asset, LiquidityPool, Trade, TradeType
from ingestion.http_client import AsyncHorizonClient, get_with_retry
from ingestion.operations_loader import _horizon_url, _parse_datetime, _parse_float

PAGE_LIMIT = 200


def _reserve_asset(code: str) -> Asset:
    if code in (None, "", "native"):
        return Asset(code="XLM", issuer=None)
    if ":" in code:
        asset_code, issuer = code.split(":", 1)
        return Asset(code=asset_code, issuer=issuer)
    return Asset(code=code, issuer=None)


def _parse_pool(record: dict) -> LiquidityPool:
    reserves = [
        (_reserve_asset(r.get("asset")), _parse_float(r.get("amount")))
        for r in record.get("reserves", [])
    ]
    return LiquidityPool(
        id=str(record.get("id") or ""),
        fee_bp=int(_parse_float(record.get("fee_bp"))),
        total_shares=_parse_float(record.get("total_shares")),
        reserves=reserves,
    )


def _price(record: dict) -> float:
    price = record.get("price")
    if isinstance(price, dict):
        try:
            return float(price["n"]) / float(price["d"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return 0.0
    return _parse_float(price)


def _parse_pool_trade(record: dict, pool_id: str) -> Trade:
    base_asset = Asset(code=record.get("base_asset_code", "XLM"), issuer=record.get("base_asset_issuer"))
    counter_asset = Asset(code=record.get("counter_asset_code", "XLM"), issuer=record.get("counter_asset_issuer"))
    return Trade(
        id=str(record.get("id") or ""),
        ledger_close_time=_parse_datetime(record.get("ledger_close_time")),
        base_account=str(record.get("base_account") or ""),
        counter_account=None,
        base_asset=base_asset,
        counter_asset=counter_asset,
        base_amount=_parse_float(record.get("base_amount")),
        counter_amount=_parse_float(record.get("counter_amount")),
        price=_price(record),
        base_is_seller=bool(record.get("base_is_seller", False)),
        trade_type=TradeType.LIQUIDITY_POOL,
        liquidity_pool_id=pool_id,
    )


def load_liquidity_pools(limit: int = PAGE_LIMIT) -> list[LiquidityPool]:
    """GET /liquidity_pools — current pool reserves and share counts."""
    url = _horizon_url("/liquidity_pools")
    with httpx.Client(timeout=30.0) as client:
        response = get_with_retry(client, url, params={"limit": limit})
        records = response.json().get("_embedded", {}).get("records", [])
    return [_parse_pool(r) for r in records]


def load_liquidity_pool_trades(pool_id: str, since: datetime, limit: int = PAGE_LIMIT) -> list[Trade]:
    """GET /liquidity_pools/{pool_id}/trades, mapped to `Trade` records.

    Each trade has `trade_type=LIQUIDITY_POOL`, `liquidity_pool_id` set, and
    `counter_account=None` — the pool is the counterparty, not a wallet.
    """
    cutoff = _parse_datetime(since)
    url = _horizon_url(f"/liquidity_pools/{pool_id}/trades")
    with httpx.Client(timeout=30.0) as client:
        response = get_with_retry(client, url, params={"limit": limit, "order": "desc"})
        records = response.json().get("_embedded", {}).get("records", [])
    trades = [_parse_pool_trade(r, pool_id) for r in records]
    return [t for t in trades if t.ledger_close_time >= cutoff]


async def async_load_liquidity_pool_trades(
    pool_id: str,
    since: datetime,
    client: AsyncHorizonClient,
    limit: int = PAGE_LIMIT,
) -> list[Trade]:
    """Async variant of `load_liquidity_pool_trades` using `AsyncHorizonClient`."""
    cutoff = _parse_datetime(since)
    data = await client.get(f"/liquidity_pools/{pool_id}/trades", params={"limit": limit, "order": "desc"})
    records = data.get("_embedded", {}).get("records", [])
    trades = [_parse_pool_trade(r, pool_id) for r in records]
    return [t for t in trades if t.ledger_close_time >= cutoff]
