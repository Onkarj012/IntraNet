"""Live-readiness risk governor for futures order tickets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import time as dtime
from pathlib import Path

import pandas as pd

from engine.orders import KILL_SWITCH, OrderTicket


@dataclass(frozen=True)
class RiskLimits:
    allowed_variants_live: set[str] = field(default_factory=lambda: {"A"})
    max_qty_lots_per_order: int = 1
    max_orders_per_day: int = 3
    max_open_positions: int = 1
    daily_loss_halt_inr: float = -15_000.0
    # Phase-1 hardening: tighter intraday cumulative halt (fires before daily_loss_halt)
    intraday_cum_halt_inr: float = -9_000.0   # halt after -₹9k in session
    # Phase-1 hardening: skip entries when VIX has spiked >15% vs prior close
    vix_spike_pct: float = 0.15               # 0.0 = disabled
    no_new_entries_after: dtime = dtime(14, 55)
    require_flat_before_entry: bool = True


@dataclass(frozen=True)
class BrokerState:
    open_positions: int = 0
    day_realized_pnl_inr: float = 0.0
    # Phase-1: intraday cumulative PnL (resets each session, tracks open+closed)
    intraday_cum_pnl_inr: float = 0.0
    # Phase-1: VIX intraday spike detection
    vix_prior_close: float = 0.0   # 0.0 = unknown / disabled
    vix_current: float = 0.0       # 0.0 = unknown / disabled


@dataclass(frozen=True)
class RiskDecision:
    ok: bool
    reasons: list[str]
    limits: dict

    def to_dict(self) -> dict:
        return asdict(self)


def order_count_for_date(order_ledger: Path, trade_date: str) -> int:
    if not order_ledger.exists():
        return 0
    n = 0
    with order_ledger.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("ticket", {}).get("trade_date") == trade_date:
                    n += 1
            except Exception:
                if trade_date in line:
                    n += 1
    return n


def evaluate_ticket_risk(
    ticket: OrderTicket,
    *,
    trade_date: str,
    order_ledger: Path,
    limits: RiskLimits | None = None,
    broker_state: BrokerState | None = None,
) -> RiskDecision:
    limits = limits or RiskLimits()
    broker_state = broker_state or BrokerState()
    reasons: list[str] = []

    if KILL_SWITCH.exists():
        reasons.append(f"kill-switch present: {KILL_SWITCH}")
    if ticket.variant not in limits.allowed_variants_live:
        reasons.append(f"variant {ticket.variant} not allowed for live execution")
    if ticket.qty_lots > limits.max_qty_lots_per_order:
        reasons.append(
            f"qty_lots {ticket.qty_lots} exceeds max {limits.max_qty_lots_per_order}"
        )
    if order_count_for_date(order_ledger, trade_date) >= limits.max_orders_per_day:
        reasons.append(f"daily order count would exceed {limits.max_orders_per_day}")
    if broker_state.day_realized_pnl_inr <= limits.daily_loss_halt_inr:
        reasons.append(
            f"day realized PnL {broker_state.day_realized_pnl_inr:+,.0f} "
            f"<= {limits.daily_loss_halt_inr:+,.0f}"
        )
    # Phase-1: intraday cumulative halt (tighter than daily_loss_halt)
    if (limits.intraday_cum_halt_inr < 0.0 and
            broker_state.intraday_cum_pnl_inr <= limits.intraday_cum_halt_inr):
        reasons.append(
            f"intraday cumulative PnL {broker_state.intraday_cum_pnl_inr:+,.0f} "
            f"<= {limits.intraday_cum_halt_inr:+,.0f}"
        )
    # Phase-1: VIX intraday spike filter
    if (limits.vix_spike_pct > 0.0
            and broker_state.vix_prior_close > 0.0
            and broker_state.vix_current > 0.0):
        spike = (broker_state.vix_current - broker_state.vix_prior_close) / broker_state.vix_prior_close
        if spike >= limits.vix_spike_pct:
            reasons.append(
                f"VIX intraday spike {spike*100:.1f}% "
                f"(current={broker_state.vix_current:.1f}, "
                f"prior_close={broker_state.vix_prior_close:.1f}) "
                f">= {limits.vix_spike_pct*100:.0f}% threshold"
            )
    if limits.require_flat_before_entry and broker_state.open_positions > 0:
        reasons.append("broker state is not flat before entry")
    if broker_state.open_positions >= limits.max_open_positions:
        reasons.append(f"open positions {broker_state.open_positions} >= max {limits.max_open_positions}")

    try:
        ts = pd.Timestamp(ticket.timestamp)
        if ts.time() >= limits.no_new_entries_after:
            reasons.append(f"ticket timestamp is at/after {limits.no_new_entries_after}")
    except Exception:
        reasons.append("ticket timestamp is invalid")

    return RiskDecision(ok=not reasons, reasons=reasons, limits={
        "allowed_variants_live": sorted(limits.allowed_variants_live),
        "max_qty_lots_per_order": limits.max_qty_lots_per_order,
        "max_orders_per_day": limits.max_orders_per_day,
        "max_open_positions": limits.max_open_positions,
        "daily_loss_halt_inr": limits.daily_loss_halt_inr,
        "intraday_cum_halt_inr": limits.intraday_cum_halt_inr,
        "vix_spike_pct": limits.vix_spike_pct,
        "no_new_entries_after": limits.no_new_entries_after.isoformat(),
        "require_flat_before_entry": limits.require_flat_before_entry,
    })
