"""Order placement abstraction for the OptiNet futures engine.

Mirrors the broker.py pattern: an abstract OrderClient with a DryRun
implementation (used by default) and an Upstox stub (raises until the
SDK is wired).

Triple-key safety gate for any real-money execution:
  1. CLI flag             --live
  2. Env var              OPTINET_LIVE=1
  3. Confirmation token   --confirm-token /path/to/LIVE_TOKEN
                          (file content must match what's stored at
                          results/router_v0/LIVE_TOKEN)
Plus: the kill-switch file must NOT exist.

If ANY gate fails, real execution is disabled and the script falls back
to dry-run.  This is checked in `LiveExecutionGate.is_clear()`.
"""
from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional


REPO_ROOT     = Path(__file__).resolve().parents[3]
KILL_SWITCH   = REPO_ROOT / "results/router_v0/PAPER_TRADING_HALTED"
LIVE_TOKEN_F  = REPO_ROOT / "results/router_v0/LIVE_TOKEN"
ORDER_LEDGER  = REPO_ROOT / "results/router_v0/order_tickets.jsonl"


@dataclass
class OrderTicket:
    """A bracket-order ticket. One of these per paper_trade_id."""
    ticket_id: str
    paper_trade_id: str          # links to paper_trading_ledger row
    timestamp: str               # ISO8601, when ticket was generated
    symbol: str                  # 'NIFTY'
    expiry: Optional[str]        # weekly/monthly expiry as ISO date string
    side: str                    # 'BUY' / 'SELL'
    qty_lots: int                # lots, not contracts
    order_type: str              # 'MARKET' or 'LIMIT'
    limit_price: Optional[float]
    target_price: float
    stop_price: float
    horizon_minutes: int
    size_mult: float
    variant: str                 # 'A' or 'C'
    intended_for_live: bool      # True if all gates were clear at gen time
    notes: str = ""
    trade_date: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class OrderClient(ABC):
    """All order placement MUST go through this interface."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def place_order(self, ticket: OrderTicket) -> dict:
        """Return broker response. dict must include 'order_id' or
        'simulated_order_id'."""

    @abstractmethod
    def get_status(self, order_id: str) -> dict:
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        ...


class DryRunOrderClient(OrderClient):
    """Logs tickets to JSONL but never places any real orders.

    This is the default and what the live_execute script uses unless the
    triple-key gate passes.  Safe to run in any environment.
    """

    def __init__(self, ledger_path: Path = ORDER_LEDGER):
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def name(self) -> str:
        return "DryRunOrderClient"

    def emitted_ticket_ids(self) -> set[str]:
        if not self.ledger_path.exists():
            return set()
        ids: set[str] = set()
        with self.ledger_path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ticket = record.get("ticket", {})
                ticket_id = ticket.get("ticket_id")
                if ticket_id:
                    ids.add(str(ticket_id))
        return ids

    def place_order(self, ticket: OrderTicket) -> dict:
        if ticket.ticket_id in self.emitted_ticket_ids():
            return {
                "ticket_id": ticket.ticket_id,
                "status": "DRYRUN_DUPLICATE_SKIPPED",
            }
        sim_id = f"DRY-{uuid.uuid4().hex[:10]}"
        record = {
            "simulated_order_id": sim_id,
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "client": self.name(),
            "ticket": ticket.to_dict(),
        }
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return {"simulated_order_id": sim_id, "status": "DRYRUN_LOGGED"}

    def get_status(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "DRYRUN", "filled_qty": 0}

    def cancel(self, order_id: str) -> bool:
        return True


class UpstoxOrderClient(OrderClient):
    """Live Upstox order client (stub).

    Setup checklist before this can fire (operator):
    1. `pip install upstox-python-sdk`
    2. Set env vars:
         export UPSTOX_API_KEY=...
         export UPSTOX_API_SECRET=...
         export UPSTOX_ACCESS_TOKEN=...   (refresh daily via OAuth)
    3. Replace each NotImplementedError with the SDK call
    4. Test on tiny size only first (1 lot) and reconcile against
       paper_trading_ledger before scaling

    Endpoints needed:
    - /order/place           place_order
    - /order/details/{id}    get_status
    - /order/cancel/{id}     cancel

    UNTIL the methods below are wired, calling them raises and an
    accidental --live invocation cannot fire a real order.
    """

    def __init__(self):
        self.api_key      = os.environ.get("UPSTOX_API_KEY")
        self.access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")

    def name(self) -> str:
        return "UpstoxOrderClient"

    def place_order(self, ticket: OrderTicket) -> dict:
        raise NotImplementedError(
            "UpstoxOrderClient.place_order: wire up "
            "upstox.OrderApi.place_order() with the bracket-order params "
            "(transaction_type, quantity, order_type='MARKET' or 'LIMIT', "
            "product='I' for intraday, validity='DAY')."
        )

    def get_status(self, order_id: str) -> dict:
        raise NotImplementedError(
            "UpstoxOrderClient.get_status: wire up "
            "upstox.OrderApi.get_order_details(order_id)."
        )

    def cancel(self, order_id: str) -> bool:
        raise NotImplementedError(
            "UpstoxOrderClient.cancel: wire up "
            "upstox.OrderApi.cancel_order(order_id)."
        )


# ── Triple-key safety gate ───────────────────────────────────────────────────

@dataclass
class LiveExecutionGate:
    cli_live_flag: bool
    env_live_set: bool
    token_match: bool
    kill_switch_clear: bool
    detail: list[str] = field(default_factory=list)

    def is_clear(self) -> bool:
        return (self.cli_live_flag and self.env_live_set
                and self.token_match and self.kill_switch_clear)

    @classmethod
    def evaluate(cls, cli_live: bool,
                  confirm_token_path: Optional[Path] = None) -> "LiveExecutionGate":
        details = []
        env_set = os.environ.get("OPTINET_LIVE") == "1"
        if not env_set:
            details.append("env OPTINET_LIVE != 1")
        else:
            details.append("env OPTINET_LIVE = 1 ✓")

        token_match = False
        if not LIVE_TOKEN_F.exists():
            details.append(f"reference token missing at {LIVE_TOKEN_F}")
        elif confirm_token_path is None:
            details.append("--confirm-token not supplied")
        elif not Path(confirm_token_path).exists():
            details.append(f"confirm-token file not found: {confirm_token_path}")
        else:
            ref = LIVE_TOKEN_F.read_text().strip()
            sup = Path(confirm_token_path).read_text().strip()
            if ref and ref == sup:
                token_match = True
                details.append("token matches reference ✓")
            else:
                details.append("token does NOT match reference")

        kill_clear = not KILL_SWITCH.exists()
        if not kill_clear:
            details.append(f"kill-switch present at {KILL_SWITCH}")
        else:
            details.append("kill-switch clear ✓")

        if cli_live:
            details.append("--live flag passed ✓")
        else:
            details.append("--live flag NOT passed (dry-run)")

        return cls(
            cli_live_flag=cli_live,
            env_live_set=env_set,
            token_match=token_match,
            kill_switch_clear=kill_clear,
            detail=details,
        )


def make_order_client(live_clear: bool,
                      dry_run_ledger_path: Optional[Path] = None) -> OrderClient:
    """Pick the order client based on whether the live gate cleared.

    NEVER returns UpstoxOrderClient unless live_clear is True. This is the
    single place where the dry-run / live decision is made.
    """
    if not live_clear:
        return DryRunOrderClient(dry_run_ledger_path or ORDER_LEDGER)
    # Even when "clear", final disposition depends on the broker env var.
    # Default broker is Upstox in live mode; any other value also routes
    # through the Upstox stub, which raises until wired.
    return UpstoxOrderClient()
