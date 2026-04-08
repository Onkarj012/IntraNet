"""
NSE (India) Transaction Cost Calculator for IntradayNet.

Accurate cost breakdown as of 2026:

Brokerage: ₹20/order flat (Zerodha, etc.)
STT: 0.025% sell side (intraday)
Exchange TXN: 0.00345%
SEBI turnover: 0.0001%
GST: 18% on (brokerage + exchange + SEBI)
Stamp duty: 0.003% buy side
Slippage: 0.05% per side

Total round-trip: ~0.15-0.20% for a ₹1L trade

Usage:
    costs = IndianMarketCosts()
    total = costs.total_cost(entry_price, qty)
    breakeven = costs.breakeven_move_pct(entry_price, qty)
"""

from dataclasses import dataclass


@dataclass
class IndianMarketCosts:
    brokerage_per_order: float = 20.0
    stt_rate: float = 0.00025
    exchange_txn: float = 0.0000345
    sebi_turnover: float = 0.000001
    gst_rate: float = 0.18
    stamp_duty: float = 0.00003
    slippage: float = 0.0005

    def total_cost(self, entry_price: float, qty: int) -> float:
        buy_value = entry_price * qty
        sell_value = entry_price * qty

        brokerage = self.brokerage_per_order * 2
        stt = sell_value * self.stt_rate
        exchange = (buy_value + sell_value) * self.exchange_txn
        sebi = (buy_value + sell_value) * self.sebi_turnover
        gst = (brokerage + exchange + sebi) * self.gst_rate
        stamp = buy_value * self.stamp_duty
        slippage_cost = (buy_value + sell_value) * self.slippage

        return brokerage + stt + exchange + sebi + gst + stamp + slippage_cost

    def cost_as_pct(self, entry_price: float, qty: int) -> float:
        position_value = entry_price * qty
        return self.total_cost(entry_price, qty) / position_value * 100

    def breakeven_move_pct(self, entry_price: float, qty: int) -> float:
        return self.cost_as_pct(entry_price, qty)

    def estimate_for_position(self, position_value: float) -> dict:
        qty = 100
        price = position_value / qty
        total = self.total_cost(price, qty)
        return {
            "position_value": position_value,
            "total_cost": total,
            "cost_pct": total / position_value * 100,
            "breakeven_move_pct": total / position_value * 100,
            "brokerage": self.brokerage_per_order * 2,
            "stt": price * qty * self.stt_rate,
            "exchange_txn": price * qty * 2 * self.exchange_txn,
            "gst": (self.brokerage_per_order * 2 + price * qty * 2 * self.exchange_txn) * self.gst_rate,
            "stamp": price * qty * self.stamp_duty,
            "slippage": price * qty * 2 * self.slippage,
        }

    def estimate_round_trip_fraction(
        self,
        entry_price: float,
        position_value: float = 100_000.0,
    ) -> float:
        qty = max(int(position_value / max(entry_price, 1e-6)), 1)
        return self.total_cost(entry_price, qty) / max(entry_price * qty, 1e-6)


DEFAULT_COSTS = IndianMarketCosts()


def estimate_liquidity_penalty(
    avg_daily_traded_value: float,
    median_minute_turnover: float,
) -> float:
    """
    Return an extra execution penalty as a fractional move.

    We keep this deliberately simple and deterministic for the v1 backend.
    Lower-liquidity symbols carry a larger penalty, which pushes expected net
    edge down even if the raw return forecast looks attractive.
    """
    adv = max(avg_daily_traded_value, 0.0)
    minute_turnover = max(median_minute_turnover, 0.0)

    penalty = 0.0
    if adv < 5_000_000:
        penalty += 0.0010
    elif adv < 20_000_000:
        penalty += 0.0005
    else:
        penalty += 0.0002

    if minute_turnover < 50_000:
        penalty += 0.0008
    elif minute_turnover < 200_000:
        penalty += 0.0004
    else:
        penalty += 0.0001

    return penalty


def print_cost_breakdown(position_value: float = 100_000):
    costs = DEFAULT_COSTS
    est = costs.estimate_for_position(position_value)

    print(f"\n{'=' * 50}")
    print(f"COST BREAKDOWN — Position: ₹{position_value:,.0f}")
    print(f"{'=' * 50}")
    print(f"Brokerage (×2):    ₹{est['brokerage']:.2f}")
    print(f"STT (sell side):   ₹{est['stt']:.2f}")
    print(f"Exchange TXN (×2): ₹{est['exchange_txn']:.2f}")
    print(f"GST (18%):         ₹{est['gst']:.2f}")
    print(f"Stamp duty:        ₹{est['stamp']:.2f}")
    print(f"Slippage (×2):     ₹{est['slippage']:.2f}")
    print(f"{'-' * 50}")
    print(f"TOTAL COST:        ₹{est['total_cost']:.2f}")
    print(f"Cost as %:          {est['cost_pct']:.3f}%")
    print(f"Breakeven move:     {est['breakeven_move_pct']:.3f}%")
    print(f"{'=' * 50}")
