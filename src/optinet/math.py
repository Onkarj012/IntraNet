from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    spot: float,
    strike: float,
    years_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    t = max(years_to_expiry, 1.0 / 365.0)
    sigma = max(volatility, 1e-4)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type.upper().startswith("C"):
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
    return strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def black_scholes_delta(
    spot: float,
    strike: float,
    years_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    t = max(years_to_expiry, 1.0 / 365.0)
    sigma = max(volatility, 1e-4)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    if option_type.upper().startswith("C"):
        return norm_cdf(d1)
    return abs(norm_cdf(d1) - 1.0)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    years_to_expiry: float,
    option_type: str,
    rate: float = 0.06,
    low: float = 0.01,
    high: float = 3.0,
    iterations: int = 60,
) -> float:
    if price <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    intrinsic = max(0.0, spot - strike) if option_type.upper().startswith("C") else max(0.0, strike - spot)
    if price < intrinsic * 0.98:
        return float("nan")
    lo, hi = low, high
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        model = black_scholes_price(spot, strike, years_to_expiry, rate, mid, option_type)
        if model > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0
