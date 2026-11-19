"""Shared live, ledger, and backtest transaction-cost accounting."""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger


# Sanity bounds for calibrated per-leg slippage fractions. The realized EWMA is
# clamped into these so a run of lucky (or catastrophic) fills can't push the
# charge model into fantasy: options 0.02%..0.60% of premium, futures 0.2..5 bps.
_OPT_CALIBRATION_BOUNDS = (0.0002, 0.006)
_FUT_CALIBRATION_BOUNDS = (0.00002, 0.0005)


def estimate_upstox_derivative_fees(
    entry: float, exit_price: float, quantity: int, inst_type: str, *,
    instrument: str, direction: str, trade_date: date, multiplier: float = 1.0,
) -> float:
    """Estimate standard Upstox charges across two intraday execution legs."""
    option = inst_type == "OPT"
    bse = instrument.upper() == "SENSEX"
    units = quantity if option else quantity * multiplier
    buy, sell = entry * units, exit_price * units
    if not option and direction.upper() in {"SHORT", "SELL"}:
        buy, sell = sell, buy
    turnover = buy + sell
    brokerage = 40.0 if option else min(20, buy * .0005) + min(20, sell * .0005)
    if trade_date >= date(2026, 4, 1):
        stt_rate = .0015 if option else .0005
    elif trade_date >= date(2024, 10, 1):
        stt_rate = .001 if option else .0002
    else:
        stt_rate = .000625 if option else .000125
    if bse:
        exchange_rate = (.000325 if trade_date >= date(2024, 10, 1) else .000495) if option else 0
    elif trade_date >= date(2026, 3, 1):
        exchange_rate = .0003553 if option else .0000183
    elif trade_date >= date(2024, 10, 1):
        exchange_rate = .0003503 if option else .0000173
    else:
        exchange_rate = .000495 if option else .0000188
    ipft_rate = 0 if bse else (1e-9 if trade_date >= date(2026, 3, 1) else .000005 if option else .000001)
    exchange_charges = turnover * (exchange_rate + ipft_rate)
    gst = .18 * (brokerage + exchange_charges)
    stamp_duty = buy * (.00003 if option else .00002)
    sebi_fee = turnover * .000001
    return round(brokerage + exchange_charges + gst + sell * stt_rate + stamp_duty + sebi_fee, 2)


def _calibrated_leg_fraction(settings: Any, inst_type: str, instrument: str) -> float | None:
    """Measured per-leg slippage fraction from real fills, or None to use statics."""
    try:
        if not bool(getattr(settings, "slippage_auto_calibration_enabled", True)):
            return None
        from engine.slippage_tracker import get_slippage_tracker

        min_samples = int(getattr(settings, "slippage_calibration_min_samples", 5) or 5)
        max_age_days = float(
            getattr(settings, "slippage_calibration_max_age_days", 30.0) or 30.0
        )
        measured = get_slippage_tracker().calibrated_fraction(
            inst_type, instrument, min_samples=min_samples, max_age_days=max_age_days
        )
        if measured is None:
            return None
        lo, hi = (
            _OPT_CALIBRATION_BOUNDS
            if str(inst_type or "FUT").upper() == "OPT"
            else _FUT_CALIBRATION_BOUNDS
        )
        return min(max(float(measured), lo), hi)
    except Exception as exc:
        # Falling back to the static assumption is the safe answer, but doing it in
        # silence hides the difference between "no fills measured yet" and "this call is
        # broken" — a stub whose signature drifted was masked here for exactly that
        # reason, surfacing only as a quietly wrong charge.
        logger.debug(f"Slippage calibration unavailable, using static assumption: {exc}")
        return None


def estimate_trade_charges(
    entry_price: float,
    exit_price: float,
    quantity: int,
    inst_type: str,
    settings: Any,
    multiplier: float = 1.0,
    instrument: str = "",
    *,
    minimum_base_charge: float = 0.0,
) -> float:
    """Return the round-trip charge (flat cost + modeled slippage) used by every trade view.

    Adds a slippage/spread component so simulated and backtest P&L are realistic rather
    than optimistic (original audit H1). Options use a % of premium per leg (wide spreads);
    futures use bps of notional. Applied to entry + exit legs. When enough REAL broker
    fills have been measured, the realized EWMA (slippage_tracker) replaces the static
    per-leg assumption so the model reflects actual execution quality.
    """
    is_opt = str(inst_type or "FUT").upper() == "OPT"
    base = (
        float(getattr(settings, "opt_cost", 80.0) or 80.0)
        if is_opt
        else float(getattr(settings, "fut_cost", 200.0) or 200.0)
    )
    qty = max(0, int(quantity or 0))
    mult = abs(float(multiplier or 1.0)) or 1.0
    entry = abs(float(entry_price or 0.0))
    exit_ = abs(float(exit_price or 0.0))
    calibrated = _calibrated_leg_fraction(settings, inst_type, instrument)
    if is_opt:
        slip_pct = (
            calibrated
            if calibrated is not None
            else float(getattr(settings, "slippage_opt_pct", 0.30) or 0.0) / 100.0
        )
        slippage = (entry + exit_) * qty * slip_pct
    else:
        slip_bps = (
            calibrated
            if calibrated is not None
            else float(getattr(settings, "slippage_fut_bps", 2.0) or 0.0) / 10000.0
        )
        slippage = (entry + exit_) * qty * mult * slip_bps
    return round(max(base, minimum_base_charge) + max(0.0, slippage), 2)
