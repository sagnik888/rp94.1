"""Production safety checks for REAL mode startup and switching."""

from __future__ import annotations

import json
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List

import pytz


IST = pytz.timezone("Asia/Kolkata")


def fyers_token_freshness_warning(now: datetime, token_path: Path | str = "fyers_token.json") -> str:
    """Pre-market checklist alarm: Fyers tokens are daily (no auto-refresh by
    broker policy), so a token not minted TODAY means degraded option-chain/WS
    feeds all session. Returns a warning string, or "" when fresh."""
    path = Path(token_path)
    if not path.exists():
        return "fyers token file missing — Fyers option-chain/WS feeds will be unavailable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = str(payload.get("timestamp") or "")
        token_dt = datetime.fromisoformat(stamp)
    except Exception:
        return "fyers token timestamp unreadable — re-auth Fyers before the session"
    if token_dt.tzinfo is not None:
        token_dt = token_dt.astimezone(IST).replace(tzinfo=None)
    session_date = (now.astimezone(IST) if now.tzinfo else IST.localize(now)).date()
    if token_dt.date() < session_date:
        return (
            f"fyers token is from {token_dt.date().isoformat()} (not refreshed today) — "
            "re-auth Fyers before the session or chains/WS degrade"
        )
    return ""


def _parse_time_setting(value: Any, default: str) -> dtime:
    text = str(value or default).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return datetime.strptime(default, "%H:%M:%S" if len(default.split(":")) == 3 else "%H:%M").time()


def _format_time(value: dtime) -> str:
    return value.strftime("%H:%M:%S" if value.second else "%H:%M")


from data.kite_auth import most_recent_token_cutoff  # noqa: E402


# IntelligenceAggregator publishes a WEIGHT-NORMALISED score: sum(votes) divided by the
# sum of the MAXIMUM vote each counted pillar could have cast. That spans a true -1..+1,
# so a threshold means what it looks like.
#
# It replaced a mean-of-votes on 2026-08-08 after measuring both over the same pillar
# votes: the mean was non-monotone (a CONFIRMING pillar lowered the score 4 times in 7),
# reached only 0.24 of the range its thresholds were written against, and scored a
# better-informed read LOWER because extra pillars grew the divisor faster than the sum.
# Discrimination was a wash (3.66 sd vs 3.74 sd) -- the case for the change is coherence,
# not sharpness.
#
# MEASURED ceiling on the new scale: 1.00 with thin evidence, 0.823 with a full option
# chain. Live always has a chain, so 0.82 is what actually applies and a gate above it
# is unreachable.
INTEL_SCORE_CEILING = 0.82

INTEL_SCORE_GATES = (
    "rare_burst_min_intel_score",
    "live_15min_trending_momentum_intel_threshold",
    "live_15min_momentum_intel_threshold",
    "live_5min_special_intel_threshold",
    "volume_spike_entry_strong_1min_intel_threshold",
)


def cheapest_one_lot_cost(settings, active_indices: list[str] | None = None) -> tuple[float, str]:
    """Rupees needed for ONE lot of the cheapest active index option, and how it was got.

    This used to be a flat Rs.50,000 constant described as "the floor for one index lot".
    That is a FUTURES margin figure. This system BUYS options, where one lot costs
    premium x lot_size, and the four indices differ by 6x on lot size alone:

        SENSEX      lot  20  ->  Rs. 2,400 at a Rs.120 ATM premium
        BANKNIFTY   lot  30  ->  Rs. 3,600
        NIFTY       lot  65  ->  Rs. 7,800
        MIDCPNIFTY  lot 120  ->  Rs.14,400   (Rs.48,000 at a Rs.400 premium)

    So the real range is roughly Rs.2,000 to Rs.48,000 depending on index and premium,
    and a flat Rs.50,000 declared an account unfundable when it could in fact buy several
    of those. Reported wrongly to this operator on 2026-08-08.

    The question that matters is "can I afford ANY entry", so this returns the MINIMUM
    across the ACTIVE indices. Lot sizes come from instruments.json; the ATM premium is an
    assumption, named in the message so the operator can judge it.

    entry_funds_warning_floor overrides the whole calculation when set above zero.
    """
    try:
        override = float(getattr(settings, "entry_funds_warning_floor", 0.0) or 0.0)
    except (TypeError, ValueError):
        override = 0.0
    if override > 0:
        return override, f"operator floor {override:,.0f}"

    try:
        premium = float(getattr(settings, "entry_funds_assumed_atm_premium", 120.0) or 120.0)
    except (TypeError, ValueError):
        premium = 120.0
    try:
        indices = getattr(settings, "get_instruments")()["indices"]
    except Exception:
        return 0.0, ""

    configured_active = (
        active_indices if active_indices is not None else getattr(settings, "active_indices", None)
    )
    active = [
        str(x).upper()
        for x in (configured_active if configured_active is not None else indices)
    ]
    best_cost, best_name, best_lot = 0.0, "", 0
    for name in active:
        row = indices.get(name) or {}
        try:
            lot = int(row.get("lot_size") or 0)
        except (TypeError, ValueError):
            continue
        if lot <= 0:
            continue
        cost = lot * premium
        if best_cost == 0.0 or cost < best_cost:
            best_cost, best_name, best_lot = cost, name, lot
    if best_cost <= 0:
        return 0.0, ""
    return best_cost, f"{best_name} {best_lot} x Rs.{premium:,.0f} assumed ATM premium"


def unreachable_intel_gates(settings) -> list[tuple[str, float]]:
    """Configured intelligence-score gates that the score can never reach."""
    found: list[tuple[str, float]] = []
    for name in INTEL_SCORE_GATES:
        try:
            value = float(getattr(settings, name, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value >= INTEL_SCORE_CEILING:
            found.append((name, value))
    return found


# The RR ladder in engine/signal_quality_policy.dynamic_rr is
#   min_rr + confidence_runway + grade_bonus + intel_adjustment
# whose terms are bounded by 1.6 + 1.44 + 1.15 + 0.833. confidence is clamped to [0,1]
# before the 3.2 multiplier, A+ is the largest grade bonus, and intel_adjustment divides
# normalize_intelligence_score (clamped to [-1,1]) by 1.20 — so its own upper clamp of
# 1.25 is unreachable too.
DYNAMIC_RR_LADDER_CEILING = 5.02


def inert_guard_warnings(settings) -> list[tuple[str, str]]:
    """Guards whose consumer runs but whose configured values make it do nothing.

    Distinct from config/inert_settings.INERT (nothing reads the setting) and from
    unreachable_intel_gates (a threshold above what the score can reach). Here the code
    executes on every trade and the arithmetic is empty — the quietest of the three,
    because nothing looks wrong at the call site.

    Reporting only. Retuning any of these changes live exit behaviour and is the
    operator's decision, exactly as config/inert_settings.py says of its own list.
    """
    found: list[tuple[str, str]] = []

    def _f(name: str, default: float) -> float:
        try:
            return float(getattr(settings, name, default) or default)
        except (TypeError, ValueError):
            return default

    # Scanner._dynamic_runner_lock_pct returns min(base, loose). A LOWER lock_pct leaves
    # more of the gain unlocked, i.e. more room, so `loose` must be strictly below the
    # base for the volatile-tape branch to mean anything.
    base_lock = _f("dynamic_rr_runner_lock_pct", 0.85)
    loose_lock = _f("trend_runner_lock_pct_high_vol", 0.80)
    if loose_lock >= base_lock:
        found.append((
            "trend_runner_lock_pct_high_vol",
            f"it is {loose_lock:.2f} against a base lock of {base_lock:.2f}, and the "
            f"function returns min(base, loose) — so the documented 'give the runner more "
            f"room in fast/volatile tape' branch returns the base unchanged in every "
            f"regime. Runners are trailed identically in calm and volatile tape.",
        ))

    # dynamic_rr_max caps a sum that cannot reach it.
    max_rr = _f("dynamic_rr_max", 5.0)
    if max_rr > DYNAMIC_RR_LADDER_CEILING:
        five_min_max = _f("dynamic_rr_5min_max", 2.5)
        found.append((
            "dynamic_rr_max",
            f"it is {max_rr:.1f} but the RR ladder cannot exceed "
            f"{DYNAMIC_RR_LADDER_CEILING:.2f}, so it never binds. What actually caps RR "
            f"on every 1min/5min entry is dynamic_rr_5min_max={five_min_max:.1f}.",
        ))

    return found


def calendar_coverage_status(data_provider, now) -> dict:
    """Which years the exchange calendar actually covers, and how long that lasts.

    The bundled NSE holiday list is versioned by year. Any date outside the covered
    years is treated as a holiday — deliberate, because trading a year whose holidays
    cannot be verified is the worse failure. But that cliff arrives silently: the
    system keeps running, scanning and serving a dashboard while placing nothing.

    Reporting how many days remain lets the operator fix it in advance instead of
    discovering it on the first morning of an uncovered year.
    """
    years = set()
    try:
        for day in getattr(data_provider, "nse_holidays", None) or set():
            head = str(day)[:4]
            if head.isdigit():
                years.add(int(head))
    except Exception:
        return {"years": [], "current_covered": True, "days_remaining": None}
    if not years:
        # No calendar loaded at all: is_market_holiday does not fail closed in that
        # case (`if coverage_years and ...`), so this is not the cliff.
        return {"years": [], "current_covered": True, "days_remaining": None}
    last = max(years)
    try:
        expires = date(last, 12, 31)
        remaining = (expires - now.date()).days
    except Exception:
        return {"years": sorted(years), "current_covered": now.year in years,
                "days_remaining": None}
    return {
        "years": sorted(years),
        "current_covered": now.year in years,
        "days_remaining": remaining,
        "expires_on": expires.isoformat(),
    }


def option_chain_readiness_max_age_seconds(settings: Any) -> float:
    """Age through which a verified chain remains safe for Auto startup."""
    try:
        network_ttl = float(
            getattr(settings, "intelligence_cache_ttl_seconds", 90.0) or 90.0
        )
    except (TypeError, ValueError):
        network_ttl = 90.0
    try:
        startup_age = float(
            getattr(settings, "option_chain_startup_max_age_seconds", 150.0)
            or 150.0
        )
    except (TypeError, ValueError):
        startup_age = 150.0
    # Never make startup freshness shorter than the normal broker refresh TTL.
    return max(10.0, network_ttl, startup_age)


def option_chain_auto_readiness_reasons(
    meta: Any,
    *,
    strike_count: int,
    age_seconds: float,
    min_strikes: int,
    max_age_seconds: float,
) -> List[str]:
    """Explain why a primary or independently certified fallback chain cannot arm Auto."""
    reasons: List[str] = []
    payload = meta if isinstance(meta, dict) else {}
    if int(strike_count or 0) < int(min_strikes or 0):
        reasons.append(f"{int(strike_count or 0)} strikes (need {int(min_strikes or 0)})")
    if (
        not isinstance(meta, dict)
        or (
            bool(payload.get("fallback", True))
            and not bool(payload.get("auto_certified", False))
        )
    ):
        reasons.append("fallback chain")
    if not bool(payload.get("expiry_aligned", False)):
        reasons.append("active expiry not verified")
    if not str(payload.get("active_expiry") or "").strip():
        reasons.append("active expiry missing")
    if not (0.0 <= float(age_seconds) <= float(max_age_seconds)):
        reasons.append("chain stale or missing")
    return reasons


def validate_real_mode_readiness(
    settings: Any,
    data_provider: Any = None,
    now: datetime | None = None,
    *,
    require_option_chains: bool = False,
    active_indices: list[str] | None = None,
) -> Dict[str, Any]:
    """Return a structured readiness decision before REAL mode is allowed.

    ``active_indices`` is an optional proposed execution universe.  Configure calls
    this before mutating live settings, so the option-chain and funding checks cannot
    accidentally certify the previous same-sized universe.
    """
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    else:
        now = now.astimezone(IST)

    blockers: List[str] = []
    warnings: List[str] = []

    env_path = Path(".env")
    if not env_path.exists():
        blockers.append(".env missing")

    # Upstox supplies market data. The bound execution broker proves account readiness separately.
    for key in ("upstox_api_key", "upstox_api_secret"):
        if not str(getattr(settings, key, "") or "").strip():
            blockers.append(f"{key} missing")

    try:
        from data.upstox_auth import get_upstox_auth
        upstox_status = get_upstox_auth().status()
    except Exception as exc:
        blockers.append(f"Upstox session state unavailable: {exc}")
        upstox_status = {}
    if upstox_status:
        configured_account = str(getattr(settings, "upstox_user_id", "") or "").strip()
        if configured_account and upstox_status.get("connected") and upstox_status.get("user_id") != configured_account:
            blockers.append("Upstox session account differs from configured market-data account")
        if upstox_status.get("auth_required") or not upstox_status.get("connected"):
            blockers.append(
                "Upstox daily login required"
                + (f" ({upstox_status.get('reason')})" if upstox_status.get("reason") else "")
                + " - complete it at /api/upstox_login"
            )

    execution_broker = getattr(data_provider, "execution_broker", None)
    if execution_broker is None and getattr(data_provider, "upstox_only", False) is True:
        blockers.append("Angel One execution broker unavailable")
    elif execution_broker is not None:
        try:
            if not execution_broker.ready_for_execution():
                blockers.append("Angel One execution session, position book, or order contract directory unavailable")
        except Exception as exc:
            blockers.append(f"Angel One execution readiness could not be verified ({type(exc).__name__})")

    risk_summary_fn = getattr(settings, "risk_configuration_summary", None)
    if callable(risk_summary_fn):
        try:
            risk_summary = risk_summary_fn()
            if not risk_summary.get("ok", True):
                blockers.extend(
                    f"risk configuration: {issue}"
                    for issue in (risk_summary.get("issues") or ["invalid"])
                )
            warnings.extend(
                f"risk configuration: {warning}"
                for warning in (risk_summary.get("warnings") or [])
            )
        except Exception as exc:
            warnings.append(f"risk configuration check unavailable: {exc}")

    if data_provider is not None:
        if not bool(getattr(data_provider, "is_connected", True)):
            blockers.append("Upstox market data provider not connected")
        try:
            health = data_provider.get_source_health()
            if health.get("all_brokers_unavailable"):
                blockers.append("all broker sources unavailable")
            if health.get("broker_degraded"):
                warnings.append("broker source degraded")
            if health.get("kite_stream_stalled") and not getattr(data_provider, "upstox_only", False):
                age = health.get("kite_tick_age_seconds")
                warnings.append(
                    "Kite tick stream stalled"
                    + (f" ({age:.0f}s since last tick)" if isinstance(age, (int, float)) else " (no ticks yet)")
                    + ("; REST prices still serving" if health.get("kite_data_fresh") else "; REST prices stale too")
                )
        except Exception as exc:
            warnings.append(f"source health unavailable: {exc}")

        # Distinguish a real holiday from an EXPIRED calendar before reporting either.
        # An uncovered year fails CLOSED — correctly, since trading a year whose holidays
        # cannot be verified is worse — but it surfaced as the bare string "market
        # holiday", which on an ordinary Monday reads like the exchange is shut rather
        # than like the software has run out of data. The system then runs, scans and
        # shows a dashboard while placing nothing, with one CRITICAL line as the only
        # evidence.
        coverage = calendar_coverage_status(data_provider, now)
        try:
            if data_provider.is_market_holiday(now):
                if coverage.get("years") and not coverage.get("current_covered"):
                    blockers.append(
                        f"exchange calendar covers {coverage['years'][0]}-{coverage['years'][-1]} "
                        f"only, so every {now.year} date reads as a holiday and NO trade can be "
                        "placed. Add this year's dates to data_store/market_holidays.json."
                    )
                else:
                    blockers.append("market holiday")
        except Exception as exc:
            warnings.append(f"holiday calendar check unavailable: {exc}")

        # Warn while there is still time to act, rather than on the morning it stops.
        try:
            remaining = coverage.get("days_remaining")
            warn_days = max(
                0, int(getattr(settings, "calendar_coverage_warn_days", 45) or 0)
            )
            if (
                coverage.get("current_covered")
                and remaining is not None
                and 0 <= remaining <= warn_days
            ):
                warnings.append(
                    f"exchange calendar runs out on {coverage.get('expires_on')} "
                    f"({remaining} days). After that every session reads as a holiday and "
                    "trading stops silently — update data_store/market_holidays.json."
                )
        except Exception as exc:
            warnings.append(f"calendar coverage check unavailable: {exc}")

        # Same principle applied to the REPLAY data. A backtest run over a thin or
        # different window produces a number that looks exactly as authoritative as one
        # run over a full window, and on 2026-08-17 that cost hours: two installs were
        # compared whose runs shared no calendar days at all, and the current code
        # replayed over the other's window makes +Rs.48,750 against -Rs.64 over its own.
        # Coverage has to travel with the result or the result gets over-read.
        try:
            from engine.replay_coverage import coverage_warnings

            warnings.extend(coverage_warnings())
        except Exception as exc:
            warnings.append(f"replay coverage check unavailable: {exc}")

    # Free cash vs what one lot actually costs. An account that cannot fund a single
    # lot will have EVERY entry refused by RMS, which surfaces only as a generic
    # rejection per signal. Warn rather than block: margin moves intraday and a stale
    # reading must not veto a session the operator can still fund.
    try:
        _cash = execution_broker.available_funds() if execution_broker is not None else None
        _floor, _detail = cheapest_one_lot_cost(settings, active_indices=active_indices)
        if _cash is not None and _floor > 0 and _cash < _floor:
            warnings.append(
                f"broker free cash is {_cash:,.2f}, below the {_floor:,.0f} needed for "
                f"the CHEAPEST one-lot option entry ({_detail}) - every entry will be "
                "refused until the account is funded"
            )
    except Exception:
        pass

    if require_option_chains:
        if data_provider is None:
            blockers.append("option-chain readiness unavailable")
        else:
            readiness_indices = [
                str(item or "").strip().upper()
                for item in (
                    active_indices
                    if active_indices is not None
                    else (getattr(settings, "active_indices", []) or [])
                )
                if str(item or "").strip()
            ]
            readiness_indices = list(dict.fromkeys(readiness_indices))
            min_strikes = max(
                3,
                int(getattr(settings, "auto_option_chain_min_strikes", 3) or 3),
            )
            # The network refresh TTL controls how often a broker is called; it is
            # not the expiry of already-valid intelligence. A full four-index morning
            # warmup can itself exceed 90s, which made its first chains "stale" before
            # the warmup completed and permanently prevented configured Auto startup.
            max_age = option_chain_readiness_max_age_seconds(settings)
            meta_by_index = getattr(data_provider, "_option_chain_meta", {}) or {}
            cache_by_index = getattr(data_provider, "_opt_chain_cache", {}) or {}
            now_epoch = now.timestamp()
            for instrument in readiness_indices:
                meta = meta_by_index.get(instrument, {})
                chain = cache_by_index.get(f"opt_chain_{instrument}")
                try:
                    count = int(len(chain)) if chain is not None else 0
                except Exception:
                    count = 0
                fetched_at = float(meta.get("fetched_at") or 0.0) if isinstance(meta, dict) else 0.0
                age = now_epoch - fetched_at if fetched_at > 0 else float("inf")
                reasons = option_chain_auto_readiness_reasons(
                    meta,
                    strike_count=count,
                    age_seconds=age,
                    min_strikes=min_strikes,
                    max_age_seconds=max_age,
                )
                if reasons:
                    blockers.append(
                        f"{instrument} option chain not Auto-ready: "
                        + ", ".join(reasons)
                    )

    special_session = False
    if data_provider:
        try:
            special_session = bool(data_provider.is_special_trading_session(now))
        except Exception:
            special_session = False

    if now.weekday() >= 5 and not special_session:
        blockers.append("weekend session")

    arm_start = _parse_time_setting(
        getattr(settings, "real_mode_premarket_arm_start", "09:00:00"),
        "09:00:00",
    )
    premarket_settle_end = _parse_time_setting(
        getattr(settings, "real_mode_premarket_settle_end", "09:08:00"),
        "09:08:00",
    )
    premarket_refresh_end = _parse_time_setting(
        getattr(settings, "real_mode_premarket_refresh_end", "09:14:30"),
        "09:14:30",
    )
    live_start = _parse_time_setting(
        getattr(settings, "real_mode_live_start", "09:15:00"),
        "09:15:00",
    )
    live_end = _parse_time_setting(
        getattr(settings, "real_mode_live_end", "15:30:00"),
        "15:30:00",
    )
    current_time = now.time()
    if not (arm_start <= current_time <= live_end):
        blockers.append(
            f"outside REAL arming window ({_format_time(arm_start)}-{_format_time(live_end)} IST)"
        )
    elif arm_start <= current_time <= premarket_settle_end:
        warnings.append(
            "pre-open settlement window: REAL armed while indicative prices settle; trade execution remains market/session gated"
        )
    elif premarket_settle_end < current_time <= premarket_refresh_end:
        warnings.append(
            "pre-open analysis window: data refresh and daily market context warmup allowed; trade execution remains market/session gated"
        )
    elif premarket_refresh_end < current_time < live_start:
        warnings.append(
            "market-open handoff window: REAL is armed; trade execution remains market/session gated"
        )

    # Position size is taken from config/instruments.json, a hand-maintained file, while
    # the exchange's own lot size is refreshed daily in the Kite contract master and read
    # by nobody. Exchanges revise index lot sizes periodically; when they do, every order
    # is sized on the wrong multiple and the broker refuses it as an invalid quantity —
    # an ordinary-looking rejection with no hint of the cause. Compare them and say so.
    if data_provider is not None:
        try:
            import json as _json
            from pathlib import Path as _Path

            raw = _json.loads(_Path("config/instruments.json").read_text(encoding="utf-8"))
            configured = {
                name: int((cfg or {}).get("lot_size") or 0)
                for name, cfg in (raw.get("indices") or raw).items()
                if isinstance(cfg, dict)
            }
            source = getattr(data_provider, "upstox", None)
            checker = getattr(source, "lot_size_mismatches", None)
            if callable(checker) and configured:
                for name, mine, actual in checker(configured):
                    blockers.append(
                        f"{name} lot size is {mine} in config/instruments.json but the "
                        f"exchange uses {actual}. Every order would be sized on the wrong "
                        "multiple and refused as an invalid quantity."
                    )
                unresolved = getattr(source, "lot_size_unresolved", None)
                if callable(unresolved):
                    names = unresolved(configured)
                    if names:
                        warnings.append(
                            "lot-size cross-check could not run for "
                            + ", ".join(names)
                        )
        except FileNotFoundError:
            pass
        except Exception as exc:
            warnings.append(f"lot-size cross-check unavailable: {exc}")

    # Options are the only instrument this system trades and they carry the widest
    # spreads, yet the charge model can be configured to assume they cost nothing to
    # cross. estimate_trade_charges reads
    #     float(getattr(settings, "slippage_opt_pct", 0.30) or 0.0) / 100.0
    # so an explicit 0.0 in .env wins over the 0.30 default and every simulated,
    # backtested and displayed option P&L comes out optimistic by the entire slippage
    # cost. The measured EWMA replaces this once real fills accumulate, but
    # slippage_stats.json currently holds FUT keys only, so nothing is replacing it.
    #
    # Not silently overridden — a deliberate zero is the operator's call — but they
    # should know that is what is configured.
    try:
        _opt_slip = float(getattr(settings, "slippage_opt_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        _opt_slip = 0.0
    if _opt_slip <= 0:
        warnings.append(
            "slippage_opt_pct is 0 — option P&L is modelled with NO slippage on the "
            "instrument this system actually trades, so every simulated and displayed "
            "figure is optimistic by the real cost of crossing the spread"
        )

    # Hedging is DORMANT: _pullback_hedge_arm/_pullback_hedge_release are written and
    # tested, but no code path places the second leg. The setting is a live toggle in the
    # dashboard, so an operator can switch it on and reasonably believe hedging is now
    # running. Say otherwise, loudly, at the one moment they are reading this output.
    _hedge_on = getattr(settings, "pullback_hedge_enabled", False)
    if _hedge_on is True or str(_hedge_on).strip().lower() in {"1", "true", "yes", "on"}:
        warnings.append(
            "pullback_hedge_enabled is ON but hedging is DORMANT — the arm/release "
            "decisions exist and nothing opens a hedge leg, so no hedge will be placed"
        )

    # A setting written into .env that nothing reads is worse than a missing feature: it
    # reads as a control that is working. ut_correlated_core_limit=2 says "maximum
    # simultaneous same-direction NIFTY/BANKNIFTY/SENSEX exposures" and the guard actually
    # allows 6. Say so at startup rather than letting a dead knob look live.
    # A threshold set above what the score can physically reach is a gate that never
    # fires, and it reads in .env as if it were tuned and active.
    for name, configured in unreachable_intel_gates(settings):
        warnings.append(
            f"{name}={configured} can never be met: the aggregate intelligence score is a "
            f"MEAN of votes whose magnitudes are all <= 0.35, so its absolute ceiling is "
            f"{INTEL_SCORE_CEILING} and its typical value is ~0.25-0.30. This gate is inert."
        )

    # Third class, next to the two above: the consumer EXISTS and runs on every trade,
    # but the shipped values make the operation a no-op, so a protection the operator
    # believes is active does nothing. Silent, because the code path is exercised — only
    # the arithmetic is empty.
    for name, why in inert_guard_warnings(settings):
        warnings.append(f"{name} is INERT at current values: {why}")

    try:
        from config.inert_settings import configured_but_inert, env_keys_from_file

        inert = configured_but_inert(env_keys_from_file())
        if inert:
            warnings.append(
                f"{len(inert)} setting(s) in .env have no consumer and do nothing: "
                + ", ".join(name for name, _why in inert[:6])
                + (f", +{len(inert) - 6} more" if len(inert) > 6 else "")
                + " (see config/inert_settings.py for what governs each instead)"
            )
    except Exception as exc:
        warnings.append(f"inert-settings check unavailable: {exc}")

    return {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "checked_at": now.isoformat(),
    }
