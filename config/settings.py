"""
UT1 Index Trading System — Configuration
Pydantic-based settings with .env support
"""

import json
import dotenv
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _setting_enabled(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(value)


TIMEFRAME_ENTRY_POLICY_TIMEFRAMES = {
    "PRIMARY_15": frozenset({"15min"}),
    "INCLUDE_5MIN": frozenset({"5min", "15min"}),
    "PRIMARY_5": frozenset({"5min"}),
    "INCLUDE_1MIN_5MIN": frozenset({"1min", "5min"}),
    "INCLUDE_1MIN_5MIN_15MIN": frozenset({"1min", "5min", "15min"}),
    # 2026-09-26 user: every 1m/5m/15m combination on the dashboard TF toggle. A 1m
    # entry is still burst/spike/snapback only under every policy.
    "PRIMARY_1": frozenset({"1min"}),
    "INCLUDE_1MIN_15MIN": frozenset({"1min", "15min"}),
}


def normalize_timeframe_entry_policy(value: Any) -> str:
    """Return a supported operator timeframe-entry policy."""
    policy = str(value or "INCLUDE_5MIN").upper()
    return policy if policy in TIMEFRAME_ENTRY_POLICY_TIMEFRAMES else "INCLUDE_5MIN"


def timeframe_entry_timeframes(value: Any) -> frozenset[str]:
    """Fresh-entry frames enabled by an operator timeframe policy."""
    return TIMEFRAME_ENTRY_POLICY_TIMEFRAMES[normalize_timeframe_entry_policy(value)]


PROTECTED_SECRET_ENV_KEYS = {
    "UPSTOX_API_SECRET",
    "UPSTOX_PASSWORD",
    "ANGELONE_API_KEY",
    "ANGELONE_PASSWORD",
    "ANGELONE_PIN",
    "ANGELONE_TOTP_SECRET",
    "ANGELONE_SECRET_KEY",
    "FYERS_SECRET_KEY",
    "FYERS_PASSWORD",
}


def _set_env_key_with_retry(env_key: str, value: str, attempts: int = 4) -> bool:
    """Write one .env key, tolerating a transient Windows file lock.

    python-dotenv writes a temp file and os.replace()s it over .env. On Windows that
    fails outright with "[WinError 5] Access is denied" whenever anything else holds a
    handle — an antivirus scan or an open editor is enough. The lock is momentary, so a
    short backoff clears it, where the single attempt simply dropped the setting.
    """
    import time as _time

    delay = 0.15
    for attempt in range(max(1, attempts)):
        try:
            dotenv.set_key('.env', env_key, value)
            return True
        except (PermissionError, OSError) as exc:
            if attempt == attempts - 1:
                print(
                    f"Could not persist {env_key} to .env after {attempts} attempts: {exc}. "
                    "Close any editor or sync tool holding the file."
                )
                return False
            _time.sleep(delay)
            delay *= 2
    return False


def _clean_env_secret(value: Any) -> str:
    return str(value or "").strip().strip("'").strip('"')


class Settings(BaseSettings):
    """Main application settings loaded from .env"""

    # ─── Upstox API v2 (Primary Broker) ──────────────────────
    upstox_user_id: str = ""
    upstox_password: str = ""
    upstox_api_name: str = "ut1"
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_redirect_uri: str = "http://127.0.0.1:8500/api/upstox/callback"
    upstox_public_ip: str = ""
    upstox_token_file: str = "upstox_token.json"
    upstox_http_timeout_seconds: float = 7.0
    upstox_failover_threshold_seconds: float = 30.0

    # ─── AngelOne API (Backup / Fallback Broker) ─────────────
    # ── Zerodha Kite Connect (primary broker) ────────────────────────────────
    # Kite has no password login: the account password is never stored or used by
    # this system. A browser login issues a request_token, which is exchanged for an
    # access_token that expires every morning (~06:00-07:30 IST), so a manual login
    # is required once per trading day via /api/kite_login.
    kite_api_key: str = ""
    kite_api_secret: str = ""
    # Kite checks the TCP source address against the app's allowed-IP list and refuses
    # EVERY order from an address that is not on it, while READS keep working — so it
    # presents as "the strategy stopped trading", not as a broken connection.
    #
    # This machine has both address families and api.kite.trade resolves to IPv6 first,
    # so orders were leaving on the IPv6 address (2026-08-04, three signals dead). An
    # IPv6 egress is normally a rotating privacy address, so it cannot be whitelisted in
    # any durable way. Pinning Kite to IPv4 leaves exactly one address to manage.
    kite_force_ipv4: bool = True
    # The IPv4 address whitelisted on the Kite developer console. Set it and the readiness
    # gate warns the moment this connection's real egress drifts off it — BEFORE a signal
    # dies. Leave blank to skip the check (correct if the console's IP list is empty,
    # which is the right setup for a dynamic home connection).
    kite_registered_ip: str = ""
    # Optional proxy for Kite REST calls ONLY (orders, margins, order book). Kite checks
    # the source address on order PLACEMENT, so this is the one path that needs a stable
    # address — point it at a host with a fixed IP and whitelist that IP instead of a
    # home connection that keeps moving. The tick WebSocket deliberately stays direct, so
    # market data does not take an extra hop; the added latency lands only on order
    # submission, where a few ms is irrelevant.
    # Any form requests understands: http://host:3128, socks5h://host:1080 (needs PySocks).
    # Blank = direct, unchanged behaviour.
    kite_http_proxy: str = ""
    kite_user_id: str = ""
    # Where the daily access token is cached. Kept out of .env because it rotates.
    kite_token_file: str = "kite_token.json"
    # Kite's redirect target must match the one registered on the app at
    # developers.kite.trade. The dashboard serves the landing route itself.
    kite_redirect_uri: str = ""
    # NRML = overnight/carryforward, MIS = intraday auto-squareoff. The system's
    # product_type strings are broker-neutral and mapped in KiteBroker.
    kite_default_product: str = "NRML"
    # Permanent postback hostname, if you have one (a NAMED Cloudflare tunnel on a
    # domain you control). Set it and scripts/start_postback_tunnel.py verifies that
    # URL instead of minting a throwaway quick-tunnel hostname that changes every
    # restart and has to be re-registered at Zerodha by hand each time.
    kite_postback_url: str = ""
    # Entries go in at the market by default. A LIMIT that does not fill is a missed
    # signal, and the cancel/replace path already converts a stale limit to MARKET
    # after ~12s — so a limit mostly buys a 12-second delay and an extra failure mode.
    # Slippage is bounded instead by market_protection on the order itself.
    #
    # A LIMIT is used only when the signal clears this confidence bar, where waiting
    # for a price is worth the risk of not being filled at all. 100 disables limits
    # entirely; 0 makes every priced entry a limit (the old behaviour).
    # 0-1, like every other confidence floor here (trade.confidence is a fraction and is
    # printed as {confidence:.0%}). This shipped as 95.0, which no confidence could ever
    # reach, so the LIMIT entry path was dead and every order went out MARKET. Values
    # above 1 are still accepted and read as a percentage.
    entry_limit_confidence_threshold: float = 0.95

    # Server-side life span for entry LIMIT orders, in minutes. 0 disables it.
    #
    # This is a BACKSTOP, not the primary mechanism: broker_cancel_replace_enabled
    # already cancels a stale limit after ~12s and re-sends it as MARKET, which
    # actually gets the position taken — TTL only cancels. Set well beyond that
    # window so it never races the replace logic. Its job is the case the replace
    # logic cannot cover: if this process dies while a limit rests, Zerodha expires
    # the order instead of leaving it to fill hours later at a stale price.
    kite_entry_limit_ttl_minutes: int = 5

    angelone_api_key: str = ""
    angelone_client_id: str = ""
    angelone_password: str = ""
    # Legacy MPIN input is a fallback only. A managed canonical password must
    # not be displaced by an older process PIN. Do not persist the input alias.
    angelone_pin: str = Field(default="", exclude=True, repr=False)
    angelone_totp_secret: str = ""
    angelone_secret_key: str = ""
    # The public IP whitelisted on your SmartAPI app (smartapi.angelone.in ->
    # My Profile -> app -> Redirect/Whitelisted IP). AngelOne validates the TCP
    # source IP and refuses EVERY order with AG7002 when it does not match, so a
    # dynamic ISP address silently kills a whole session. Set this and the runtime
    # watchdog warns the moment the machine's real egress IP drifts off it.
    # Leave blank to disable the check.

    # ─── Fyers API ───────────────────────────────────────────
    fyers_app_id: str = "6P37DKRJG6-100"
    fyers_secret_key: str = ""
    fyers_login_id: str = ""
    fyers_password: str = ""
    fyers_redirect_uri: str = "http://127.0.0.1:8001/"

    # Kite owns execution. Angel One supplements it; FYERS remains a capability
    # backup, including constituent volume until replacement parity is proven.
    # The earlier two-provider candidate and legacy route remain selectable.
    supplementary_market_data: Literal["upstox", "upstox_angel", "fyers", "kite_angel", "angel_fyers"] = "upstox"
    replacement_chain_strikes_each_side: int = 30
    constituent_membership_file: str = "data_store/index_constituents.json"
    constituent_history_refresh_seconds: float = 60.0
    constituent_cache_dir: str = "data_store/candidate_candles"

    # ─── Trading Mode ────────────────────────────────────────
    # Canonical values separate data source from execution authority. HISTORICAL,
    # PAPER and REAL remain accepted aliases during migration.
    trading_mode: Literal[
        "REPLAY", "LIVE_PAPER", "LIVE_REAL", "HISTORICAL", "PAPER", "REAL"
    ] = "REPLAY"
    auto_mode: bool = False
    strategy_revision: str = "ut1-current"
    strategy_direction_mode: Literal["LONG_SHORT", "LONG_ONLY", "SHORT_ONLY"] = "LONG_SHORT"
    # Grade every candidate with independently refreshed intelligence from its OWN
    # timeframe. 1m is a fast support/monitoring layer for a 15m trade, not a renamed
    # 15m decision. When an own-timeframe decision is requested, missing evidence must
    # not silently borrow the 5m cache.
    per_timeframe_grading_enabled: bool = True
    own_timeframe_intel_fail_closed: bool = True
    active_indices: list[str] = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    # One final fresh-exposure invariant, evaluated from actual execution prices after
    # all lane/conversion mutations and once more in the queued broker worker. Recovery
    # of an existing broker position is deliberately not a fresh entry.
    entry_policy_enabled: bool = True
    entry_policy_min_rr: float = 1.5
    entry_policy_expected_move_floor_pct_by_index: dict[str, float] = {
        # These percentages are directional expected favourable movement, not an
        # already-realised spot move. They are compared against the 25th PERCENTILE of
        # the historical forward move (entry_policy_forecast_quantile = 0.25), so the
        # floor is a statement about the PESSIMISTIC case: "in 75% of analogous setups
        # we would have got at least this much".
        #
        # RECALIBRATED 2026-08-30. The previous values (0.15 / 0.15 / 0.15 / 0.25) were
        # unreachable -- not rarely, never. Replaying this gate's own computation over
        # the stored candles, the rolling q25 it reads spans 0.017%-0.063% and its
        # MAXIMUM over the whole history is:
        #
        #     NIFTY 5min 0.0469   BANKNIFTY 5min 0.0662   SENSEX 5min 0.0522
        #     NIFTY 15min 0.0773  BANKNIFTY 15min 0.1220  MIDCPNIFTY 15min 0.1399
        #
        # Every one of those maxima is BELOW the floor it had to clear. Put the other
        # way, 0.15% sits at the 91st percentile of NIFTY 5min forward moves and 0.25%
        # at the 96.5th for MIDCPNIFTY -- so the gate demanded that the 25th percentile
        # clear a level only 3-32% of moves ever reach. This is the LAST gate before an
        # entry, so what it refused had already passed everything else.
        #
        # It went live on 2026-08-28 and the system took ZERO trades that session,
        # against 5 / 10 / 17 / 14 on the four sessions before it.
        #
        # The values below are the 10th percentile of each index's own rolling q25 on
        # the 5-minute series -- the binding timeframe, since 15min q25 runs roughly
        # double. They refuse 5-9% of 5min decisions and ~0% of 15min. Deliberately at
        # the loose end: being too tight is the failure that just happened, and a gate
        # that refuses the weakest tenth can be tightened later on evidence, whereas one
        # that refuses everything produces no evidence at all.
        "NIFTY": 0.023,
        "BANKNIFTY": 0.029,
        "SENSEX": 0.022,
        "MIDCPNIFTY": 0.031,
    }
    # MAJOR_WIN_GUARD's arming rung, as a share of what the position put at risk.
    #
    # It was a flat Rs.3,000 across a 6x lot-size spread, and it had NEVER FIRED on half
    # the grid: over 26+ sessions the largest peak ever recorded is Rs.2,603 on NIFTY
    # (0 of 80 lifecycles reached the rung) and Rs.1,556 on BANKNIFTY (0 of 56). The
    # guard was structurally inoperative on two of four indices, so its health there was
    # not "good", it was unobserved.
    #
    # Cost-normalising it while keeping the Rs.3,000 anchor was measured and REJECTED:
    # SENSEX's median position costs Rs.9,519 so Rs.3,000 is 31.5% of it, and applying
    # 31.5% to BANKNIFTY/MIDCPNIFTY (median cost ~Rs.19,500) would raise their rung to
    # ~Rs.6,140 and make it LESS reachable, not more.
    #
    # Calibrated instead on what a win in this book actually looks like. Peak P&L as a
    # share of position cost, per index, over 193 winning lifecycles:
    #
    #     index        p50    p75    p90    p95    max
    #     NIFTY        3.0    5.7   11.7   15.4   28.6
    #     BANKNIFTY    1.8    3.4    5.6    9.6   18.6
    #     SENSEX       3.0    5.4   12.3   13.9   31.3
    #     MIDCPNIFTY   2.8    5.6   14.0   30.0   59.9
    #
    # Each index arms on its OWN p90, so "major win" means the same thing everywhere --
    # the top decile of that contract's wins -- rather than the same rupee number
    # meaning six different things. BANKNIFTY is genuinely lower in relative terms and
    # a pooled percentage leaves it inert, which is why this is per index.
    #
    # Falls back to the flat Rs.3,000 when the row cannot be priced, which preserves the
    # old behaviour rather than arming on an unpriced position.
    major_win_guard_cost_pct_by_index: dict[str, float] = {
        "NIFTY": 11.5,
        "BANKNIFTY": 5.5,
        "SENSEX": 12.0,
        "MIDCPNIFTY": 14.0,
    }
    # Futures are margined instruments: option premium-cost percentages are not a
    # meaningful denominator. A major futures win arms at this multiple of the frozen
    # entry-stop risk, with the flat rupee rung retained as a conservative minimum.
    major_win_guard_futures_r_multiple: float = 1.50
    # Previous-session bars may warm indicators, but every timeframe starts from a
    # clean directional state at 09:15. One current-session forming bar is enough for
    # the strict opening-quality routes; no first-hour 5-minute blackout.
    live_signal_5min_warmup_bars: int = 1
    major_win_guard_flat_fallback_rs: float = 3000.0

    entry_policy_underlying_stop_max_pct_by_index: dict[str, float] = {
        "NIFTY": 0.15,
        "BANKNIFTY": 0.15,
        "SENSEX": 0.15,
        "MIDCPNIFTY": 0.15,
    }
    # Explicit forecast horizons. The 1m series forecasts the next 15 minutes and is
    # therefore honest about its role supporting live 15m decisions.
    entry_policy_forecast_horizon_minutes_by_timeframe: dict[str, int] = {
        "1min": 15,
        "5min": 30,
        "15min": 60,
    }
    entry_policy_forecast_quantile: float = 0.25
    entry_policy_forecast_min_samples: int = 30
    entry_policy_forecast_max_samples: int = 480
    entry_policy_forecast_max_age_seconds: float = 90.0
    entry_policy_geometry_max_age_seconds: float = 15.0
    # Bid/ask shadow mark-outs for BOTH accepted and rejected entry decisions. These
    # rows are research evidence only: they never promote a rule or alter a live order.
    # A bounded, durable pending book survives process restarts and emits one outcome
    # per horizon so after-cost expectancy can be compared without candle-close or LTP
    # look-ahead.
    entry_shadow_outcomes_enabled: bool = True
    entry_shadow_horizons_minutes: list[int] = [15, 30, 60]
    entry_shadow_update_interval_seconds: float = 5.0
    # Cache-only bid/ask evidence older than this is unavailable, never silently
    # treated as an executable mark. Keep this separate from the worker cadence so
    # quote freshness remains an explicit, tunable measurement contract.
    entry_shadow_quote_max_age_seconds: float = 10.0
    entry_shadow_entry_quote_grace_seconds: float = 30.0
    entry_shadow_exit_quote_grace_seconds: float = 180.0
    entry_shadow_max_watches: int = 1024
    entry_shadow_update_batch_size: int = 64
    entry_shadow_state_file: str = "data_store/entry_policy_shadow_pending.json"
    # Indices that must NEVER trade futures (route to options instead). Was set in
    # .env but never read by any code — so MIDCPNIFTY was reaching the FUT stop
    # gate and being rejected. Now enforced in _resolve_instrument_type.
    auto_futures_blocked_indices: list[str] = ["MIDCPNIFTY"]
    # Lean instrument routing toward options: send the loss-prone marginal futures
    # cases (choppy chop, ordinary catch-all) to defined-risk options instead.
    # Genuine FUT cases (high IV, high spot vol, 0DTE-late) are preserved.
    lean_routing_to_options: bool = True
    # Slow-grind trend routing: a real but low-thrust directional move (trending
    # regime, modest ADX, compressed per-bar ATR%) earns less on an option's delta
    # than it loses to theta+spread; route it to linear futures instead of letting
    # the lean catch-all send it to options. Aligned-trend FUT passes the regime
    # veto and uses the trend stop tolerance.
    slow_grind_futures_routing_enabled: bool = True
    slow_grind_min_adx: float = 18.0
    slow_grind_max_atr_pct: float = 0.38
    # ── Structural price-action patch (separate, A/B-toggleable) ──
    # Adds S/R + swing-structure + candle-pattern + volume-profile awareness.
    # DEFAULT OFF: enable to A/B-test P&L, keep if it helps, else remove cleanly.
    # block_adverse=False -> advisory (telemetry only); True -> rejects ADVERSE setups.
    price_action_module_enabled: bool = False
    price_action_block_adverse: bool = False
    price_action_adverse_threshold: float = -0.50
    # Deliberately SEPARATE from the setting above, which the exit-side loss-recovery
    # checks in scanner.py read and where -0.50 has always been the effective value.
    # The ENTRY veto has always cut at -0.40, hardcoded in intelligence/price_action.py.
    # Pointing both at one number silently moves one of them: at -0.50 the entry veto
    # releases 145 of 564 recorded ADVERSE refusals (25.7%), and at -0.40 the exit
    # checks tighten. Two questions, two thresholds.
    price_action_entry_adverse_threshold: float = -0.40
    # Day-type-aware premium/discount (2026-07-10): on a CONFIRMED aligned trend
    # day (effective phase, >= min confidence) the "chasing the extreme" penalty
    # is scaled by trend_scale — continuation entries at the top (bottom) of the
    # range on trend days were being blocked blind. Range days keep the full
    # penalty. Set scaled=False to restore day-type-blind behaviour exactly.
    price_action_premium_trend_scaled: bool = True
    price_action_premium_trend_scale: float = 0.5
    price_action_premium_trend_min_confidence: float = 0.70
    post_exit_reversion_enabled: bool = False
    # Reversion scalp window after exit/distress (minutes). Counter-trend 1min
    # signals inside the window trade the snap-back with RR >= min_rr, tight stop,
    # and scalp-lock; the index cooldown is bypassed for them only.
    post_exit_reversion_window_minutes: float = 3.0
    # RR floor for reversion / snapback scalps (aligned with 1m spike scalp RR 1.5).
    post_exit_reversion_min_rr: float = 1.5
    # Also open the window after stop-loss / early-loss style exits (not only wins).
    post_exit_reversion_include_stop_loss: bool = True
    # Open window when exit reason mentions OI/PCR/regime/flow reverse pressure.
    post_exit_reversion_include_oi_pcr: bool = True
    # While an open trade is underwater and tape reverses against it, arm window.
    post_exit_reversion_open_distress_enabled: bool = True
    # Only arm SNAPBACK after 5min/15min trades (not continuous 1min scalps).
    post_exit_reversion_source_timeframes: List[str] = Field(
        default_factory=lambda: ["5min", "15min"]
    )
    # Soft loss band (₹): 0 → -soft (immediate reverse while still shallow).
    post_exit_reversion_loss_band_soft_rs: float = 1000.0
    # Deep loss band (₹): -soft → -deep (fast opposite move + reverse regime).
    post_exit_reversion_loss_band_deep_rs: float = 2000.0
    # Size-of-event gate (OR logic — either band qualifies):
    #   fast: >= 0.30% over ~5 minutes
    #   slow: >= 0.40% over ~10 minutes
    # Micro-chop after tiny exits does not open SNAPBACK.
    post_exit_reversion_fast_move_pct: float = 0.30
    post_exit_reversion_fast_lookback_minutes: float = 5.0
    post_exit_reversion_slow_move_pct: float = 0.40
    post_exit_reversion_slow_lookback_minutes: float = 10.0
    # Legacy aliases (kept for config UI / older saves; dual gate is authoritative).
    post_exit_reversion_min_move_pct: float = 0.30
    post_exit_reversion_move_lookback_minutes: float = 5.0
    # Pure SNAPBACK bare-min data confirm (below full SPIKE floors). Ensures the
    # 1min counter-signal is tape-backed, not a naked UT1 flip.
    post_exit_reversion_min_volume_ratio: float = 1.5
    post_exit_reversion_min_range_atr_ratio: float = 1.0
    post_exit_reversion_min_close_location: float = 0.55
    # Rare dual TF fill: when 1m (spike/snapback) AND 5m/15m are both excellent
    # same-direction, allow up to 2 entries (second usually OPT; never 2x FUT).
    rare_dual_tf_entry_enabled: bool = True
    rare_dual_tf_min_grade: str = "A"
    rare_dual_tf_min_confidence: float = 0.70
    rare_dual_tf_prefer_opt_second: bool = True
    price_action_reversal_relief_enabled: bool = True # Confirmed reversals can override into-S/R vetoes
    # Structure robustness + LuxAlgo-style context filters (only active when the
    # module above is enabled). swing_strength is the confirmed fractal half-width
    # (was a noisy 2). premium_discount penalises chasing the extreme of the swing
    # range; liquidity penalises entering into an equal-highs/lows stop-hunt pool.
    # All CONTEXT-ONLY (adjust the price-action score/verdict), never entry triggers.
    price_action_swing_strength: int = 5
    price_action_premium_discount_enabled: bool = True
    price_action_liquidity_enabled: bool = True

    # ─── Risk Settings ───────────────────────────────────────
    # Absolute backstop for rare burst mode: normal scanner capacity is 4 indices x
    # 2 trades, while a guarded all-index impulse can temporarily stretch to 4 x 3.
    max_concurrent_positions: int = 12
    # Per-bucket concurrent caps. NIFTY/BANKNIFTY/SENSEX share CORE; MIDCPNIFTY is
    # independent. The concurrency toggle now changes strictness, not normal capacity:
    # normal -> CORE 6 + MIDCAP 2 = 8; rare-burst guard-off cap -> CORE 9 + MIDCAP 3.
    core_bucket_max_positions: int = 6
    core_bucket_max_positions_guarded: int = 6
    midcap_bucket_max_positions: int = 2
    midcap_bucket_max_positions_guarded: int = 2
    rare_burst_core_bucket_max_positions: int = 9
    rare_burst_midcap_bucket_max_positions: int = 3
    max_concurrent_trades_per_index: int = 2
    rare_burst_max_concurrent_trades_per_index: int = 3
    rare_burst_third_trade_enabled: bool = True
    rare_burst_min_confidence: float = 0.95
    rare_burst_min_volume_ratio: float = 3.5
    rare_burst_min_adx: float = 35.0
    rare_burst_min_intel_score: float = 0.70
    rare_burst_min_breadth_votes: int = 3
    rare_burst_require_hourly_anchor: bool = True
    rare_burst_require_session_bias: bool = True
    rare_burst_block_range_phase: bool = True
    max_daily_loss_pct: float = 3.0
    capital_total: float = 1100000.0   # Total Demat Capital (Base for Daily Loss)
    
    # Futures Specific
    capital_fut: float = 750000.0   # Shared futures capital across active indices
    risk_fut_pct: float = 2.0        # FUT Daily Max Loss Limit (%) — circuit breaker ONLY (single source of truth with .env)
    # Absolute ceiling for the regime-selected FUT stop tolerance (e.g. 0.10% of LTP).
    # This is the traded stop policy — not a lot-size hardcap.
    futures_sl_pct: float = 0.15
    fut_force_fixed_lots: bool = False # Always trade 1 lot for Futures
    fut_cost: float = 200.0         # Transaction cost per Futures trade
    
    # Options Specific
    capital_opt: float = 300000.0   # Shared options capital across active indices
    risk_opt_pct: float = 8.0        # OPT daily loss budget on option capital (%) — circuit breaker ONLY (single source of truth with .env)
    options_sl_pct: float = 15.0    # OPT per-trade premium stop cap (%)

    # EXPERIMENT ONLY: stand aside entirely while the session bias reads BEARISH.
    # The measured cohort is advisory candidate-mark data without an authoritative
    # broker fill join, and this blanket gate conflicts with the confirmed-trend-day
    # census waiver that admits bearish SHORTs. Keep it opt-in until held-out,
    # executable option outcomes and live/replay parity support a default.
    session_bearish_standaside_enabled: bool = False

    # ─── Lot sizing policy ───────────────────────────────────
    # DEFAULT OFF: always trade the configured dashboard/user lot size for both
    # FUT and OPT. Daily risk % and max_trade_loss_abs do NOT shrink lots.
    # Stops remain futures_sl_pct / options_sl_pct from settings.
    auto_risk_sizing_enabled: bool = False
    # Equal-rupee-risk option sizing (2026-10-08 user): lots = floor(risk_per_trade_rs /
    # one-lot premium risk), 1..risk_based_lots_max. One lot already over budget is kept
    # ('allow_one') or refused beyond the tolerance ('refuse'). Lot sizes come from
    # config/instruments.json (NIFTY 65, BANKNIFTY 30, SENSEX 20, MIDCPNIFTY 120).
    risk_based_lots_enabled: bool = True
    risk_per_trade_rs: float = 2000.0
    risk_based_lots_max: int = 3
    risk_based_lots_over_budget_action: str = "allow_one"
    risk_based_lots_over_budget_tolerance: float = 0.5
    entry_pre_submit_lot_validation_timeout_seconds: float = 10.0
    # Optional research mode only (when auto_risk_sizing_enabled=True):
    risk_per_trade_fut_pct: float = 0.0
    risk_per_trade_opt_pct: float = 0.0
    risk_per_trade_daily_fraction: float = 0.5
    # ── Lot scale policy ─────────────────────────────────────────────
    # Default: always 1× configured lots (user_lots / user_lots_fut).
    # Cascade: OPT only multiplies 2×–3× max (FUT stays 1×).
    # Heavy trend (non-cascade, ≥2/4 indices same dir + TF align): OPT 2× only.
    # A+/power_move alone never multiplies.
    cascade_lot_scale_enabled: bool = True
    cascade_lot_multiplier: float = 2.0  # legacy fallback
    cascade_lot_multiplier_fut: float = 1.0   # FUT always flat
    cascade_lot_multiplier_opt: float = 3.0   # cascade OPT max (clamped with min below)
    cascade_lot_multiplier_opt_min: float = 2.0  # cascade OPT floor when scaling
    # 0 = no hard max (operator margin responsibility); >0 clamps cascade size (R7).
    cascade_lot_max_lots: int = 0
    cascade_lot_max_lots_fut: int = 0
    cascade_lot_max_lots_opt: int = 0
    cascade_lot_scale_warn: bool = True
    # Non-cascade heavy trend: 3+ of 4 indices align trend+TF → OPT ×2 only
    # (2026-09-26 user: was 2 of 4; every engine, S1 and UT SMART EXIT included).
    heavy_trend_lot_scale_enabled: bool = True
    heavy_trend_lot_multiplier_opt: float = 2.0
    heavy_trend_lot_multiplier_fut: float = 1.0
    heavy_trend_min_aligned_indices: int = 3  # 3/4 indices
    # ── Multi-signal mixed book (dual engines × multi-TF × multi-index) ──
    # When several instruments/timeframes/engines fire together, take at most
    # ONE futures (fixed 1 lot) and put the rest on options (2–4 indices).
    # CE/PE, strike, stops still follow TF/engine + existing option logic.
    multi_signal_mixed_book_enabled: bool = True
    multi_signal_min_instruments: int = 3
    multi_signal_min_timeframes: int = 2
    multi_signal_min_engines: int = 2
    multi_signal_max_fut: int = 1
    multi_signal_fut_lots: int = 1
    # Core 3×(3 primary+2 dual) − 1 FUT + midcap 2+1 ≈ up to 17 OPT on full cascade
    multi_signal_max_opt_instruments: int = 18
    multi_signal_collapse_best_tf_per_index: bool = True
    # ── Cascade multi-TF concurrency (CASCADE ONLY; normal stays 2 / dual 2+1) ──
    # NIFTY + BANKNIFTY + SENSEX each get the same multi-TF stack:
    #   primary = 1×1min + 1×5min + 1×15min (cascade_primary_max_tf)
    #   dual    = +1×5min lane-B + +1×15min lane-B (cascade_dual_addon_max)
    # MIDCPNIFTY (FUT blocked → OPT only):
    #   solo engine  = best 2 of {1m, 5m, 15m}
    #   dual engine  = best 2 main-lane + best 1 dual of {5m, 15m}  (= 2+1)
    # Mixed book: at most 1 FUT (1 lot) across the whole wave; rest OPT.
    cascade_multi_tf_stack_enabled: bool = True
    cascade_primary_max_tf: int = 3
    cascade_dual_addon_max: int = 2
    # Indices that receive full cascade multi-TF stack
    cascade_multi_tf_core_indices: list[str] = ["NIFTY", "BANKNIFTY", "SENSEX"]
    # MIDCAP cascade stack (tighter than core)
    cascade_midcap_solo_max_tf: int = 2          # best 2 of 1/5/15 solo engine
    cascade_midcap_dual_main_max: int = 2        # best 2 main-lane
    cascade_midcap_dual_addon_max: int = 1       # + best 1 dual of 5/15 → 2+1
    # Dual modes on normal (non-cascade) bars: base concurrent + 1 dual addon.
    dual_mode_max_per_index: int = 3  # 2 + 1
    # Dual clone reduction: same index + same TF from both engines within this
    # window → keep only the FIRST signal (earlier timestamp); drop the clone.
    dual_clone_reduction_enabled: bool = True
    dual_clone_reduction_window_seconds: int = 60  # 1 minute
    # Heavy-trend peer vote freshness (seconds). Same-wave + recent cache votes
    # within this window count toward 2/4 align without waiting for open fills.
    heavy_trend_vote_max_age_seconds: float = 600.0  # 10 minutes
    # Cascade-aware bucket / global caps (R4): multi-TF stack needs room beyond
    # normal 4×2 core / midcap 2. Prevents late indices being starved mid-wave.
    cascade_core_bucket_max_positions: int = 12   # e.g. 3 indices × up to 4 slots
    cascade_midcap_bucket_max_positions: int = 3  # midcap solo 2 / dual 2+1
    cascade_max_concurrent_positions: int = 18    # soft ceiling during cascade waves
    # A flat options_sl_pct stop is timeframe- and ATR-blind: a 5min scalp gets the
    # same 15% room as a 15min swing. When enabled, derive the premium stop from the
    # signal's ATR natural stop (converted to premium via delta) bounded by the
    # per-timeframe min/max % below, so 5min stops sit tighter and 15min wider —
    # driven by real volatility. options_sl_pct stays the hard ceiling.
    option_stop_tf_atr_aware: bool = True
    option_stop_5min_min_pct: float = 4.0
    option_stop_5min_max_pct: float = 10.0
    option_stop_15min_min_pct: float = 6.0
    option_stop_15min_max_pct: float = 15.0
    opt_cost: float = 80.0          # Transaction cost per Options trade
    # Modeled slippage (H1) so sim/backtest P&L isn't optimistic; applied per entry+exit leg.
    # These are the STATIC defaults; once real fills accumulate, the realized-slippage
    # tracker overrides them with measured values (see slippage_auto_calibration_enabled).
    slippage_opt_pct: float = 0.30  # Options: % of premium per leg (wide bid/ask)
    slippage_fut_bps: float = 2.0   # Futures: bps of notional per leg
    # ─── Execution slippage controls ─────────────────────────
    # Marketable-limit buffer for option entries (% past signal price). This is the
    # worst-case entry slippage CAP: the order crosses the spread and fills at the
    # touch, but can never fill worse than signal ± buffer. ATM index option spreads
    # are typically 1-4 ticks, so 0.40% (with the tick floor below) fills reliably
    # while capping freak fills at ~2.5x less slippage than the old 1.0% buffer.
    option_limit_buffer_pct: float = 0.40
    # Absolute floor on the option limit buffer in ticks (0.05 each) so cheap
    # premiums (e.g. Rs.30 far-week MIDCPNIFTY) still cross the spread and fill.
    option_limit_buffer_min_ticks: int = 2
    # Guarded touch-quote execution. Option entries use a fresh executable book and a
    # marketable LIMIT (ask+ticks for BUY, bid-ticks for SELL), so a thin contract can
    # never turn an AUTO entry into an unbounded MARKET fill. AUTO additionally refuses
    # to enter when the book is missing, stale, crossed, or too wide.
    execution_use_touch_quote: bool = True
    execution_touch_quote_shadow_enabled: bool = True
    execution_touch_quote_strict: bool = True
    execution_touch_quote_max_age_seconds: float = 1.5
    # Reprice an accepted REAL advisory row (price/stop/target) at the acceptance-time
    # executable quote. OFF = record-only: the quote is stamped on the row but the row
    # keeps its stabilization-start price. Moving it shifts the gate inputs paper exits
    # feed (loss streak, cooldown, reversion window, manual Execute refusal), and
    # spot-mapped trailing needs entry_spot re-anchored first (2026-09-17 review).
    accepted_entry_rebase_enabled: bool = False
    option_touch_limit_extra_ticks: int = 1
    option_touch_quote_max_spread_pct: float = 1.0
    option_touch_quote_max_spread_ticks: int = 40
    # 2026-08-11 (rtracker, 28 Jul - 11 Aug): one entry cost Rs.10,402 -- ten times the
    # entire 106-trade book. A signal timestamped 09:10:00, before the 09:15 open, was
    # executed at 09:15:34.516 on an entry mark of 131.70 and exited 40 MILLISECONDS
    # later at 52.30. An option cannot round-trip -60% in 40ms of real fills: the mark
    # was the previous close, which is what an option's LTP still reads until its first
    # trade of the day prints. _option_premium_quote_plausible could not catch it -- it
    # only bounds premiums from ABOVE (to spot*0.12 ~ Rs.2,880 on NIFTY), because its
    # job is catching an index price leaking into premium space, not a stale-but-
    # in-range mark. Two independent nets, because the touch quote is not always usable:
    #
    #   (a) refuse an entry whose SIGNAL predates the session open. Across the two-week
    #       window this blocks exactly 2 trades, both losers, together -Rs.10,866.
    #   (b) refuse an entry whose MARK is inconsistent with the live bid/ask. The bad
    #       mark sat 2.5x above the real offer; normal spread noise is far inside 1.5x.
    entry_preopen_signal_guard_enabled: bool = True
    entry_mark_touch_validation_enabled: bool = True
    entry_mark_max_ask_multiple: float = 1.50
    entry_mark_min_bid_multiple: float = 0.50
    # The two guards above live in TradeManager.open_trade, which the MANUAL signal path
    # never calls — it stamps accepted_by_gate on the candidate as-is, and every P&L
    # figure downstream is computed from that recorded price. 2026-08-11 15:09:36: NIFTY
    # 24500PE was accepted at 133.00 with spot at 24453.65, i.e. Rs.87 of time value on a
    # same-day-expiry option 20 minutes from expiry, against a true price of 40.65. The
    # mark was ~2.5 hours stale and the book took -Rs.6,083 the market never charged.
    # Both conditions must hold: a large FRACTIONAL divergence and a material RUPEE gap,
    # so a cheap contract ticking a few paise is never mistaken for a stale mark.
    candidate_entry_mark_validation_enabled: bool = True
    candidate_entry_mark_max_divergence_pct: float = 25.0
    candidate_entry_mark_min_gap_rs: float = 500.0
    # Futures entries were plain MARKET (uncapped in a freak spike). ON places a
    # marketable LIMIT at signal price +/- fut_limit_buffer_bps instead: same
    # immediate fill in normal books, but worst-case entry slippage is hard-capped.
    fut_entry_protection_limit_enabled: bool = True
    fut_limit_buffer_bps: float = 5.0
    # Measure REAL broker fills against signal/decision prices and feed the charge
    # model with the realized EWMA instead of the static assumptions above, once
    # enough samples exist. Bounded to sane ranges so one bad fill can't distort it.
    slippage_auto_calibration_enabled: bool = True
    slippage_calibration_min_samples: int = 5
    # A sample count is not evidence of currency. On 2026-08-11 slippage_stats.json held
    # 12,051 FUT samples at ewma_frac exactly 0.0, every row stamped last_leg "ENTRY",
    # from an integration that has since been retired — and because the measured value
    # REPLACES the static assumption rather than blending with it, that fossil pinned
    # modelled futures slippage to the 0.2bp floor of _FUT_CALIBRATION_BOUNDS. Rows
    # older than this fall back to the static model. 0 disables the age test.
    slippage_calibration_max_age_days: float = 30.0
    broker_fill_confirmation_timeout_seconds: float = 12.0
    broker_cancel_replace_enabled: bool = True
    # Final broker-side affordability preflight.  The requested order value is
    # compared with the freshest available cash/margin snapshot plus this safety
    # buffer; the very short cache avoids duplicating funds API calls for paired
    # checks while preventing a stale balance from authorising later entries.
    entry_margin_buffer_pct: float = 10.0
    # Buying a multi-lot entry the account cannot fund trims it to the lots it can fund
    # (at least 1) instead of refusing the whole trade; only a 1-lot shortfall refuses.
    entry_trim_lots_to_funds_enabled: bool = True
    entry_margin_cache_seconds: float = 2.0
    # Pre-market warning floor for broker free cash. Index option lot margins run
    # ~37k-56k, so an account below this cannot fund a single lot and every entry will
    # be refused by RMS with a generic rejection. Advisory only: it warns, never blocks,
    # because margin moves intraday and the operator may fund mid-session.
    # 0 = derive it. A flat floor cannot describe an option account: this system BUYS
    # options, so one lot costs premium x lot_size, and the four indices span 6x on lot
    # size alone (SENSEX 20, MIDCPNIFTY 120). The real range is about Rs.2,000 to
    # Rs.48,000. Set this above zero only to override the calculation.
    entry_funds_warning_floor: float = 0.0
    # The ATM premium the funding estimate assumes. Named in the warning so the operator
    # can judge it against the day's actual chain.
    entry_funds_assumed_atm_premium: float = 120.0
    recovery_auto_flatten_seconds: float = 5.0
    # When the order book cannot be read we cannot prove a position is unprotected, so a
    # flatten is withheld — a double exit (stop still resting + our market exit) opens a
    # naked reversed leg, which is worse than a late exit. That is safe intraday but it
    # can carry an NRML position past the close with only an alert.
    #
    # 0 (default) keeps that behaviour: withhold indefinitely and escalate to the operator.
    # Set >0 to accept the double-exit risk after that many seconds of being blocked and
    # flatten anyway; the trade is then flagged for a reversal watch, because a stop that
    # fires afterwards opens a position in the opposite direction. This is a deliberate
    # risk trade-off and is left switched off until an operator chooses it.
    recovery_flatten_block_deadline_seconds: float = 0.0
    max_entry_orders_per_minute: int = 6
    entry_order_rate_window_seconds: float = 60.0
    entry_order_kill_latch_seconds: float = 300.0
    clock_skew_max_seconds: float = 180.0
    # Two feeds disagreeing about one INDEX by more than this (percent of LTP) is a fault
    # in one of them: both quote the same computed number, so there is no spread to
    # explain a gap away. Past it the system keeps the venue that fills the orders and
    # quarantines the other for that symbol; inside it, the fresher quote wins.
    #
    # 0.05% was too loose to catch a stale feed on an index this system gates entries on:
    # NIFTY at 24,000 could carry a 12-point disagreement silently. At 0.015% that is
    # ~3.6 points on NIFTY, ~7 on BANKNIFTY and ~12 on SENSEX — tight enough that a
    # lagging feed is caught while a normal tick-timing difference is not.
    quote_divergence_alert_pct: float = 0.015

    # The CRITICAL bar, separate from the tracking bar above (2026-08-19).
    #
    # One threshold used to do both jobs, and at 0.015% it logged CRITICAL on ordinary
    # tick-timing jitter. 2026-08-18: 1,815 CRITICAL lines, 98% of every critical event
    # in the session. Measured over those same lines, the divergence distribution is:
    #     BANKNIFTY  n=524  p50 0.020%  p90 0.030%  p99 0.040%  max 0.060%
    #     MIDCPNIFTY n=456  p50 0.020%  p90 0.020%  p99 0.040%  max 0.150%
    #     NIFTY      n=411  p50 0.020%  p90 0.020%  p99 0.040%  max 0.050%
    #     SENSEX     n=424  p50 0.020%  p90 0.020%  p99 0.030%  max 0.120%
    # The noise floor tops out around 0.04%, so 0.05% sits just above it: it would have
    # kept 10 of the 1,815 lines (0.6%) — the actual outliers — and silenced the rest.
    #
    # Tracking is deliberately left at the sensitive 0.015%. The dashboard and watchdog
    # still see every disagreement via get_quote_divergences(); what changes is only
    # which ones are worth waking someone for. An alarm that fires 1,815 times a day is
    # not an alarm, and it was burying everything else in the log.
    quote_divergence_critical_pct: float = 0.05
    shared_host_mode: bool = False
    allow_plaintext_broker_secrets: bool = True
    option_strike_selection: Literal["ATM", "ITM", "BOTH"] = "BOTH"
    # INTERIM P0 CONTAINMENT. Kite reports POSITIONS per tradingsymbol, so when several
    # REAL trades share one contract the broker cannot say who owns what:
    # _allocate_broker_quantity splits the row by a rule, and that rule is a GUESS as
    # soon as one sibling exits and the remainder must be re-attributed. broker_quantity
    # SIZES the protective stop, so a wrong guess arms the wrong quantity.
    #
    # One REAL position per contract removes the need to guess at all. It costs cascade
    # multi-timeframe stacking on a SINGLE strike; stacking across different strikes or
    # instruments is unaffected, as is every non-REAL mode.
    #
    # Retire this by setting it False once the persisted per-order fill ledger is in
    # place and certified against sibling exits, restarts and missing acknowledgements.
    forbid_concurrent_same_contract_real: bool = True
    # The second lock, and the one that makes lifting the first survivable. Turning
    # forbid_concurrent_same_contract_real off does NOT return to unrestricted stacking:
    # a second REAL trade on a contract is still refused unless the fill ledger can
    # account for every trade already on it.
    #
    # That is the difference between "an operator decided the ledger works" and "the
    # ledger demonstrates it works, per contract, at the moment of the entry". A trade
    # the ledger cannot attribute is the entire hazard -- reconciliation falls back to
    # splitting the aggregate, and that split becomes a guess the moment a sibling exits
    # -- so refusing to stack onto it confines the guess to one position, where it
    # cannot mis-size anything.
    #
    # Set False only for a deliberate unrestricted run (a backtest, or an emergency
    # where stacking matters more than attribution).
    same_contract_requires_ledger_evidence: bool = True


    # With "BOTH", the ATM and ITM candidates are scored and the higher score wins.
    # Nothing in that score describes how TRADEABLE the contract is, so two setups of
    # near-identical quality were separated by whatever the score rounded to, and the
    # thinner strike won as often as not. Delta is already per-strike, so the pricing
    # was right; only the choice between the two was unranked.
    #
    # The chain this system already fetches carries call_oi / call_volume / put_oi /
    # put_volume PER STRIKE, so this needs no extra broker call and adds no latency --
    # it reads the cached chain (allow_fetch=False) or does nothing.
    option_strike_liquidity_ranking_enabled: bool = True
    # Only ever a TIE-BREAK. A materially better setup must still win, so liquidity may
    # decide only when the two scores are within this band.
    option_strike_liquidity_score_band: float = 0.05
    # And only when the alternative is better on BOTH open interest and traded volume by
    # at least this ratio. Requiring both, rather than a weighted blend, keeps the test
    # scale-free: OI is standing interest and volume is today's activity, and inventing
    # a weight between them would be the arbitrary part.
    option_strike_liquidity_min_advantage: float = 1.25
    # Absolute contract-admissibility floor, evaluated after strike selection and
    # again from any fresh executable quote.  Thresholds are expressed in lots so
    # the same policy scales across NIFTY/BANKNIFTY/SENSEX lot-size changes.  A
    # metric is enforced only when the broker/chain explicitly supplied it; a
    # measured zero is rejected, while an unavailable metric is kept distinct.
    option_contract_liquidity_gate_enabled: bool = True
    option_contract_min_oi_lots: float = 5.0
    option_contract_min_volume_lots: float = 1.0
    option_contract_min_touch_depth_lots: float = 1.0
    # Long-premium options are the system's primary product.  FUT and AUTO stay
    # explicit operator modes; a fresh install must never silently become a
    # hybrid/FUT book merely because INST_PREF was omitted from its environment.
    inst_pref: Literal["FUT", "OPT", "AUTO"] = "OPT"
    sl_mode: Literal["NATURAL", "HARDCODED"] = "NATURAL"
    signal_grade_preference: Literal["auto", "B", "B+", "A", "A+"] = "A"
    # Quality admission is owned by engine/quality_verdict.py. The former B/C grade
    # bypass, rank override and fitted late-ADX ceiling were removed rather than left
    # as dashboard knobs that appear to work. Unknown legacy .env keys are ignored by
    # model_config below.
    ut_timeframe_entry_policy: Literal[
        "PRIMARY_15",
        "INCLUDE_5MIN",
        "PRIMARY_5",
        "INCLUDE_1MIN_5MIN",
        "INCLUDE_1MIN_5MIN_15MIN",
        "PRIMARY_1",
        "INCLUDE_1MIN_15MIN",
    ] = "INCLUDE_5MIN"
    strict_entry_timeframes: bool = False
    ut_5min_option_min_confidence: float = 0.60
    ut_5min_loss_cooldown_minutes: int = 45
    # Enforces the line above. Until 2026-08-10 that value was read by nothing but two
    # broadcast payloads: the real post-loss block was the flat (and regime-scoped)
    # index_cooldown_minutes, so a configured 45 meant an actual 3. Ships False so
    # turning the gate on is an explicit decision — at four indices and a ~360-minute
    # entry window a 45-minute per-index lockout caps each index near 8 entries, which
    # is a material change to how much the system trades.
    ut_5min_loss_cooldown_enabled: bool = False
    # Local quality a candidate must have before a cross-index reversal may carry it
    # past the raw quality floor. Ported back from restore_point_51, where the bypass
    # read `market_wide and market_wide_ok`; this build had lost the guard and the
    # bare flag admitted anything (2026-08-12: grade C at 12% and 13%).
    # restore_point_51 sets B+/60% here, but its raw-entry `allowed` expression
    # carries an admission route this build does not have — `or bool(census_ctx)`,
    # from _census_range_scalp_context. Porting the restriction without its
    # compensator made this build strictly TIGHTER than the one being matched.
    # 2026-08-12 14:30 BANKNIFTY sat exactly in the gap: grade B, 54%, market-wide
    # core 3/3 total 4, and the session's largest winner at +Rs.1,300. The junk this
    # guard exists to stop was C grade at 12-13%. The floor every other bypass in
    # that expression already uses separates the two with room to spare, so the bar
    # is set there until the census path exists to justify a higher one.
    # ─── Confidence calibration (de-saturation) ──────────────
    # The legacy model was confidence = clamp(0.5 + Σpillars, 0, 1), which pinned
    # at 1.0 for any net-positive setup and destroyed ranking power. The logistic
    # squash keeps the same midpoint (score 0 -> 0.50) and closely tracks the old
    # mapping across the 0.55-0.85 gate band, but asymptotes toward 1.0 instead of
    # clipping, so grades regain discrimination. k≈5 preserves existing gate floors.
    ut_confidence_logistic: bool = True
    ut_confidence_logistic_k: float = 5.0
    # R5: fit confidence to realized win-rate (isotonic) from outcome_calibration.csv.
    # DEFAULT OFF — respects the ADVISORY_ONLY guardrail. Even when enabled it is a
    # no-op until the dataset is ready (>= min_samples AND both win/loss classes).
    ut_confidence_calibration_enabled: bool = False
    ut_confidence_calibration_min_samples: int = 200
    # Collection/readiness and live application are deliberately separate. A fitted
    # curve may be inspected offline, but cannot change grades until walk-forward and
    # shadow validation have been reviewed and this second switch is explicitly set.
    ut_confidence_calibration_live_apply_enabled: bool = False
    ut_confidence_calibration_min_segment_samples: int = 50

    # ─── Regime veto (directional FUT in chop) ───────────────
    # Block directional FUTURES entries when the regime is choppy/ranging/volatile
    # AND ADX is below the floor (no trend to ride). Options and power-moves are
    # exempt. This is a hard veto, not the soft -0.10 regime score penalty.
    regime_veto_enabled: bool = True
    regime_veto_adx_floor: float = 18.0
    # A 60-bar high/low TOUCH is only a real breakout with momentum or volatility
    # expansion. Below this ADX (and without vol expansion) it's noise grazing the
    # band in a dead tape — label it RANGING/VOLATILE honestly, not BREAKOUT.
    regime_breakout_adx_floor: float = 20.0
    # FUT entries are rejected when the natural ATR stop exceeds the regime-selected
    # stop tolerance x this multiplier. futures_sl_pct remains the absolute ceiling.
    # This governs only the REJECT threshold, not the traded stop itself.
    futures_sl_safety_multiplier: float = 2.0
    # Early loss-cut (loss-guard): symmetric for FUT and OPT — cut a failing trade
    # at this R-multiple of initial risk before the full stop is hit. Mirrors the
    # profit-lock / peak-threat path on the winner side. Winners untouched.
    fut_early_loss_cut_enabled: bool = True
    fut_loss_cut_r: float = 0.8
    opt_early_loss_cut_enabled: bool = True
    opt_loss_cut_r: float = 0.8
    # ── Peak-anchored give-back cap (2026-08-17) ────────────────────────────────
    # The single biggest measured leak in the book, and it is a CALIBRATION problem
    # rather than a signal one. Across 288 positions in every stored session:
    #   77% of trades went green at some point   -> entries work
    #   aggregate peak reached Rs.179,599, booked Rs.48,687 -> only 27% captured
    #   105 trades went green then closed NEGATIVE, booking -Rs.27,585
    # The cause is that every profit guard arms ABOVE where the trades actually live.
    # Peak distribution of green trades: p25 Rs.216, median Rs.454, p75 Rs.875, against
    # arms of Rs.250 (profit lock), Rs.500 (giveback) and Rs.750 (peak-threat) — so the
    # strongest guard sees only 26% of trades, and 61 green-then-red trades peaked below
    # even the lowest arm and had NO profit protection possible at all (-Rs.15,102).
    #
    # This guard is deliberately NOT another retention ladder. Retention is a fraction of
    # peak, which is too loose on a small peak (38% of Rs.200 is Rs.76) and gives away a
    # fortune on a large one (62% of Rs.1,194 is Rs.740). Premium noise is roughly
    # ABSOLUTE, so the cap is absolute: once the trade has shown `arm` of profit, never
    # let it hand back more than `cap` from its own high-water mark.
    #
    # It is really a loss-limiter anchored to the PEAK instead of to entry. Its value is
    # turning -Rs.428/-477/-534 into about -Rs.50, not banking gains.
    #
    # Validated by replaying recorded pnl_path samples (49 positions, 4 sessions):
    #     policy                net      delta   helped  hurt  worst   per-session
    #     ACTUAL             Rs.  694       +0        0     0      0   -705/+1209/+512/-323
    #     arm 100 / cap 200  Rs.6,569   +5,875       23     3   -238   +471/+1239/+2653/+2207
    #     arm 150 / cap 200  Rs.5,210   +4,516       17     2    -66
    #     arm  50 / cap 250  Rs.4,849   +4,155       20     5   -412
    #     arm  50 / cap 150  Rs.4,640   +3,946       24     9   -706  (08-13 goes NEGATIVE)
    # arm 100 / cap 200 is positive in all four sessions with the best helped:hurt ratio.
    #
    # It DOES cost money sometimes and that is accepted: a trade peaking Rs.1,179 books
    # Rs.896 here against Rs.961 actual, and one trade breached the cap off a running
    # peak of Rs.335 and then ran to Rs.670. 23 helped against 3 hurt is the trade.
    #
    # Do NOT tighten cap below 200 without re-running the replay: at 150 the hurt count
    # jumps to 9, worst-case damage to -Rs.706, and one whole session flips negative.
    # The allowance is max(cap_rs, cap_pct x peak) — a HYBRID, and the pct half is not
    # optional. A flat Rs.200 was tried first and the suite caught it immediately: on a
    # Rs.4,859 peak that fires on a 4% pullback and guts the major-win guard, whose whole
    # job is to let a large winner breathe through a retracement. The replay set topped
    # out near Rs.1,194 of peak, so the large-peak regime was outside the validated range.
    # 0.25 keeps the absolute cap binding below ~Rs.800 of peak (where the leak actually
    # is) and hands large peaks a proportional allowance instead.
    #
    # And there is a CEILING. The replay set's peaks are median Rs.247, p75 Rs.526,
    # p90 Rs.1,038, max Rs.2,401 — so the evidence is dense below ~Rs.1,000 and thins to
    # two positions above Rs.1,500. The large-winner regime already has purpose-built,
    # tested handling: the major-win guard and the trend lock deliberately tolerate a deep
    # pullback on a big runner, and the suite caught a flat Rs.200 cap pre-empting exactly
    # that (a Rs.4,000 peak handing back Rs.1,250, which they exist to hold). Above the
    # ceiling this guard steps aside rather than overruling them on two samples.
    # The ceiling excludes 2 of 49 replay positions (4%), so the measured result below is
    # essentially unchanged by it. Raise it only after the replay set covers that range.
    # DEFAULT OFF as of 2026-08-18. It nets more on the replay (+Rs.5,505 against the
    # breakeven stop's +Rs.3,192) but it earns that by exiting trades that are STILL IN
    # PROFIT — a Rs.800 peak sitting at +Rs.600 trips it — and it hurt 3 of 49 replayed
    # positions. The operator's stated complaint is precisely "exiting early on gaining
    # trades", so a rule whose whole mechanism is capping a live gain works against the
    # thing being fixed, whatever its total. The breakeven stop below delivers the other
    # half of the complaint with zero hurt. Turn this back on only to trade some winner
    # upside for a smaller give-back, deliberately.
    peak_giveback_cap_enabled: bool = False
    peak_giveback_cap_arm_rs: float = 100.0
    peak_giveback_cap_rs: float = 200.0
    peak_giveback_cap_pct: float = 0.25
    peak_giveback_cap_max_peak_rs: float = 1500.0

    # ── Breakeven stop: a proven trade may not close negative ───────────────────
    # The operator's own description of the problem was "exiting early on gaining
    # trades and late on losing trades". Measured over 288 positions, those are two
    # different buckets of very different size:
    #   105 trades went GREEN and then closed NEGATIVE, booking -Rs.27,585. Had they
    #       merely closed FLAT the book goes Rs.48,687 -> Rs.76,272, i.e. +57%.
    #   117 winners kept 75% of their peak, leaving Rs.37,429 on the table.
    #
    # This rule addresses ONLY the first bucket, and that is the point. Once a trade
    # has shown `arm` of profit its stop moves to `floor`; it is otherwise left alone.
    # It cannot exit a trade that is still in profit, so it cannot make the second
    # bucket worse — which the give-back cap above can, and does on 3 of 49 replayed
    # positions.
    #
    # Replayed through the real guard (49 positions, 4 sessions):
    #     ACTUAL                     Rs.  694
    #     breakeven after peak 150   Rs.3,886   +3,192   helped 10   HURT 0
    #     breakeven after peak 250   Rs.1,761   +1,067   helped  4   HURT 0
    #     give-back cap              Rs.6,199   +5,505   helped 22   hurt 3
    # Zero hurt at every arm tested: it never books a worse number than what actually
    # happened, because it only fires on a trade already on its way to a loss.
    #
    # floor = the ROUND-TRIP COST, not zero (2026-08-19).
    #
    # A floor of 0.0 states "do not close negative" in GROSS terms, so the trade still
    # closes negative net of the ~Rs.80 it paid to open and close. Worse, the condition
    # `now <= 0` is by construction only observable once the trade is already there:
    # P&L is sampled on a poll (2026-08-18 cadence: median 2.4-6.9s, max 20s) and option
    # premium moves a long way between samples. That session's five firings each had a
    # POSITIVE reading 1.9-8.6s before they fired, and booked -Rs.135 to -Rs.294 anyway:
    # Rs.1,242 handed back below the last price the guard itself had seen.
    #
    # Re-measured over every recorded pnl_path (60 positions, 5 sessions, 08-12..08-18),
    # replayed through the real guard, holding arm at 150:
    #     ACTUAL                  Rs.2,256
    #     floor   0 (was)         Rs.5,849   +3,592   helped 15   hurt 0
    #     floor  80 (now)         Rs.6,687   +4,431   helped 15   hurt 4   worst -Rs.226
    #     floor 100               Rs.4,917   +2,661   helped 15   hurt 6   worst -Rs.1,806
    #     floor 150               Rs.5,729   +3,473   helped 15   hurt 8   worst -Rs.1,806
    #
    # This DOES give up the "hurt 0" property the floor-0 version had, and that is the
    # deliberate trade: 4 positions book Rs.226-or-less worse so that 15 stop booking a
    # loss on a trade they had already proven.
    #
    # 80 is chosen because it is the round trip, NOT because it is the argmax. Note the
    # cliff above it: at floor 100 a single position (2026-08-14 SENSEX 5min, actual
    # +Rs.1,901, peak Rs.2,401) dips to +Rs.95 at 25.3s and gets clipped for -Rs.1,806.
    # Floor 80 happens to sit just under that dip. Reading that as evidence FOR 80 would
    # be fitting to one trade; read it instead as the warning it is — above the round
    # trip this guard starts taking money off winners, which is not its job.
    #
    # Two things measured and rejected here, recorded so they are not re-tried blind:
    #  - A max-peak ceiling (as the give-back cap has) changes NOTHING at any value.
    #    The guard fires early, while the running peak is still small; a ceiling on that
    #    peak cannot see the big winner coming.
    #  - A min-hold lowers the total at every floor (it delays protective exits and lets
    #    losses run). It does defuse the floor-100 cliff at >=45s, but only by paying
    #    more elsewhere than it saves. RE-MEASURED 2026-08-26 on the arm axis as well,
    #    because a min-hold is the obvious answer to "the rule fires 3 seconds after
    #    entry" -- and it is still wrong. Sweeping arm x min-hold over the 100 recorded
    #    lifecycles, versus the recorded outcome:
    #        arm      0s      15s      30s      45s      60s
    #        100  +5,164   +4,774   +4,413   +2,782   +2,999
    #        150  +3,038   +2,790   +2,558   +1,776   +1,697
    #    Every hold costs money at both arms. The 3-second firings are not a defect:
    #    that is a spike-and-fade being scratched at breakeven, and waiting is what
    #    turns the scratch into a loss.
    #
    # ARM 150 -> 100, measured 2026-08-26. pnl_path is GROSS (the recorded `pnl` field
    # is path_last - 80.00 in 81 of 107 lifecycles), so the floor of 80 IS net-zero and
    # an arm of 100 means "it was up Rs.20 net at some point, so it may not close red".
    #
    # Why 150 was leaving money: the largest losing cohort in the corpus is 33 trades
    # that dipped early, poked into profit at a MEDIAN PEAK OF Rs.139, and finished at
    # a median of -Rs.212 -- Rs.12,991 surrendered. A median peak of 139 sits just below
    # a 150 arm, so the guard whose entire job is "a proven trade may not close negative"
    # could not arm on the median member of the cohort it exists for.
    #
    # Sweep vs recorded, 100 deduped lifecycles over 10 sessions, net of Rs.80/trip:
    #        arm  0(off)     60      80     100     120     150     200     250
    #        net  +0     +3,190  +3,942  +5,164  +3,199  +3,038  +1,533  +1,157
    # A plateau across 80-120, not a spike, so this is not fitted to one number. 100 is
    # taken over the marginally better 80 because arm==floor makes "proven" mean nothing.
    #
    # It converts LOSSES INTO SCRATCHES rather than manufacturing winners, which is why
    # the headline win-rate falls while the money rises -- read the loss column:
    #        recorded        Rs. 5,454   win 34  scratch 12  loss 54   avg win 705
    #        arm 150         Rs. 8,492   win 30  scratch 27  loss 43   avg win 772
    #        arm 100         Rs.10,617   win 25  scratch 42  loss 33   avg win 874
    # Robustness vs arm 150: 7 of 10 sessions improve, median +Rs.343, and it is still
    # +Rs.1,736 with the single best session removed.
    #
    # The rule still cannot clip a winner at any arm: it fires only when the mark is at
    # or below the floor, so lowering the arm widens WHICH trades it defends, never how
    # much profit it takes off one.
    breakeven_stop_enabled: bool = True
    breakeven_stop_arm_rs: float = 100.0
    # The absolute arm remains the measured small-position floor.  On positions with
    # known stop geometry, require at least this fraction of initial risk as well; a
    # fixed Rs.100 was only 0.17R on the 2026-09-07 BANKNIFTY fill and treated a tiny
    # premium fluctuation as a proven trade.  0.40R matches the existing confirmed-trend
    # runner handoff: below it normal 5m movement keeps breathing room; at/above it the
    # runner trail can take ownership. Small legacy projections without geometry still
    # use the measured absolute floor.
    breakeven_stop_arm_r_multiple: float = 0.40
    breakeven_stop_floor_rs: float = 80.0
    breakeven_stop_min_age_seconds: float = 60.0

    # ── Peak trail, gated to LARGE peaks only (2026-08-19) ─────────────────────
    # Raises the breakeven stop's exit level from a flat floor to peak*(1-giveback),
    # but ONLY once the peak is big enough for a percentage to mean anything. Below
    # the gate a percentage is noise (40% of a Rs.300 peak is Rs.120, inside the
    # spread); above it, handing back 40% of a proven peak is worse than banking 60%.
    #
    # DEFAULT OFF, and the measurement is why. An ungated percentage trail is simply
    # the give-back cap again, and it fails the same way — over the same 60 recorded
    # paths, at arm 150 / floor 80:
    #     no trail                Rs.6,687   helped 15  hurt  4  worst -Rs.226
    #     trail 50% (ungated)     Rs.5,375   helped 16  hurt  8  worst -Rs.1,806
    #     trail 25% (ungated)     Rs.3,413   helped 18  hurt 14  worst -Rs.1,806
    # Tighter is monotonically worse. That is the operator's original complaint about
    # the cap — "exiting early on gaining trades" — reproducing exactly.
    #
    # Gated above Rs.600 of peak it stops clipping winners and turns mildly positive:
    #     trail 40% above Rs.600  Rs.6,986   helped 17  hurt  4  worst -Rs.226
    # Hurt count and worst case are UNCHANGED from no-trail; the whole difference is
    # two positions that kept more. Two positions out of 60, across 2 of 5 sessions,
    # is not enough to change a live default on, which is why this ships off.
    #
    # A note for whoever measures this next: evaluating a trail on a hand-picked set
    # of trades that ended NEGATIVE will always flatter it, because a trail cannot
    # lose money on a trade that already lost money. It has to be measured on the
    # whole book, winners included, or it will look like free money and is not.
    # The dirty worktree's ON/500 experiment is not a certified live default: only a
    # handful of path-covered positions changed and the report calls for a frozen,
    # basis-normalized held-out window first. Keep the conservative shipped behavior
    # until that validation is complete.
    peak_trail_enabled: bool = False
    peak_trail_above_rs: float = 600.0
    peak_trail_giveback_pct: float = 0.40

    # ── Let the BACKTEST see the live trade-management guards ───────────────────
    # The replay's exit simulator (_simulate_actual_option_exit) can only ever produce
    # SL HIT / TARGET_HIT / HISTORICAL / SESSION_END. It walks the option candles for a
    # stop or target and force-closes at the window end. Every live guard — profit lock,
    # peak-threat, loss-threat, early loss cut, breakeven stop — exists in scanner.py and
    # is NOT reached from that path. PEAK_GIVEBACK_CAP had never fired in any log for
    # exactly this reason: the fix was not in the code being measured.
    #
    # So a backtest cannot evaluate trade management at all, and two builds compared on
    # it are being compared on entry selection plus a fixed SL/target. That is worth
    # knowing before reading any such comparison as a verdict on exits.
    #
    # With this ON the simulator additionally evaluates the SHARED guard at each bar.
    # It CALLS Scanner._peak_giveback_cap_exit rather than reimplementing it — there is
    # deliberately no second copy of the rule to drift out of step with the live one.
    #
    # Default OFF: it changes backtest numbers, so it must be an explicit choice, and a
    # run with it on is not comparable to a run with it off.
    # Historical results should represent the management policy that will trade live.
    # The dashboard switch remains available to produce the frozen stop/target baseline.
    backtest_live_exits_enabled: bool = True
    # 2026-10-08 payoff audit: QP/S1 replays entered at the signal candle's OPEN (one
    # candle before the signal exists) and took the next opposite signal at its open.
    # Every QP replay profit sat inside that window. True = enter and exit only once
    # the candle has closed (as UTS always did). False only for comparison runs.
    replay_closed_candle_entries: bool = True
    # Replay S1 trades with the S1 lane's own premium risk, target and exit_decision
    # (index SL1/T1 lock, trend from closed S1 bars, profit lock, premium trail) instead
    # of the other engines' generic guards and a 2.5-3.8R target.
    replay_s1_native_exits: bool = True
    # Replay D1/D2 (s1s2_dual) trades with the lane's own exits -- the engine's stop and
    # EXIT events, the premium cap and the square-off -- instead of the generic guards.
    replay_s12_native_exits: bool = True

    # EARLY_LOSS_CUT guards (2026-08-16). Measured over 19 sessions: 18 cuts (ex the
    # two stale-mark trades already fixed) for -Rs.9,108. The exit EARNS its place —
    # on the 9 rows with a measurable stop it lost Rs.3,455 where the full stop would
    # have cost Rs.7,710, saving Rs.473/trade. So the R thresholds are deliberately
    # NOT touched here. What it lacked was the three guards every other major exit
    # already carries.
    #  - min hold: p10 was 2.0s and 5 of 18 fired under 15s. Kept SHORT on purpose —
    #    this exit's job is to be fast, so the 60s used by LOSS_THREAT_EXIT would
    #    disable it rather than guard it.
    #  - defer to lock: 7 of 18 had already peaked POSITIVE (to Rs.348) before being
    #    booked as an "early loss". A green trade giving back is a give-back, and the
    #    profit lock owns that case. It never got the chance: the two monitor paths
    #    disagreed on order — loss guard ran BEFORE the lock in one and AFTER it in
    #    the other. Deferring inside the decision fixes both paths at once.
    #  - cost floor: a loss smaller than the round trip cannot be worth a round trip.
    early_loss_cut_min_hold_seconds: float = 15.0

    # The rupee-loss-band guard was the last exit in the system with no clock at all —
    # its CATASTROPHE branch short-circuits ahead of every freefall and slow-bleed test,
    # so it fired the instant the rupee threshold was crossed. On 2026-08-18 that booked
    # -Rs.4,695 two and a half SECONDS after entry on a premium that had "fallen" 46.6%,
    # which was a stale entry mark rather than a move: the single largest loss in the
    # post-2026-08-12 book, 20.6% of that period's loser pot, from one trade.
    # Same 15s as the early loss cut, and for the same reason — this exit exists to be
    # fast, so LOSS_THREAT's 60s would disable it rather than guard it. The rupee
    # thresholds are untouched.
    rupee_loss_band_min_hold_seconds: float = 15.0
    early_loss_cut_defer_to_profit_lock_enabled: bool = True
    early_loss_cut_cost_floor_enabled: bool = True
    # Intel-accelerated loss-guard: when OI/PCR/volume/order-flow + price action
    # confirm a BIG reversal (not a small pullback), cut earlier than base 0.8R.
    # Small pullbacks (trend still supportive, no fast reverse) still wait for 0.8R.
    loss_guard_intel_accelerated_enabled: bool = True
    loss_guard_intel_min_r: float = 0.35   # severe multi-pillar reverse
    loss_guard_soft_intel_r: float = 0.55  # moderate reverse evidence
    loss_guard_big_reversal_severity: int = 5
    loss_guard_soft_reversal_severity: int = 3
    # Gap fixes: rejection wick, multi-bar grind, intel age, OI votes when losing,
    # option premium-melt reverse (without requiring index reverse).
    loss_guard_rejection_wick_enabled: bool = True
    loss_guard_rejection_wick_min_ratio: float = 0.45  # wick / full range
    loss_guard_grind_bars: int = 3
    loss_guard_grind_range_mult: float = 1.50  # cumulative adverse ≥ mult × median range
    loss_guard_intel_max_age_seconds: float = 90.0  # older intel is stale for acceleration
    loss_guard_oi_pcr_votes_winning: int = 3  # hard exit while profitable
    # The reversal exit reads PCR/OI/flow only, though the bundle also carries volume,
    # regime and a divergence alert. Corroboration from those may move the vote bar by
    # ONE, and only the safe way: a losing trade exits a vote sooner when they confirm,
    # a winning trade demands a vote more when they oppose. Neutral extras (the common
    # case) leave the decision byte-identical to the 3-pillar behaviour.
    intel_reversal_corroboration_enabled: bool = True
    intel_reversal_corroboration_votes: int = 2
    loss_guard_oi_pcr_votes_losing: int = 2   # faster OI/PCR exit when already red
    loss_guard_opt_premium_melt_pct: float = 8.0  # premium drop % → melt reverse signal
    loss_guard_opt_premium_melt_severity: int = 2
    # Severity stacking caps (R4): wick+grind+OI+PCR cannot all fully stack.
    loss_guard_severity_max_total: int = 8
    loss_guard_severity_max_price: int = 5
    loss_guard_severity_max_intel: int = 5
    # R5: when intel age is unknown, treat as not fully fresh for acceleration.
    loss_guard_unknown_intel_age_half_weight: bool = True
    # Post-2:59pm sudden-spike capture: a spike LEADS the 15m, so a power-move
    # with big range expansion + hard directional close may enter WITHOUT 15m
    # agreement (otherwise spikes are structurally un-catchable near close).
    power_move_spike_bypass_15m: bool = True
    # Measured 5-min range/ATR(14) over 7,068 bars, four indices:
    #   median 0.88   p90 1.50   p95 1.78   p99 3.03
    # so 1.8x is the p95 "unusual bar" and admits ~4.8% of bars to the power-move
    # gate. The spike tier that WAIVES 15-minute alignment is derived from it
    # (power_move_spike_threshold) rather than set beside it: 1.8 x 1.25 = 2.25,
    # around p97-98 at ~2.1% of bars, so the waiver stays meaningfully rarer than
    # the gate. Set the two equal — as .env did — and the waiver applies to every
    # power move.
    power_move_spike_range_atr_mult: float = 2.5
    #: Separation between the range gate and the spike tier. Adjustable, but
    #: clamped above 1.0 in code: the tiers meeting is the one thing it may not say.
    power_move_spike_range_margin: float = 1.25
    # 5min trend-CONTINUATION entry path: admit a high-grade, 15m-aligned 5min
    # signal in a confirmed trending regime with strong ADX, even if it isn't an
    # impulse reversal and option-premium data is incomplete. Fixes "system misses
    # big directional moves" (A+/95%/ADX48 trend signals were being rejected).
    live_5min_trend_continuation_enabled: bool = True
    live_5min_trend_continuation_adx: float = 25.0
    # Higher-conviction continuation tier: a strongly-confident, 15m-aligned 5min
    # signal in a trending regime may enter at a LOWER ADX, since conviction
    # substitutes for raw ADX. Catches grind trends that never spike ADX to 28
    # (e.g. NIFTY +80 / BANKNIFTY +300 on 2026-06-24, rejected at ADX 24-26 / B+ 69%).
    live_5min_trend_continuation_strong_conf: float = 0.65
    live_5min_trend_continuation_min_adx: float = 22.0
    # Regime: when ADX>=20 confirms a trend but the strict price>sma9>sma21 stack
    # breaks on a pullback within a grind, resolve direction via MA structure
    # (sma9>sma21 AND price>sma21, with macd_hist not reversing) instead of
    # defaulting to VOLATILE. Chop-safe: needs a stacked MA structure, so it cannot
    # fire when the SMAs are intertwined (true chop).
    regime_trend_soft_direction_enabled: bool = True
    # Regime smoothing: the detector computed a raw single-bar opinion and exposed it
    # raw, flipping 17-27x/day (a change every ~3 bars) — gates then saw a noisy regime.
    # Return a majority vote over the last N bars instead. ~50% fewer flips, stabler gates.
    regime_smoothing_enabled: bool = True
    regime_smoothing_window: int = 5
    # Regime hysteresis: ADX>=20 means trend strength IS present, so a bar with strong
    # ADX but ambiguous direction (NEUTRAL) is NOT choppy — hold the recent trend rather
    # than mislabeling it VOLATILE (which routes it to the choppy gates / rejects entries).
    regime_strong_adx_hold_trend: bool = True
    # Confirmed range-day guard: ADX can rise inside a 50-100pt box and manufacture
    # TRENDING/BREAKOUT labels even though the session is still rotating. Demote those
    # phantom directional regimes to RANGING unless the day has clear directional proof.
    regime_range_day_trend_guard_enabled: bool = True
    regime_range_day_trend_guard_min_confidence: float = 0.60
    # ADX-strength escape (2026-07-24 audit): a genuinely strong trend is real even
    # when the session day-type classifier calls the day a RANGE_DAY. Above this ADX
    # the range-day demotion is skipped so strong directional moves are NOT relabelled
    # RANGING (which had cascaded into choppy gating, scalp-lock early exits and dormant
    # trend re-entry — the system missed big moves). Set 0 to disable the escape.
    regime_range_day_trend_guard_adx_override: float = 35.0
    # Kaufman Efficiency Ratio early-trend promotion: ADX lags the first clean
    # directional drive. When ADX is still below the trend floor but price is moving
    # straight enough (ER) with a clean SMA stack, promote RANGING/VOLATILE to an
    # early trend. Confirmed RANGE_DAY context blocks this so wide swing-legs do not
    # manufacture phantom trends.
    regime_early_trend_er_promotion_enabled: bool = True
    regime_early_trend_er_period: int = 10
    regime_early_trend_er_threshold: float = 0.60
    regime_early_trend_er_adx_ceiling: float = 20.0

    # ── ER may now VETO a trend label, not only promote one (2026-08-19) ───────
    #
    # Until today the Efficiency Ratio could only ever make the regime MORE
    # directional: the promotion above fires when ADX is below its ceiling (20), so
    # above ADX 20 the ER was computed, logged, and then consulted by nothing. Every
    # regime decision keyed on ADX alone, and ADX is an EMA-smoothed memory of a move
    # that may already be over.
    #
    # 2026-08-19 is the clean demonstration. MIDCPNIFTY made a real move 09:15-10:30
    # (the session's only winner, +Rs.2,806). The tape then died: 10:30-13:41 held a
    # 41-point range with ~90 one-minute direction flips. ADX, still remembering the
    # morning, read 51-58 on the 5m. So:
    #     adx >= 20  -> TRENDING_UP
    #     adx >= 20  -> any 60-bar band touch becomes BREAKOUT_UP at confidence 70
    #     adx >= 35  -> the confirmed-range-day guard STEPS ASIDE (see below)
    #     adx >= 20  -> the ER promotion is skipped, so ER is never read at all
    # The engine entered MIDCPNIFTY 14850CE five times on that reading, at A+ and
    # 93-96% confidence, for -Rs.1,900. Measured ER at those five entries: 0.070,
    # 0.135, 0.097, 0.013, 0.201. An ER of 0.013 means price travelled 77x its net
    # displacement. Nothing in the stack was allowed to say so.
    #
    # Swept over every recorded lifecycle (312 positions, 25 sessions, 07-08..08-19),
    # blocking entries whose own-timeframe ER sat below the floor:
    #     floor 0.10   +Rs. 7,425     floor 0.25   +Rs. 7,231
    #     floor 0.15   +Rs.12,934     floor 0.30   +Rs. 6,118
    #     floor 0.20   +Rs. 8,653     floor 0.35   -Rs. 2,058
    # At 0.15: 14 sessions better, 7 worse, worst single session -Rs.2,084. The
    # blocked set is 64 trades with an 18/46 win/loss split against 105/143 for the
    # kept set — so this is removing a genuinely worse population, not a random slice.
    #
    # 0.15 is deliberately LOW. Winner and loser ER distributions overlap heavily
    # (median 0.285 vs 0.242), so ER is not a classifier and must not be used as one.
    # It is a TAIL filter: it identifies tape that is provably going nowhere. Raising
    # it toward the medians is where the measured edge dies — by 0.35 the sweep is
    # already negative.
    regime_er_veto_enabled: bool = False   # SAFETY-OFF 2026-08-19 14:15, see note below
    regime_er_veto_floor: float = 0.15
    # The range-day guard's ADX escape must clear the same bar. That escape exists so a
    # genuinely strong trend is not demoted on a range-day label — but it was keyed on
    # ADX alone, which is exactly the number that lies in this failure mode. On
    # 2026-08-19 ADX 51-58 disabled the one guard written for "ADX can rise inside a
    # 50-100pt box and manufacture TRENDING/BREAKOUT labels". Requiring ER as well
    # keeps the escape for real trends and closes it for remembered ones.
    regime_range_day_adx_override_needs_er: bool = False  # SAFETY-OFF with the veto above
    regime_early_trend_er_range_day_min_confidence: float = 0.60
    # Relax the VOLATILE 5min confidence floor (0.70 -> this) when the 15m trend
    # agrees with the signal — an aligned pullback-entry in a volatile-but-directional
    # tape is not the same risk as counter-trend chop. RANGING/UNKNOWN keep 0.70.
    live_5min_volatile_aligned_relax_enabled: bool = True
    live_5min_volatile_aligned_min_confidence: float = 0.65

    # ─── Directional census (counter-trend guard) ────────────
    # Track distinct directional signals across the session. When the session is
    # strongly skewed one way (ratio over min sample), require counter-direction
    # entries to clear an elevated confidence floor (B+ floor + penalty).
    directional_census_enabled: bool = True
    directional_census_ratio: float = 1.5
    directional_census_min_signals: int = 12
    # A confirmed trend day outranks the scanner's own signal census. The census counts
    # directions THIS SCANNER emitted and never looks at price, so it goes stale: on
    # 2026-08-24 every index fell (BANKNIFTY -582) and 53 shorts were generated against
    # 45 longs, but a long-heavy morning left it 18L/9S and shorts were refused as
    # "counter-bias" at 14:40/14:45/15:05/15:10 while session context simultaneously read
    # day_type=TREND_DAY_DOWN, bias=BEARISH at 0.92. Mirror of
    # session_trend_guard_census_yield, which handles the same clash the other way round.
    # Only TREND_DAY_UP/TREND_DAY_DOWN agreeing with the signal waives it, and only when
    # the label itself is confident. RANGE_DAY and UNDECIDED change nothing.
    directional_census_trend_day_yield_enabled: bool = True
    directional_census_trend_day_min_confidence: float = 0.60
    directional_census_penalty: float = 0.05

    # ─── Repaint intra-candle lockout ────────────────────────
    # Once a forming-candle signal repaints (vanishes) intra-candle, lock that
    # instrument+timeframe candle from being re-buffered/re-analysed until the
    # candle closes. If a live trade is exited by a repaint abort while the same
    # candle is still forming, re-check that same candle once per minute so a
    # fresh reappearing signal can go through the normal buffer/filter pipeline.
    # ── Stabilization: the system's largest filter (2026-08-11) ──────────────
    # 252 signals entered the 30s buffer that session and 48 matured — 19%. Every
    # rejection read "UT1 arrow vanished", 63 of them inside two seconds. All four
    # indices produced the correct SELL/PE signal within 22 seconds of the open, on the
    # only hour of the day that moved, and the buffer killed them.
    #
    # The latch: a one-way cascade is evidence about a move that ALREADY happened, so
    # re-proving it every scan asks the wrong question and goes false the moment volume
    # ticks under its ratio. Once confirmed it holds for the rest of the window. A
    # FLIPPED arrow still vetoes — that check runs first and is untouched.
    stabilization_cascade_latch_enabled: bool = True
    # Thrust scaling: a flat 30s on a 5-minute candle is 10% of its life, and the arrow
    # is least stable when price moves fastest. A thrusting bar drops to the floor.
    stabilization_thrust_scaling_enabled: bool = True
    stabilization_thrust_min_seconds: float = 10.0
    stabilization_thrust_range_atr: float = 1.50
    stabilization_thrust_volume_ratio: float = 1.50
    # Every buffered signal's outcome is recorded, not just the repaint rejections.
    # rtracker held 887 stabilization rejections across 47 sessions while a single
    # session's log held 203 — the largest gate in the system was effectively unmeasured.
    stabilization_rtracker_logging_enabled: bool = True
    live_repaint_candle_lockout: bool = True
    live_repaint_post_exit_recheck_seconds: float = 60.0

    # ─── Let winners run (anti-"pennies" profit-taking) ──────
    # The profit-lock was snatching tiny gains: it armed at a flat ₹500-1000 peak
    # and gave back on any dip under a flat ₹450, so trades were locked long before
    # reaching target/runner. With let_winners_run, the profit-lock does NOT arm
    # until peak >= profit_lock_min_r_multiple × initial risk (₹) — so every trade
    # gets room to develop to at least that R-multiple before locks engage. The
    # flat give-back floor is also made tunable. Runner mode unlocks earlier so
    # winners flip to trail-and-ride sooner.
    let_winners_run: bool = True
    # 2026-08-11 (rtracker, 12 sessions / 106 trades): 1.0 measured as the reason the
    # non-scalp book gave everything back. The system reached Rs.42,444 of peak profit
    # and banked -Rs.992; 45 trades went green then closed red, 42 of them peaking below
    # every arming floor. Median trade peaks at Rs.274 while the median non-scalp arm
    # computed to Rs.868 -- only 4 of 59 non-scalp trades ever reached their own lock.
    #
    # The Rs.500 base was NOT what bound: it beat risk*1.0 on just 5 of those 59 rows.
    # This R-multiple bound the other 54, so lowering the base alone would have been
    # very nearly a no-op (2 of the 45 give-backs would have armed, against 20 for the
    # pair below).
    #
    # 0.20 mirrors fifteen_min_choppy_scalp_profit_lock_r_multiple, whose cohort is the
    # live control group: scalp_lock trades ran that arm and captured +42.6% of peak
    # against -4.5% for the rest, at a 48.9% vs 20.3% win rate (z=3.11, p=0.002), and
    # the effect holds inside RANGING and outside it. The "let winners run" worry -- that
    # arming early truncates the fat tail -- did not appear in that cohort: it carried
    # the SAME eight peak>=Rs.750 winners as the loose cohort and a HIGHER best peak
    # (Rs.3,582 vs Rs.2,603). Arming is not exiting; the ladder still trails.
    profit_lock_min_r_multiple: float = 0.20
    # True scalp rows still need an R-aware arm; otherwise a Rs.500 blip can cap
    # winners while the configured stop risk is several thousand rupees.
    scalp_profit_lock_min_r_multiple: float = 0.50
    # Absolute arming floors, previously a bare 500.0 literal inside
    # Scanner._profit_lock_min_peak with no way to reach it from the dashboard. Split in
    # two because the branches want different answers: scalp keeps its historical base,
    # while the non-scalp base drops to the same Rs.250 the choppy-scalp path already
    # uses in production (and which .env has pinned at 250.0 since before this change).
    profit_lock_min_peak_rs: float = 500.0
    profit_lock_nonscalp_min_peak_rs: float = 250.0
    profit_lock_abs_floor_rs: float = 450.0
    # The give-back floor as a share of what the position actually cost, because a flat
    # rupee line means six different things across four indices. SENSEX trades 20 to a
    # lot and MIDCPNIFTY 120, so Rs.500 is 25 points of premium on the first and barely
    # 4 on the second — the same instruction asking one index to sit through six times
    # the swing of another, on a system whose entries come from an ATR flip that needs
    # room to breathe.
    #
    # 5% of position cost reproduces both reference points independently: SENSEX at a
    # ~500 premium gives Rs.500 (25 points, the number that has been working), and
    # MIDCPNIFTY at a ~170 premium gives Rs.1,020 (the ~Rs.1,000 the operator asked
    # for). NIFTY and BANKNIFTY land at Rs.390 and Rs.375, below profit_lock_abs_floor_rs,
    # which then does its original job — a cost-coverage minimum, since below roughly
    # Rs.450 the round trip is mostly charges.
    #
    # The effective floor is max(cost x pct, round_trip x cost_multiple): at least this
    # share of what was put at risk, and at least enough to clear the real round trip.
    # Set to 0 to disable and fall back to the flat floor alone.
    profit_lock_cost_pct: float = 5.0
    # Cost coverage, derived rather than assumed. profit_lock_abs_floor_rs was carrying
    # this job at a flat Rs.450 while the actual round trip on these four contracts is
    # Rs.125-202 — so it was never a cost floor, just a line that happened to bind on the
    # two mid-sized ones. That handed NIFTY 5.77% of position cost and BANKNIFTY 6.00%
    # while SENSEX got 4.50% and MIDCPNIFTY 2.21%: the same unevenness the percentage
    # above exists to remove, surviving in the place it was least visible.
    #
    # At 2x the round trip the coverage floor lands at Rs.250-404, below the 5% figure on
    # every one of the four, so the proportional rule governs uniformly and this binds
    # only where a position really is too small to be worth locking.
    # profit_lock_abs_floor_rs remains the fallback for a row that cannot be priced or
    # sized — never a floor of zero.
    profit_lock_min_cost_multiple: float = 2.0
    # The LOWEST locks — the rungs a lock arms on — scaled the same way, because this is
    # where a flat rupee number bites hardest. The non-scalp base of Rs.250 is 12.5 points
    # of premium on a 20-lot SENSEX and 2.1 points on a 120-lot MIDCPNIFTY. Two points is
    # inside the noise of an ATR flip: MIDCPNIFTY was arming a lock before its entry had
    # room to prove anything, while SENSEX got six times the room from the same setting.
    #
    # Anchored on SENSEX, whose behaviour today is the reference: on a ~Rs.10,000 position
    # Rs.250 is 2.5% and Rs.500 is 5%. Every index now arms at the same share of what it
    # put at risk, in points of premium:
    #
    #     arm (non-scalp, 2.5%)      NIFTY 3.0   BANKNIFTY  6.25   SENSEX 12.5   MIDCP 4.25
    #     arm (scalp, 5.0%)          NIFTY 6.0   BANKNIFTY 12.50   SENSEX 25.0   MIDCP 8.50
    #
    # The R-multiple above still governs whenever risk x r_mult is larger; these set the
    # floor under it. Set either to 0 to return to the flat rupee base.
    profit_lock_arm_cost_pct: float = 2.5
    profit_lock_arm_scalp_cost_pct: float = 5.0
    # The sub-Rs.1,000 rung retains 50% of peak, i.e. it stops a trade that gives back
    # half. Measured heat as a fraction of peak across every winner in 47 sessions:
    # TRENDING_UP 8%, TRENDING_DOWN 28%, RANGING 35%, BREAKOUT_DOWN 43%, BREAKOUT_UP
    # 47%, VOLATILE 49%. In the fast regimes that floor sits ON the pullback those
    # winners routinely make and survive. 0.38 = 62% breathing, clear of the measured
    # median. Applied through a min(), so it can only loosen; TRENDING_UP and RANGING
    # are untouched because their heat is nowhere near the floor.
    first_rung_regime_retention_enabled: bool = True
    first_rung_fast_regime_retention: float = 0.38
    # A closed trade has only ever recorded peak_pnl and max_drawdown, and those cannot
    # answer "would a tighter trail have kept more?" -- a trail exits at f x peak only if
    # P&L reached that level AFTER the peak, and a max carries no ordering. Asked of the
    # 2026-08-11 book the honest answer was "between -Rs.18,114 and +Rs.10,988", which is
    # no answer at all. Sampling the path makes the next such question measurable rather
    # than arguable. Bounded three ways in _sample_pnl_path; at a 1.2-minute median hold
    # a typical trade stores ~36 samples.
    # Ceiling on the Fyers index-history call, which runs on the scanner's event loop
    # and is the ONLY source of index volume. The SDK takes no timeout and its transport
    # has no default, so a degraded upstream blocks the loop outright: on 2026-08-31 at
    # the open it produced 11s / 25s / 45s stalls, dropped the scan cycle from ~740ms to
    # roughly one per four minutes, and cost 92 dashboard timeouts in nine minutes.
    #
    # 8s is deliberately generous against a healthy call (sub-second) -- the aim is to
    # bound a pathological one, not to trim a slow one, because every miss falls through
    # to a source that lacks index volume.
    # Cut a trade that has never printed a positive P&L once it is this old.
    #
    # 72 of 204 losing trades in the 61-session rtracker archive never ticked green,
    # and 0 of 109 winners ever did -- a 72/72 loser with no counterexample, losing
    # Rs 653 on average against Rs 230 for losers that at least went green first.
    # The R-multiple loss guard has no time term, so these were held a median of 46
    # minutes. Replayed tighten-side over 118 exits with a pnl_path: +Rs 3,577, with a
    # session-clustered bootstrap 95% CI of +Rs 1,537 .. +Rs 5,731.
    #
    # 300s sits in the middle of a broad optimum (3-10 minutes all gain +2,840 or more),
    # so it is not perched on a knife edge.
    # One quality policy for live/recovery/historical/EOD.  A >=70% setup may stand on
    # its own; B+ (55--70%) is *eligible*, not automatically admitted, and must clear
    # the explicit evidence routes below.  The old tracker has sparse outcomes and mixed
    # accounting, so it is not evidence for a lower unconditional live floor.
    live_5min_quality_min_confidence: float = 0.70
    quality_policy_bplus_min_confidence: float = 0.55
    quality_policy_bplus_min_adx: float = 22.0
    quality_policy_risky_bplus_min_adx: float = 25.0
    # Risky-regime (VOLATILE/RANGING/...) B+ needs 15m trend plus an INDEPENDENT leg.
    # False (2026-10-07): two agreeing engines no longer count as that leg, because a
    # quad-union entry is built from exactly that agreement. True restores the old rule.
    quality_policy_risky_bplus_count_multi_source: bool = False
    # v10 (2026-10-08 audit AC-02): 'multi_source' no longer counts as a confirmation.
    # 'shadow' enforces v9 and records the v10 result; 'enforce' uses v10; 'legacy' off.
    quality_policy_multi_source_mode: str = "shadow"
    # Tracker exit P&L at executable prices: buy at the acceptance ask, sell at the mark less
    # the index half-spread, plus the statutory opt_cost (2026-10-08 audit PA8). Recording only;
    # management still runs on marks.
    tracker_executable_pricing_enabled: bool = True
    # Manual/session rows: price a stop exit at the first recorded contract tick through the
    # stop since the previous pass, not the later scan-cycle mark (2026-10-08 audit).
    session_tick_stop_check_enabled: bool = True
    # Telemetry: flag a broker fill paired with an acceptance more than this many seconds
    # away (e.g. a MANUAL_SYNC pairing hours later) so fill statistics can exclude it.
    fill_pairing_suspect_seconds: float = 2700.0
    # Regime/trend market-context gate for every lane and timeframe (2026-10-08 user;
    # engine/entry_context_gate.py). 'shadow' records what it would refuse without
    # refusing; 'enforce' refuses; 'off'. The regime table starts in shadow because the
    # 20-session audit found no single filter with a robust edge. The MIDCPNIFTY 15m
    # peer rule (choppy/range-bound context needs >= 2 of the other 3 indices moving the
    # same way, from closed bars) is enforced as the user asked.
    ctx_gate_mode: str = "shadow"
    ctx_peer_alignment_mode: str = "enforce"
    ctx_peer_alignment_required_indices: str = "MIDCPNIFTY"
    ctx_peer_alignment_timeframes: str = "15min"
    ctx_peer_alignment_min: int = 2
    ctx_trend_min_adx: float = 18.0
    ctx_trend_rsi_band: float = 2.0
    ctx_breakout_min_adx: float = 20.0
    ctx_breakout_min_volume: float = 1.0
    ctx_range_min_adx: float = 22.0
    ctx_range_rsi_band: float = 5.0
    ctx_range_min_peers: int = 2
    ctx_volatile_min_adx: float = 20.0
    ctx_volatile_min_peers: int = 2
    ctx_unknown_min_adx: float = 20.0
    ctx_min_atr_pct_1min: float = 0.025
    # Opening volume (2026-10-09 user). Fyers cash-index volume reaches a bar only with
    # the next history poll, so at the open the signal bar is still unmeasured when the
    # gate decides. When that happens, the closed bar of the near-month Kite future,
    # against its own previous bars, counts as measured volume (data/futures_bar_volume.py).
    volume_futures_fallback_enabled: bool = True
    volume_futures_fallback_timeout_seconds: float = 2.0
    volume_futures_fallback_baseline_bars: int = 20
    volume_futures_fallback_min_baseline_bars: int = 10
    # A Fyers cash total older than this marks the feed degraded (health and alerts).
    # 120s flapped every morning against the ~2 minute history poll.
    volume_cash_freshness_seconds: float = 180.0
    # Special entries (A+, LTP shock, volume spike) need fresh 1-minute volume. While
    # the Fyers cash minutes are missing, the Kite futures minute series stands in
    # (engine/entry_data_quality.py).
    entry_data_quality_futures_fallback_enabled: bool = True
    # Engine lab (comparison-only, never orders) refresh during 09:15-15:30. Its
    # full-CPU passes overlapped live scan timeouts on 2026-10-09.
    engine_lab_market_refresh_seconds: float = 300.0
    # Shadow only (2026-10-09): log when a trend-day QP peak between the give-back arm and
    # the 1R smart lock fades past half (data_store/trend_gap_shadow.jsonl).
    trend_gap_shadow_enabled: bool = True
    # Early-entry build-up detector, shadow only (2026-10-09, user; engine/buildup_shadow.py).
    # Logs when a futures price impulse comes with volume, OI and/or PCR building, to
    # data_store/buildup_shadow.jsonl; scripts/evaluate_buildup_shadow.py judges it.
    buildup_shadow_enabled: bool = True
    buildup_window_seconds: float = 180.0
    buildup_impulse_min_pct: float = 0.08
    buildup_impulse_baseline_mult: float = 2.0
    buildup_baseline_minutes: float = 60.0
    buildup_min_baseline_windows: int = 5
    buildup_volume_ratio: float = 1.5
    buildup_oi_window_seconds: float = 300.0
    buildup_oi_min_pct: float = 0.05
    buildup_pcr_window_seconds: float = 300.0
    buildup_pcr_max_age_seconds: float = 240.0
    buildup_pcr_min_change: float = 0.02
    buildup_min_legs: int = 2
    buildup_cooldown_seconds: float = 600.0
    # Gap-open shadow rule (2026-10-09 user): after a gap of at least this size that
    # holds, with this many of the four indices gapped and holding the same way, the
    # prior session's swing structure is stale for the first minutes. Shadow only:
    # verdicts are recorded on candidates and logged, nothing is admitted or refused.
    gap_open_shadow_enabled: bool = True
    gap_open_min_gap_pct: float = 0.8
    gap_open_min_aligned: int = 3
    gap_open_window_minutes: float = 5.0
    ctx_min_atr_pct_5min: float = 0.05
    ctx_min_atr_pct_15min: float = 0.09
    quality_policy_min_volume_ratio: float = 1.20

    # Deprecated compatibility input.  The versioned evaluator intentionally ignores
    # grade as an independent switch because grade is derived from confidence; retaining
    # this field avoids breaking saved settings while removing its non-monotonic effect.

    # Where today's 4-index warmup record is kept so a mid-session restart does not
    # erase it. The auto-execution gate reads this when its in-memory copy is empty;
    # the session_day inside the file is what stops yesterday's record arming today.
    morning_refresh_state_file: str = "data_store/morning_refresh_summary.json"

    loss_guard_never_green_cut_enabled: bool = True
    loss_guard_never_green_cut_seconds: float = 300.0

    fyers_history_timeout_seconds: float = 8.0

    pnl_path_logging_enabled: bool = True
    pnl_path_min_interval_seconds: float = 2.0
    pnl_path_min_move_rs: float = 50.0
    pnl_path_max_samples: int = 400

    # ── Intel path: the same idea as pnl_path, for the MARKET rather than the P&L ──
    # decision_intel already freezes pcr / oi_signal / adx / atr_pct / volume_ratio /
    # close_above_vwap_pct / kaufman_er and the rest at ONE instant, at exit. One instant
    # cannot answer the only question worth asking of it -- "did the case change while we
    # held?" -- so no proposed exit rule keyed on those values can be replayed before it
    # ships. That is why the 2026-08-25 audit of a market-aware exit engine had to end in
    # "unmeasurable" rather than yes or no.
    #
    # Sized from what was measured, not guessed. PCR only CHANGES every ~99s per
    # instrument (median over 21 sessions, n=3,067; SENSEX 188s) while the median hold is
    # ~103s, so a 5s timer oversamples the fastest-moving chain value by 20x and the whole
    # series still costs almost nothing -- because a sample is only APPENDED when the
    # snapshot actually differs from the last one kept. The typical trade therefore stores
    # 2-3 rows, and those rows are exactly the transitions.
    #
    # Observation only. Nothing reads this to make a decision, and a failure to record it
    # must never be able to touch a live position.
    intel_path_logging_enabled: bool = True
    intel_path_min_interval_seconds: float = 5.0
    intel_path_max_samples: int = 60
    # ── Sub-1R give-back guard (Fix 1) ──────────────────────────────
    # Below the full 1R profit-lock arm, a modest peak used to be unprotected and
    # could round-trip into a loss (e.g. a +0.4R peak fading past breakeven). Once
    # a trade shows a meaningful positive peak, don't let that winner become a
    # loser: arm at max(min_peak_rs, arm_r_multiple × risk) and exit at the current
    # (still-positive) price once profit gives back to <= exit_ratio × peak.
    profit_giveback_guard_enabled: bool = True
    profit_giveback_arm_r_multiple: float = 0.40
    profit_giveback_min_peak_rs: float = 500.0
    profit_giveback_exit_ratio: float = 0.50
    # No protective exit may arm below what a round trip costs. 2026-08-11 15:11:
    # PROFIT_GIVEBACK_GUARD closed BANKNIFTY 57400PE to protect a Rs.36 peak and booked
    # -Rs.44, because opening and closing cost Rs.80. Inside the EOD window the arm
    # collapses to eod_protect_min_peak_rs (Rs.1), so any blip armed an exit that could
    # not come out ahead. The LOSS side of peak-threat has refused on this reasoning
    # since it was written (peak_threat_exit_loss_min_cost_multiple); the profit side
    # never had the equivalent. Raises an arm only — it can never lower one.
    profit_protect_cost_floor_enabled: bool = True
    profit_protect_min_cost_multiple: float = 1.0
    # ── Volatility + regime scaling of the profit-side floors (2026-08-11) ────
    # Every profit-side threshold is fixed rupees (Rs.750 peak-threat, Rs.500/250 lock,
    # Rs.500 give-back, ladder at 1k/2k/3k) while the moves they catch scale with
    # volatility. As the median daily range fell 0.97% -> 0.64% across W30-W33, the share
    # of trades ever reaching a Rs.750 peak fell 43% -> 4%; PEAK_THREAT_EXIT, which banks
    # 87% of peak, fired ONCE in a week, and LOSS_THREAT_EXIT went from 30% to 50% of all
    # exits. Win rate followed, 47.8% -> 12.5%.
    #
    # Only the ABSOLUTE floors are scaled — each arm is max(floor, risk x r_multiple) and
    # the r_multiple half already scales, since initial_risk_rs is an ATR-derived stop
    # times quantity. The max() is exactly what lets the blind floor dominate when risk
    # is small, i.e. in the quiet markets where it does the damage.
    #
    # Volatility term: the trade's own risk against this reference (median observed risk
    # is Rs.937). Regime term: measured peak potential, so a trend develops before the
    # lock arms and chop banks sooner. Bounded hard both ways.
    # ── Entry-time intel snapshot (2026-08-12) ───────────────────────────────
    # OBSERVATION ONLY. Nothing reads this to make a decision.
    #
    # The engine computes volume ratio, PCR, cumulative OI, order flow, breadth, VWAP
    # position, the full greeks and a VIX percentile — and the trade record kept almost
    # none of it: volume_ratio survived on 2% of 239 accepted rows, the grading pillars
    # on 33%, PCR/OI/order_flow on none. intelligence_history.json does hold the series
    # per minute, but as a rolling 5,000-point window only 76 of 240 trades were still
    # joinable to it: the evidence expires before anyone can use it.
    #
    # A composite ENTRY GATE built on these features was measured and REJECTED. It looked
    # excellent in-sample (55.6% win rate / Rs.752 at K>=3) and inverted out-of-sample
    # (16.7% / -Rs.336 against a 32.5% baseline), and the adaptive rolling-median variant
    # beat its own baseline win rate in no configuration. This snapshot exists so that
    # question can be re-asked later against complete, permanent data instead of a
    # 6-feature shadow of what the engine actually knows.
    entry_intel_snapshot_enabled: bool = True
    profit_threshold_scaling_enabled: bool = True
    profit_threshold_regime_scaling_enabled: bool = True
    profit_threshold_reference_risk_rs: float = 1000.0
    profit_threshold_scale_min: float = 0.35
    # Capped at 1.0 deliberately: the scale exists to RESTORE PEAK_THREAT_EXIT where the
    # fixed floor has gone unreachable, and letting it exceed 1.0 in trending regimes
    # suppressed the very exit it was meant to rescue — measured as an exact wash, 17
    # trades reaching the arm before and after. Regime still differentiates below 1.0
    # (RANGING 0.85 vs TRENDING_UP 1.00), it just cannot push the arm further away.
    profit_threshold_scale_max: float = 1.00
    # ── Entry budget scaled to the day's measured range ──────────────────────
    # Trades per 1% of median daily range across the stored weeks: 15.5, 13.8, 16.9,
    # 18.8 — activity rose against opportunity as the market quietened, and the median
    # peak fell Rs.533 -> Rs.78 over the same stretch. 0.90% is the reference day;
    # quieter days scale the per-index budget down, never up, and never below the floor.
    range_scaled_budget_enabled: bool = True
    range_scaled_budget_reference_pct: float = 0.90
    range_scaled_budget_min_trades: int = 3
    # Confirmed-live-trend "let winners run" (2026-07-24 audit): a with-trend PROVEN
    # winner (peak >= min_peak_r_multiple x initial risk) in a strong-trend regime
    # rides the runner lock (10% give-back) instead of the tighter give-back guard,
    # even when the session day-type classifier lags or scalp-lock was armed. EOD and
    # unconfirmed-repaint protection are NEVER suppressed. Set enabled False to revert.
    trend_runner_let_winners_run_enabled: bool = True
    # 0.4R crossover (operator design 2026-07-24): BELOW this the current profit/loss
    # guards run unchanged (give-back arms at ~0.40R); AT/ABOVE it, in a confirmed strong
    # trend only, the trade hands off from the 50%-fade give-back to the 85-90% runner
    # lock (protects more AND lets it run). The regime+ADX gate keeps this from engaging
    # on chop days, so it is regime-adaptive rather than a blanket low threshold.
    trend_runner_min_peak_r_multiple: float = 0.4
    # Dynamic runner profit-lock: give a trending premium more room in fast/volatile
    # tape (lock the looser high-vol pct so noise does not shake the runner out) and
    # tighten toward dynamic_rr_runner_lock_pct (currently 0.85) in calm regimes.
    trend_runner_dynamic_lock_enabled: bool = True
    # Keep the original 5-point calm/high-vol separation after the base lock moved
    # from 0.90 to 0.85. 0.80 locks 80% of peak and gives confirmed volatile runners
    # 20% retracement room; calm runners retain the tighter 85% lock.
    trend_runner_lock_pct_high_vol: float = 0.80
    # Big-move gate: the ride/runner engages EARLY (before the fixed target) only when a
    # proven winner is in a strong trend with ADX above this floor — so small gains are
    # banked normally and the ride is reserved for moves with room to run.
    trend_runner_big_move_min_adx: float = 25.0
    # Runner mode requires a regime for the trade's OWN timeframe. _get_trade_regime
    # falls back to the 5-minute label, which is fine for tightening but must not GRANT
    # the looser runner trail to a 15-minute trade on a 5-minute reading. Set False to
    # restore the previous borrow-and-assume behaviour.
    trend_runner_require_own_timeframe_regime: bool = True
    # A HISTORICAL replay writes the regime it computed from PAST candles into the same
    # intelligence caches live management reads, stamped with the current wall clock — so
    # without this purge the replayed label is served as fresh live intel for a full
    # intelligence_cache_ttl_seconds after switching back to REAL. Set False to restore
    # the old carry-over behaviour.
    purge_intel_cache_on_mode_change: bool = True
    trend_runner_engage_before_target_enabled: bool = True
    # ── Pullback hedge (Phase 3, operator design 2026-07-24) ──────────────────
    # While a proven runner (>= the 0.4R crossover) is riding a confirmed trend, an
    # early pullback arms a FRESH-ATM opposite option leg ("pullback insurance"): it
    # captures the down-move that the runner lock would otherwise give back, and is
    # RELEASED when the move resumes so the runner keeps riding. The hedge is never a
    # standalone directional bet — it closes with its parent and never outlives it.
    # DEFAULT OFF: this opens a second live broker leg (sizing/exposure/accounting),
    # so it stays off until the archived-session backtest justifies enabling it.
    #
    # DORMANT AS OF 2026-08-07, and the distinction matters because this is a toggle in
    # the dashboard UI. _pullback_hedge_arm and _pullback_hedge_release are written and
    # tested, but NOTHING IN THE SYSTEM OPENS A HEDGE LEG — there is no call site that
    # places the second order. Turning this on therefore changes nothing at all.
    #
    # The decision helpers are kept rather than deleted: they encode real thinking about
    # when a hedge is justified (only a proven runner, armed before the lock's give-back)
    # and that work is worth having when the leg is eventually built. What is NOT
    # acceptable is a switch an operator can flip believing it does something, so the
    # REAL-mode preflight warns when this is on. See engine/safe_mode.py.
    pullback_hedge_enabled: bool = False
    # Arm when the premium has retraced this FRACTION of the distance from peak to the
    # runner stop. Relative (not a fixed premium %) because the runner lock sits only
    # ~4-5% below peak: a fixed 5% trigger could never fire — the parent always exited
    # first (proven by the archived-session backtest, 0 hedges armed).
    pullback_hedge_trigger_lock_fraction: float = 0.5
    pullback_hedge_trigger_giveback_pct: float = 0.0   # optional extra floor (0 = off)
    pullback_hedge_resume_recovery_pct: float = 50.0   # % of pullback depth recovered -> release
    pullback_hedge_size_fraction: float = 0.5          # hedge lots vs parent lots
    pullback_hedge_max_per_trade: int = 2              # anti-whipsaw churn cap
    pullback_hedge_max_hold_minutes: float = 15.0      # never hold a hedge indefinitely
    # Option-chain clocks have different jobs. Network fetch stays API-safe at 90s;
    # startup tolerates the four-index warmup; fresh entries use a tighter bound; open
    # trade management may retain older context while live contract ticks own price risk.
    option_chain_startup_max_age_seconds: float = 150.0
    option_chain_entry_max_age_seconds: float = 120.0
    option_chain_high_vol_entry_max_age_seconds: float = 90.0
    option_chain_high_vol_percentile: float = 70.0
    # The background chain refresh shares the 90s high-volatility limit as its TTL, so
    # scan latency alone aged chains past it at decision time (2026-10-09: MIDCPNIFTY
    # refused at 96s and 101s). Fetch once on demand before refusing an entry for age.
    option_chain_entry_refresh_on_stale: bool = True
    option_chain_entry_refresh_cooldown_seconds: float = 15.0
    option_chain_intelligence_max_age_seconds: float = 300.0
    # ── Stop-out re-entry (2026-07-25) ───────────────────────────────────────
    # A trade stopped out early on a trend day, then the move ran without us. When the
    # SAME direction re-validates in a confirmed trend shortly after a STOP-type exit,
    # allow one more attempt by bypassing ONLY the post-exit cooldown. The daily trade
    # budget and the consecutive-loss halt are real risk limits and are NEVER bypassed.
    # Cost-correct replay across the current archive (2026-08-30): 33 replayable
    # retries, 64% wins and +Rs 2,519 gross, but Rs 4,952 modeled round-trip
    # costs/slippage -> -Rs 2,433 net.  Narrow 8-10 minute sensitivity was only
    # marginally positive and failed again by 12 minutes, so it is not robust enough
    # to ship enabled.  Keep the operator A/B toggle, but fail closed by default until
    # executable shadow outcomes demonstrate positive after-cost expectancy.
    stop_out_reentry_enabled: bool = False
    stop_out_reentry_window_minutes: float = 30.0   # same move, not a new session phase
    stop_out_reentry_max_per_index: int = 1         # one second chance per index per day
    stop_out_reentry_min_adx: float = 25.0          # only a genuinely strong trend
    # ── Rupee loss ladder (freefall OR confirmed slow bleed) ───────
    # Eligibility floors OPT -1000 / FUT -1500 are NOT auto-exits. Two cut paths:
    # (A) FREEFALL — sudden opposite flow, post-close repaint, spike dump, etc.
    # (B) SLOW BLEED — easier to detect: past arm for N minutes, loss keeps
    #     making new lows / never recovers, other guards missed the grind.
    rupee_loss_band_enabled: bool = True
    rupee_loss_band_opt_only: bool = False
    # Arms/steps/force levels are PER LOT (2026-07-10): a 2-lot book arms at
    # 2× the rupee thresholds so bigger size is not cut on 1-lot noise.
    rupee_loss_band_scale_per_lot: bool = True
    # MIDCPNIFTY carries the largest lot size (120 vs NIFTY 65 / BANKNIFTY 30 /
    # SENSEX 20) — relax its rupee thresholds by this % per lot.
    rupee_loss_band_midcap_relax_pct: float = 20.0
    rupee_loss_band_start_rs: float = 1000.0
    rupee_loss_band_start_rs_opt: float = 1000.0
    rupee_loss_band_start_rs_fut: float = 1500.0
    rupee_loss_band_size_rs: float = 500.0
    rupee_loss_band_size_rs_opt: float = 500.0
    rupee_loss_band_size_rs_fut: float = 500.0
    # Path A: freefall special occasions
    rupee_loss_band_freefall_enabled: bool = True
    rupee_loss_band_min_severity: int = 4
    rupee_loss_band_min_severity_opt: int = 4
    rupee_loss_band_min_severity_fut: int = 4
    rupee_loss_band_min_freefall_flags: int = 2
    # Path B: slow bleed (monotonic grind past arm — easy detector)
    rupee_loss_band_slow_bleed_enabled: bool = True
    rupee_loss_band_slow_bleed_minutes: float = 4.0       # past arm this long
    rupee_loss_band_slow_bleed_worsen_rs: float = 150.0   # new lows beyond first arm touch
    rupee_loss_band_slow_bleed_max_recovery_rs: float = 80.0  # no meaningful bounce
    # Absolute catastrophe force — last resort only.
    rupee_loss_band_force_loss_rs_opt: float = 3000.0
    rupee_loss_band_force_loss_rs_fut: float = 4000.0
    # Spike freefall: adverse PnL velocity Rs/min after entry.
    rupee_loss_band_spike_rs_per_min: float = 400.0
    # Legacy alias (maps to freefall path when true; slow bleed still available).
    rupee_loss_band_freefall_only: bool = False
    # Special-case only: the give-back guard fires ONLY in a protective context
    # (near square-off, or a scalp-lock / late entry). A normal mid-day trade keeps
    # let-winners-run below the 1R arm. Set False for the blanket (every-trade) guard.
    profit_giveback_guard_special_only: bool = True
    # An intracandle entry whose repaint check has not yet passed is structurally
    # unconfirmed — let-winners-run does not apply, so the give-back guard also arms
    # during that window (exit reason REPAINT_GIVEBACK_GUARD). 2026-07-02: a Rs.990
    # peak faded negative before its repaint abort fired.
    repaint_window_giveback_guard_enabled: bool = True
    # ── EOD-proximity profit protection (Fix 2) ─────────────────────
    # Within this many minutes of the square-off, a trade physically cannot run to
    # 1R, so collapse the profit-lock arm toward a small peak so protection engages
    # early instead of letting a winner round-trip until the forced EOD kill.
    eod_protect_enabled: bool = True
    # 14 minutes, set FROM the session schedule rather than independently of it:
    # ut_force_exit_time (15:15) minus this is 15:01, which is exactly
    # ut_5min_no_entry_after. Protection therefore begins the moment the last entry is
    # allowed, so no position can ever be opened inside the window that closes it. At
    # the previous 30 it began at 14:45 and overlapped the entry gate by 20 minutes.
    # Changing either time without the other re-opens that overlap.
    eod_protect_window_minutes: float = 14.0
    # 2026-09-26 user: every exit is rechecked by 15:15:30 -- one broker reconcile and a
    # critical alert naming anything still open this many seconds after the force exit.
    eod_exit_recheck_seconds: float = 30.0
    eod_protect_arm_r_multiple: float = 0.0
    # Floored at the round-trip cost by _profit_protect_cost_floor_rs — on its own this
    # Rs.1 armed protective exits that banked less than they paid to close.
    eod_protect_min_peak_rs: float = 1.0
    # ── EOD-proximity loss minimizer ───────────────────────────────
    # During the last 15-20 minutes, a losing trade should not casually drift into
    # force-exit/full-stop. Track its best reduced-loss mark and exit if it reaches
    # near-breakeven, starts worsening again, or has only a few minutes left.
    eod_loss_minimize_enabled: bool = True
    eod_loss_minimize_window_minutes: float = 20.0
    eod_loss_minimize_hard_exit_minutes: float = 5.0
    eod_loss_minimize_max_hard_exit_minutes: float = 10.0
    eod_loss_minimize_min_age_minutes: float = 3.0
    eod_loss_minimize_near_breakeven_rs: float = 250.0
    eod_loss_minimize_giveback_rs: float = 250.0
    eod_loss_minimize_giveback_r_multiple: float = 0.12
    eod_loss_minimize_material_loss_r: float = 0.45
    eod_loss_minimize_severe_loss_r: float = 0.75
    # Former-winner recovery watch: when a trade has already shown profit, then
    # slips slightly negative while still inside normal option breathing and the
    # candle/trend context remains supportive, monitor it with a reduced-loss
    # virtual trail instead of passively waiting for the full hard stop.
    loss_recovery_watch_enabled: bool = True
    loss_recovery_watch_min_peak_rs: float = 500.0
    loss_recovery_watch_min_peak_r: float = 0.25
    loss_recovery_watch_max_loss_r: float = 0.45
    # Rupee breathing for former winners: if trend still holds but momentum
    # softens, allow a controlled red slip of -500..-700 before exiting.
    loss_recovery_watch_max_loss_rs: float = 700.0
    loss_recovery_watch_soft_loss_rs: float = 500.0
    loss_recovery_watch_breathing_mult: float = 1.0
    loss_recovery_watch_giveback_rs: float = 250.0
    loss_recovery_watch_giveback_r_multiple: float = 0.10
    loss_recovery_watch_max_minutes: float = 8.0
    loss_recovery_watch_adverse_intel: float = 0.65
    # Winners use the opposite logic: preserve more of peak when late-session
    # context turns adverse/neutral, but give supportive trends a little more room.
    eod_profit_adaptive_peak_enabled: bool = True
    eod_profit_final_window_minutes: float = 10.0
    eod_profit_adverse_exit_ratio: float = 0.75
    eod_profit_neutral_exit_ratio: float = 0.65
    eod_profit_supportive_exit_ratio: float = 0.45
    # ── Live regime-reversal profit-taker (Fix 3) ───────────────────
    # Babysit an open, profitable trade: if the instrument's live regime flips to
    # oppose the position AND profit is fading off its peak, take the profit.
    # Profit-only — never forces a loss (the hard stop still handles downside).
    live_reversal_exit_enabled: bool = True
    live_reversal_exit_fade_ratio: float = 0.85
    # Near-peak threat exit: if a meaningful winner is still holding most of its
    # peak but live evidence flips hard against it (fast reversal, weak volume/OI,
    # adverse intel, cross-index turn), exit at market instead of waiting for the
    # profit-lock/giveback layers to be hit.
    peak_threat_exit_enabled: bool = True
    # 750 -> 400 on 2026-08-17. Calibration, not a new rule: at Rs.750 this guard could
    # only ever see 26% of trades, because the measured peak distribution of the 222
    # positions that went green is p25 Rs.216 / median Rs.454 / p75 Rs.875. It was set
    # for a book of Rs.750+ peaks and the book is made of Rs.454 peaks.
    #
    # The uncovered middle is where the money went: 110 trades peaked between Rs.150 and
    # Rs.750, and 66 of those closed NEGATIVE for -Rs.16,837.
    #
    # Low risk because the ARM is not the trigger. Every evidence gate still applies —
    # peak_threat_exit_min_score, >=2 evidence domains with price mandatory,
    # peak_threat_exit_min_retention, the confirm window and the round-trip cost floor.
    # Lowering the arm only lets the guard CONSIDER a trade it previously could not look
    # at; it cannot fire without adverse evidence.
    #
    # NOT replay-validated: this guard needs OI/volume/intel evidence scores that the
    # pnl_path replay does not carry, so unlike the breakeven stop this is a threshold
    # aligned to a measured distribution rather than a simulated result. Revert to 750
    # if live sessions show it firing on trades that would have recovered.
    peak_threat_exit_min_peak_rs: float = 400.0
    peak_threat_exit_min_peak_r: float = 0.35
    peak_threat_exit_min_retention: float = 0.72
    peak_threat_exit_min_score: int = 5
    peak_threat_exit_severe_score: int = 4
    peak_threat_exit_adverse_intel: float = 0.55
    peak_threat_exit_weak_volume_ratio: float = 1.05
    peak_threat_exit_loss_enabled: bool = True
    peak_threat_exit_loss_min_score: int = 6
    peak_threat_exit_loss_max_loss_r: float = 0.35
    # The loss arm used to fire on `current_pnl <= 0.0` — ANY negative, including one
    # rupee of bid-ask spread. Closing costs a round trip (opt_cost/fut_cost), so an exit
    # taken while the price loss is smaller than that cost destroys value by construction:
    # on 2026-08-10 nine trades were cut for price losses of Rs.10-91 at Rs.80 a round
    # trip, and half the day's loss was pure cost. The loss must now be at least this
    # multiple of the round trip before the arm is live. 0 restores the old behaviour.
    peak_threat_exit_loss_min_cost_multiple: float = 1.0
    # No loss-threat exit inside this many seconds of entry. The same day produced exits
    # 4s, 33s and 60s after a fill, on signals that had just spent 31s in stabilization —
    # the position was cut before its own thesis had a chance to print a bar.
    peak_threat_exit_min_hold_seconds: float = 60.0
    # Winner side of the same guard, ported back from restore_point_51. That build:
    # "gap/synthetic marks at the open can manufacture an instant peak that arms
    # this exit on a seconds-old position (2026-07-15; ledger rows exited 1-22s
    # after entry)". This build kept the loss-side hold above and lost this one.
    # The winner exit must survive two observations, as the loss exit already does
    # (peak_threat_exit_loss_confirm_seconds). Ported from restore_point_51, which
    # confirms both sides; this build confirmed only losses, so a single tick that
    # scored high enough cut a winner outright.
    peak_threat_exit_confirm_seconds: float = 5.0
    peak_threat_exit_min_age_enabled: bool = True
    peak_threat_exit_min_age_seconds: float = 45.0
    peak_threat_exit_open_window_end: str = "09:30"
    peak_threat_exit_open_window_min_age_seconds: float = 120.0
    # Whether a NEUTRAL/HOLD/MIXED intelligence read counts toward the adverse score.
    # It used to add +1 each for the aggregate and order-flow pillars, so an unclear tape
    # contributed 2 of the 6 points needed to exit — absence of evidence scored as
    # evidence against the trade, the same defect pattern as the repaint and
    # single-pillar-conviction bugs. Off means only genuinely opposing reads score.
    peak_threat_exit_neutral_counts: bool = False
    # Ported from restore_point_51, whose loss arm required BOTH of these before cutting.
    # min_domains: how many distinct evidence kinds must speak (price is mandatory on top
    # of this) — a score alone cannot tell "the tape broke" from "soft intel agrees".
    # confirm_seconds: the same domain signature must be seen twice, that far apart on the
    # MARKET clock. Scanner cycles repeat against one timestamp, so counting calls would
    # manufacture persistence; the clock must actually advance.
    # Days of remaining exchange-calendar coverage below which readiness warns. The
    # bundled NSE holiday list is versioned by year and an uncovered year fails CLOSED —
    # correct, but silent: the system runs, scans and serves a dashboard while placing
    # nothing. Warning early turns a five-month fuse into a routine data update.
    # The unfilled tail of a partially filled ENTRY. The 12s cancel/replace timer stops
    # applying the moment a partial fill lands (it requires position is None AND
    # ENTRY_SUBMITTED), so the remainder used to rest untouched all session — and a late
    # spike could fill it while the protective stop still covered only the first tranche.
    partial_entry_remainder_cancel_enabled: bool = True
    partial_entry_remainder_timeout_seconds: float = 120.0
    calendar_coverage_warn_days: int = 45
    peak_threat_exit_loss_min_domains: int = 2
    peak_threat_exit_loss_confirm_seconds: float = 3.0
    # A same-index loss run used to be measured on the GLOBAL exit tape, so any other
    # index's completed trade cleared it. With four books interleaving, the
    # max_consecutive_losses halt became unreachable: on 2026-08-10 NIFTY lost four
    # times and the streak read 1, 1, 1, 2, so the halt at 3 never fired.
    consecutive_loss_streak_per_index: bool = True
    # Post-entry cross-index correlation read: while a trade is live, if a broad set
    # of the OTHER active indices' live regimes turn against the position (peers need
    # not all correlate — non-directional peers just don't vote), take the profit.
    cross_index_reversal_exit_enabled: bool = True
    cross_index_reversal_min_opposing: int = 2
    # ── Late-entry protect-first-profit (Fix 4) ─────────────────────
    # An entry taken in the final minutes has no runway to run, so mark it
    # scalp-lock (protect first profit) at the live gate.
    late_entry_scalp_enabled: bool = True
    late_entry_scalp_window_minutes: float = 30.0
    trend_option_profit_lock_defer: bool = True
    trend_option_profit_lock_severe_retention: float = 0.50
    # 5m/15m grade-qualified option winners can keep pushing after T1. Do not let
    # the major-win floor ratchet to 99-100% of peak unless adverse evidence exits
    # it first; the peak-threat layer handles true reversal/melt warnings.
    option_major_win_trend_lock_ratio_cap: float = 0.95
    # Broad one-way cascade: when core indices expand in the same direction with
    # range + volume, treat winners as runners, not protected scalps. Entry gates
    # may admit same-side raw signals with tight initial risk, while exit management
    # trails 10-15% off total gains instead of firing ordinary profit locks.
    one_way_cascade_enabled: bool = True
    one_way_cascade_core_votes: int = 3
    one_way_cascade_total_votes: int = 3
    one_way_cascade_volume_votes: int = 3
    one_way_cascade_min_volume_ratio: float = 1.40
    one_way_cascade_min_range_atr_ratio_1min: float = 1.50
    one_way_cascade_min_range_atr_ratio_5min: float = 1.20
    one_way_cascade_min_range_atr_ratio_15min: float = 0.75
    one_way_cascade_min_close_location: float = 0.68
    one_way_cascade_runner_lock_pct: float = 0.85
    one_way_cascade_options_sl_pct: float = 5.0
    one_way_cascade_futures_sl_pct: float = 0.05
    one_way_cascade_rr: float = 1.5
    # Missed signal re-entry: a recent rejected/missed signal can be rebuilt only
    # while the same-side trend/cascade evidence is still fresh and nearby. It is
    # intentionally one-trade-per-index and refuses stale/chase entries.
    # Retrack window: 20s → 360s (6.0 min) after soft reject so 1min + 5min
    # (and 15min) can re-enter if the setup strengthens post-rejection.
    missed_signal_reentry_enabled: bool = True
    # ── Scored re-entry (A/B, 2026-08-12) ────────────────────────────────────
    # The strict path demands FOUR conditions at once — trending regime AND regime
    # aligned AND ADX>=24 AND intel confirmed — on a signal that was already rejected
    # once, with confidence and cascade gating separately. is_trending_regime alone
    # fails on the ~62% of labels that read RANGING, so the other three are rarely even
    # evaluated. Result: 3 re-entries in 19 sessions against 60,425 rejections.
    #
    # Scored mode asks for N of these SIX instead of all of one branch:
    #   confidence, trending_regime, regime_aligns, adx, intel, cascade
    #
    # What it does NOT relax, because these are safety rather than quality: the signal
    # must still be 20s-6min old (a stale seed is a different trade), there must be
    # runway before square-off, the price must not have run past the chase limit, and
    # every repaint rejection still stands. Ships OFF — the strict path is unchanged
    # until this is switched on, which is what makes it an A/B.
    missed_signal_reentry_scored_mode_enabled: bool = False
    missed_signal_reentry_min_conditions: int = 5
    missed_signal_reentry_max_age_minutes: float = 6.0
    missed_signal_reentry_min_age_seconds: float = 20.0
    # 1min + 5min primary retrack; 15min kept for structure flips.
    missed_signal_reentry_timeframes: list[str] = ["1min", "5min", "15min"]
    missed_signal_reentry_max_chase_atr: float = 0.75
    # Cascade chase extension: when a one-way cascade is CONFIRMED (3+ indices
    # synchronized impulse), the 0.75-ATR chase cap is exactly what kept the
    # 2026-07-08 13:35 SENSEX seed out of a -1,185pt crash — by the time the
    # cascade confirms, price has always moved past 0.75 ATR. Cascade-confirmed
    # seeds may chase further (still bounded, still scalp-locked/tight-SL).
    missed_signal_reentry_cascade_max_chase_atr: float = 2.0
    missed_signal_reentry_min_confidence: float = 0.55
    # Exception-route intelligence floors, measured 2026-08-11 against 824 real
    # aggregator reads over the stored index candles. These are the fraction of a
    # fully-informed max-conviction read a candidate must show, and 0.20 admitted 39%
    # of all reads — 2 in 5 is not an exception. 0.35 admits ~15% (1 in 7). Not raised
    # further (0.40 -> 8%, 0.50 -> 2%) because the measurement covers the thin-evidence
    # regime with no option chain; confirm against a live session before tightening.
    #
    # These were 0.20/0.20/0.25 in rp46-2.1 and restore_point_51 too, but the aggregate
    # normaliser changed underneath them: it divided by the COUNT of voting pillars and
    # now divides by their WEIGHT MASS, so the same evidence scores ~3.5x higher and the
    # thresholds were never re-cut to match.
    missed_signal_reentry_min_intel_score: float = 0.35
    missed_signal_reentry_min_trend_adx: float = 24.0
    missed_signal_reentry_min_remaining_minutes: float = 8.0
    missed_signal_reentry_allow_trend_context: bool = True
    missed_signal_reentry_prefer_options: bool = True
    missed_signal_reentry_options_sl_pct: float = 5.0
    missed_signal_reentry_futures_sl_pct: float = 0.05
    missed_signal_reentry_rr: float = 1.5
    # Choppy option scalp protection: once a scalp-lock option trade has already
    # shown a meaningful peak, do not let the generic option breathing room dilute
    # a SMART_PROFIT_LOCK/MAJOR_WIN_GUARD floor. The base arm is Rs.1000; quiet or
    # warning-heavy conditions may lower it to Rs.750/Rs.500 so a low-range day does
    # not need a large absolute move before protecting a premium melt.
    choppy_profit_lock_bypass_peak_rs: float = 1000.0
    choppy_profit_lock_low_move_peak_rs: float = 750.0
    choppy_profit_lock_urgent_peak_rs: float = 500.0
    choppy_profit_lock_low_range_vs_10day: float = 0.65
    choppy_profit_lock_adverse_intel: float = 0.65
    choppy_profit_lock_weak_volume_ratio: float = 1.05
    choppy_profit_lock_premium_melt_ratio: float = 0.45
    choppy_profit_lock_premium_fade_ratio: float = 0.65
    # Trend-day exit hygiene (2026-06-24): the system scalped +Rs.174-694 winners on a
    # +1-1.6% trending day while an older build rode the same names for +Rs.2,400-3,240.
    # These three guards stop premature exits on aligned trend trades. All settings-gated.
    #  - A 15m-aligned entry in a trending regime is a trend pullback, not a scalp, so it
    #    must NOT be forced into scalp_lock_mode (which arms the tightest profit-lock).
    trend_entry_skip_scalp_lock: bool = True
    #  - Do not repaint-abort a position that is currently in profit: the entry is working
    #    regardless of an intra-candle signal flicker on a fast trend.
    repaint_abort_skip_when_profitable: bool = True
    # Options often flicker/repaint during the source candle just before premium expansion.
    # Defer those repaint aborts until candle close unless the premium loss is severe
    # relative to the option breathing room.
    option_repaint_abort_defer_until_close: bool = True
    option_repaint_abort_severe_breathing_mult: float = 2.5
    option_repaint_abort_max_deferred_checks: int = 2
    option_repaint_abort_max_deferred_seconds: float = 55.0
    option_internal_mark_on_stale_quote: bool = True
    option_internal_mark_quote_age_seconds: float = 3.0
    option_internal_mark_min_diff_points: float = 1.0
    option_internal_mark_min_diff_pct: float = 0.5
    option_contract_quote_max_age_seconds: float = 10.0
    option_contract_quote_max_age_next_expiry_seconds: float = 30.0
    #  - PCR is a contrarian indicator; do not let an OI/PCR reversal exit a position that
    #    is aligned with a trending regime (it fires constantly mid-trend).
    oi_pcr_reversal_skip_in_trend: bool = True
    # Tight-trailing timeframe (risky-regime stop trail). Trailing a 15min trade by
    # 1-min candle lows shakes trend trades out on noise — e.g. a +9-pt NIFTY exit
    # right before a +190 move. Use a coarser candle so trend trades breathe. The
    # trade's own timeframe is used when >= this floor; set "1min" for legacy.
    tight_trailing_timeframe: str = "5min"

    # Optional absolute per-trade loss backstop (₹). 0 = DISABLED (default):
    # per-trade risk is governed ONLY by the user's visible settings — the
    # segment stop caps (futures_sl_pct / options_sl_pct) and the daily
    # circuit-breaker budgets (risk_fut_pct / risk_opt_pct). Set >0 only if an
    # extra rupee backstop per trade is explicitly wanted.
    max_trade_loss_abs: float = 0.0
    dynamic_rr_min: float = 1.5 # Minimum initial RR; final policy recomputes it from execution levels
    # Reachable quality ladder tops near 4.92R. 5.0 is an honest global ceiling;
    # runner mode, not an unreachable 12R initial target, captures exceptional moves.
    dynamic_rr_max: float = 5.0
    dynamic_rr_runner_unlock_ratio: float = 0.85 # T1 progress before removing hard target (lowered so winners flip to trail-and-ride earlier; was 1.00)
    dynamic_rr_runner_lock_pct: float = 0.85 # Runner-mode trailing lock on open gains
    # 5min/1min entries cannot travel the top of the RR ladder within one short bar's
    # horizon before the profit-lock/square-off fires, so cap short-timeframe RR
    # separately from the 15min swing ladder. Set == dynamic_rr_max to disable.
    dynamic_rr_5min_max: float = 2.5
    # RR-scaled position sizing. The legacy `1.0 if rr>=1.5 else rr/1.5` factor was
    # dead because dynamic_rr already floors at 1.5R, so it never shrank size. When enabled, size is
    # scaled down for low-RR setups toward rr_size_floor and reaches full size once
    # final_rr >= rr_size_reference_rr. Default OFF so live sizing is unchanged until
    # replay-validated.
    rr_size_scaling_enabled: bool = False
    rr_size_reference_rr: float = 2.5
    rr_size_floor: float = 0.5

    # ─── Dashboard & Backtest ────────────────────────────────
    dashboard_port: int = 8001
    dashboard_host: str = "127.0.0.1"
    dashboard_auth_token: str = ""
    dashboard_allowed_origins: list[str] = []
    default_backtest_days: int = 1 # Default backtest window to 1 day as requested
    settings_recalculation_debounce_seconds: float = 8.0
    settings_persistence_debounce_seconds: float = 0.75
    recalculation_worker_enabled: bool = True
    recalculation_worker_timeout_seconds: float = 120.0

    # Machine-wide resource guard for unattended REAL Auto.  These thresholds
    # block only new Auto exposure; exit monitoring, broker reconciliation, and
    # manual lifecycle handling remain active.  The guard is conservative because
    # host commit exhaustion can kill the whole service before Python reports it.
    auto_resource_guard_enabled: bool = True
    auto_resource_ram_block_pct: float = 94.0
    auto_resource_commit_block_pct: float = 92.0
    auto_resource_commit_min_available_mb: float = 1024.0
    auto_resource_clear_hysteresis_pct: float = 4.0
    auto_resource_clear_consecutive_samples: int = 2
    auto_resource_sample_max_age_seconds: float = 5.0
    # Whether the RAM/commit PERCENTAGES above block new Auto entries (True) or only
    # raise a watchdog warning (False). Blocking always applies to exhausted commit
    # headroom (auto_resource_commit_min_available_mb) and to a missing/stale sample.
    # 2026-09-11: the percentages refused every LONG of a 0.5-0.8% four-index rally at
    # 92-95.8% commit with 1.2-2.3 GB of headroom still free; the headroom floor has
    # never fired in any recorded block.
    auto_resource_pct_blocks_entries: bool = False
    # Keep repeated backtests deterministic by default. Set this true only when
    # a recalculation should also refresh broker candle files before replay.
    backtest_refresh_data_on_recalc: bool = False
    scanner_stale_after_seconds: float = 5.0
    # While historical rebuild / calculation lock is held, use a much longer
    # stale threshold so multi-minute deterministic replays are not "critical".
    scanner_stale_while_busy_seconds: float = 600.0
    api_default_trade_limit: int = 100
    api_max_trade_limit: int = 2000
    api_default_chart_bars: int = 300
    api_max_chart_bars: int = 1000
    outcome_calibration_enabled: bool = True
    production_safe_mode_required: bool = True
    # REAL mode can be armed before the cash-market open so broker feeds,
    # reference quotes, expiries, and option intelligence are warm before 09:15.
    # Order placement remains blocked by market-open and strategy session gates.
    real_mode_premarket_arm_start: str = "09:00:00"
    real_mode_premarket_settle_end: str = "09:08:00"
    real_mode_premarket_refresh_end: str = "09:14:30"
    real_mode_live_start: str = "09:15:00"
    real_mode_live_end: str = "15:30:00"
    real_mode_mid_session_restart_refresh_end: str = "15:00:00"
    # Runtime-only helper: when the app is left running in HISTORICAL mode,
    # arm REAL/manual mode at the premarket window. It never enables Auto Mode.
    auto_real_mode_premarket_enabled: bool = True
    auto_real_mode_premarket_switch_time: str = "09:00:00"
    auto_real_mode_premarket_retry_seconds: float = 30.0
    # Explicit unattended-start intent.  REAL processes still begin manual; only a
    # complete same-day warmup plus the normal broker/option-chain readiness gate can
    # turn Auto on at/after the live session start.  This remains separate from
    # ``auto_mode`` because that flag is runtime authority and is deliberately never
    # persisted across a broker-process restart.
    auto_real_mode_startup_enabled: bool = False

    # ─── Signal engine selector ──────────────────────────────
    # Which raw-signal scanner feeds the pipeline. Default "quad_prime"
    # (all four sources can enter; opposite exits need cross-family confirm).
    # Prior default was "utbot_consensus"; set "utbot" for pure UT1 baseline.
    # "ema": EMA(9/21) crossover. "consensus": Supertrend(10,3) AND EMA(9/21) must
    # agree (holds through disagreement) — far less whippy than UT1 in backtests.
    # Direction comes from the chosen source; the protective stop stays the UT ATR
    # trail, and all downstream grading/gates/exits are untouched. "quad_union"
    # unions UT1 + EMA + Supertrend + Consensus raw arrows after repaint buffering.
    # "quad_prime": all four raw sources can enter like quad_union, but opposite exits
    # need cross-family confirmation (UT1 + EMA/Supertrend/Consensus) unless the
    # trade is losing or protected profit is threatened.
    # "ut_smart": UT SMART EXIT (2026-09-25), the user's "UT Bot + Smart Peak Exit v6"
    # Pine script with its own ATR entries and exits (engine/ut_smart_engine.py and
    # engine/uts_lane.py). It replaced ADAPTIVE DUAL in the dashboard selector;
    # adaptive_dual stays accepted here for .env/engine-lab compatibility only.
    # "adaptive_dual": Engine 11. Lane A is adaptive_hybrid v2; Lane B runs the
    # uploaded APEX v5.5 five-gate confluence engine (regime, HTF direction,
    # score, conflict margin, volume/body confirmation), then sends both lanes
    # through the same gates, sizing, quote, risk, and exit stack.
    # "s1s2_dual": S1+S2 Dual (2026-10-10), the user's Pine v5.4 S1 + S2 Dual script
    # (engine/s12_engine.py, engine/s12_lane.py): two independent lanes, D1 and D2,
    # each with its own stop and exits.
    scanner_engine: Literal[
        "utbot",
        "ema",
        "consensus",
        "utbot_consensus",
        "quad_union",
        "quad_prime",
        "adaptive_dual",
        "s1",
        "ut_smart",
        "quad_prime_s1",
        "ut_smart_s1",
        "s1s2_dual",
        "ut_ema",
    # The code default is what a deployment silently runs when SCANNER_ENGINE is
    # missing from .env (a pruned .env has reverted keys before). Keep it on the
    # engine live trades with; s1 (added 2026-09-23) is selectable through .env or
    # the dashboard.
    ] = "quad_prime"
    # Quad Prime raw-entry hygiene: raw source access is allowed, but weak solo flips
    # must not skip all live quality floors or every winner becomes a tiny scalp while
    # low-confidence losers keep wide option stops.
    # The intelligence bar the exceptional-spike leg must clear, ALIGNED with the
    # trade's own side. Matches rare_burst_min_intel_score, which uses the same 0.70
    # with the same signed comparison.
    # ── Engine 10 v2: regime-switched delegation to the proven engines ──
    # v1 re-invented selection from raw arrows and lost 10x to the engines it
    # routed (three 30d backtests Rs.17-21k vs Rs.190k). v2 emits ONLY the two
    # proven engines' flip streams — trend/opening state passes the full
    # utbot_consensus union; chop demands utbot_prime-style cross-source
    # agreement or strict consensus; volume thrust enters fast. v2 signals are
    # NEVER raw-union flagged: they get full normal grading/RR/runner/FUT
    # economics, exactly like utbot_consensus.
    adaptive_hybrid_trend_adx: float = 25.0
    adaptive_hybrid_trend_er: float = 0.68
    adaptive_hybrid_opening_er: float = 0.58
    adaptive_hybrid_chop_min_agreement: int = 2
    adaptive_hybrid_thrust_volume_ratio: float = 1.8
    adaptive_hybrid_thrust_range_atr: float = 1.15
    adaptive_hybrid_thrust_close_loc: float = 0.62
    # Engine 11 dual-lane safety. Selecting adaptive_dual is the main gate; this
    # toggle lets the second APEX lane be disabled without losing Lane A.
    dual_lane_enabled: bool = True
    dual_lane_same_dir_min_bars_gap: int = 3
    dual_lane_same_dir_min_lane_a_r: float = 0.50
    dual_lane_size_fraction: float = 0.65
    dual_lane_max_index_exposure_rs: float = 12000.0
    dual_lane_min_lane_b_confidence: float = 0.70
    dual_lane_jaccard_alert_threshold: float = 0.30
    # Engine 12 independent confluence thresholds.
    quality_hybrid_min_vote_score: float = 3.0
    quality_hybrid_broad_agreement: int = 4
    quality_hybrid_chop_min_agreement: int = 4
    quality_hybrid_trend_adx: float = 22.0
    quality_hybrid_trend_er: float = 0.50
    quality_hybrid_thrust_volume_ratio: float = 1.6
    quality_hybrid_thrust_range_atr: float = 1.05
    quality_hybrid_thrust_close_loc: float = 0.60

    # --- S1 confluence engine (SCANNER_ENGINE=s1; engine/s1_engine.py) ---
    # Port of the user's "S1" TradingView script. Defaults are the user's 2026-09-23
    # inputs (the Pine's own defaults in brackets). Editable from the dashboard.
    s1_candle_type: Literal["standard", "heikin_ashi"] = "heikin_ashi"
    s1_min_score: float = 60.0          # Min bull/bear confluence score [70]
    s1_conflict_margin: float = 20.0    # Min bull-minus-bear score gap [20]
    s1_min_adx: float = 20.0            # Regime gate: ADX(14) floor [20]
    s1_use_htf: bool = True             # Require closed-15m EMA9/21 alignment and slope [on]
    s1_signal_cooldown: int = 0         # Min bars between S1 signals [5]
    s1_sl_mode: str = "anchor"         # Live SL1: dynamic 0.10-0.25% (anchor = 3-bar extreme) | atr | fixed
    s1_sl1_pct: float = 0.10            # Initial SL1 % of index LTP (user setting)
    s1_sl_min_pct: float = 0.10         # Dynamic SL1 floor, % of index price
    s1_sl_max_pct: float = 0.25         # Dynamic SL1 cap, % of index price
    s1_sl_atr_mult: float = 1.0         # atr mode: SL1 distance = k x ATR(14)
    s1_use_sl2: bool = False            # SL2 gap stop on the live order [chart: always on]
    s1_sl2_pct: float = 0.40            # Live SL2 % (only when s1_use_sl2 is on)
    # The chart's own SL inputs, used by the TradingView state machine that decides which
    # bars signal (the user's chart labels print SL1 0.20% / SL2 0.40%).
    s1_chart_sl1_pct: float = 0.20
    s1_chart_sl2_pct: float = 0.40
    # TradingView's index charts end at 15:15; later feed bars are dropped before S1 runs.
    s1_session_end: str = "15:15"
    s1_t1_r: float = 2.0                # Targets as multiples of the SL1 distance [2/3/4]
    s1_t2_r: float = 3.0
    s1_t3_r: float = 4.0
    s1_trail_start_r: float = 1.5       # Trail starts at this R [1.5]
    s1_trail_mult: float = 1.8          # Trail = peak -/+ k x ATR [1.8]
    s1_exit_conf_bars: int = 3          # Momentum-exit confirmation bars [3]
    s1_circuit_pause: int = 5           # Bars paused after the loss streak [10]
    # Consecutive losses before the pause, per index. 0 = follow the system setting
    # max_consecutive_losses ("Stop After Consecutive Losses"); any other value is a
    # custom per-index count. The run is per index and resets on a win.
    s1_max_consec_loss_nifty: int = 0
    s1_max_consec_loss_banknifty: int = 0
    s1_max_consec_loss_sensex: int = 0
    s1_max_consec_loss_midcpnifty: int = 0
    # pine = the supplied indicator state rules; close = close-entry replay.
    s1_entry_mode: Literal["pine", "close"] = "pine"
    # S1 lane (engine/s1_lane.py): S1 signals skip the other engines' quality/grading
    # selection and are managed by S1 exits; every safety control still applies.
    s1_lane_enabled: bool = True
    s1_lane_guard_pct_1min: float = 5.0    # Max premium loss per trade by signal timeframe
    s1_lane_guard_pct_5min: float = 10.0
    s1_lane_guard_pct_15min: float = 15.0
    s1_lane_guard_late_after: str = "14:30"  # Entries from here get one guard step tighter
    s1_lane_guard_strong_score: float = 80.0  # S1 score at/above this gets one step wider
    # Guard cap by regime is fixed: 10% in a clear trend, 15% in choppy/volatile/unknown.
    s1_lane_t1_lock_r: float = 0.7         # Index stop moves to entry + this R once T1 prints
    # Profit lock arms at this multiple of premium risk. 1.0R (2026-10-09, user): live S1
    # rows Oct 5-9 (30) averaged +Rs406 per win against -Rs1,609 per loss, the lock arming at
    # 0.5R and closing winners on the first 10% dip. The 2026-09-25 0.5R study ran on the
    # look-ahead replay with 1-minute bars, which cannot reproduce a tight trail.
    s1_lane_lock_arm_r: float = 1.0
    # 2026-10-08 payoff audit (user-approved): the entry RR floor is the base target
    # (0.5R admitted 0.7R-target geometry that cannot pay after costs), and a breakeven
    # premium stop (entry + round-trip cost) arms once the peak gain reaches 0.25R so a
    # trade that worked a little no longer runs to a full -1R.
    s1_lane_rr_floor_at_target: bool = True
    # S1 stays signal-only under Auto until a look-ahead-free replay shows edge (audit 2026-10-08).
    s1_lane_signal_only: bool = True
    s1_lane_breakeven_enabled: bool = True
    # 1.0R (2026-10-09, user): at 0.25R the breakeven stop sat a few points under the peak
    # and closed winners for a scratch (SENSEX 72300CE +Rs29 on 2026-10-09).
    s1_lane_breakeven_arm_r: float = 1.0
    s1_lane_breakeven_cost_rs: float = 140.0
    # Fixed index stop per timeframe for live S1 levels (% of index LTP); 0 keeps s1_sl_mode.
    # User proposal 2026-10-08 (5m 0.10%, 15m 0.15%). Corrected 20-session replay with native
    # S1 exits (Sep 9-Oct 8, 15 trades): -4.7k vs -8.7k for the anchor stop and -8.6k for a
    # fixed 0.25% -- least negative, still negative, tiny sample. S1 stays signal-only.
    s1_sl1_pct_5min: float = 0.10
    s1_sl1_pct_15min: float = 0.15
    # ...then exits on a giveback from the peak gain that follows the S1 trend (the score
    # gap on the last closed bar, >= s1_lane_trend_margin counts as a clear trend).
    s1_lane_giveback_with_trend: float = 25.0
    s1_lane_giveback_neutral: float = 10.0
    s1_lane_giveback_against: float = 0.0
    # UNKNOWN trend: no closed bar after the signal yet, engine state missing, or a stale
    # bar; capped at s1_lane_giveback_with_trend whatever it is set to. 25% (2026-10-09,
    # user): UNKNOWN covers every trade's first bar (5 or 15 minutes), where 10% closed
    # winners on the first dip.
    s1_lane_giveback_unknown: float = 25.0
    s1_lane_trend_margin: float = 20.0
    # S1 entries end at the system-wide last-entry times (ut_no_entry_after and siblings,
    # engine/entry_window.py; no EOD-protect clamp; the 15:15 force exit still applies).
    # Buffer for an S1 signal on a forming bar: 15s, 5s on 1min (2026-10-09, user; was 30s).
    s1_lane_intrabar_wait_seconds: float = 15.0
    s1_lane_intrabar_wait_seconds_1min: float = 5.0
    # While only the forming bar changes, S1 recomputes a chart at most this often; a new
    # closed bar (or a stabilization request) always recomputes at once. 0 = every scan.
    s1_lane_forming_recompute_seconds: float = 15.0
    # S1 target (2026-09-25 user): 1R on a normal trade, 0.7R when the move is weak (at
    # entry, or later when a closed bar shows the move has died); while the S1 trend is
    # clearly with the trade the target does not close it and it may run to
    # s1_lane_runner_max_r. Weak = at least two of: volume below s1_lane_weak_rel_volume x
    # its 20-bar average, ADX below s1_lane_weak_adx, ATR below its 50-bar average.
    s1_lane_base_rr: float = 1.0
    s1_lane_weak_rr: float = 0.7
    s1_lane_runner_max_r: float = 30.0
    # The runner may reach s1_lane_runner_max_r only in a special trend: S1's closed-bar
    # score gap on the trade's side >= s1_lane_special_trend_margin AND a clear trend regime
    # on its side, or a heavy-trend entry. Otherwise it is capped at s1_t3_r (4R).
    s1_lane_special_trend_margin: float = 40.0
    s1_lane_weak_adx: float = 20.0
    s1_lane_weak_rel_volume: float = 1.2
    # Final entry check floor for S1 (below 0.7R so a fill a tick worse still passes).
    s1_lane_min_entry_rr: float = 0.5
    # Missed-signal re-entry for S1: a refused or repainted S1 signal may re-enter while
    # the S1 score on the latest closed bar is the same or stronger than the missed one's.
    s1_lane_reentry_enabled: bool = True
    # Re-entry window in bars after the signal candle closes (2026-09-25 user: one bar on
    # 15m and 5m, three bars on 1m).
    s1_lane_reentry_bars_1min: int = 3
    s1_lane_reentry_bars_5min: int = 1
    s1_lane_reentry_bars_15min: int = 1
    # ── UT SMART EXIT engine (scanner_engine="ut_smart") ──
    # Match the uploaded "UT Bot + Smart Peak Exit v6" inputs. The earlier
    # research tuning was a different signal profile, not Pine chart parity.
    # Option admission and the user's 5%/10%/15% premium stops are separate.
    uts_key_value: float = 1.0
    uts_atr_period: int = 10
    # Independent overrides remain available, with the same Pine defaults.
    uts_key_value_5min: float = 1.0
    uts_atr_period_5min: int = 10
    uts_key_value_1min: float = 1.0
    uts_atr_period_1min: int = 10
    uts_use_heikin_ashi: bool = False
    uts_smart_exit: bool = True
    uts_arm_atr: float = 0.8
    uts_retrace_atr: float = 0.45
    uts_hard_retrace_atr: float = 1.1
    uts_min_bars: int = 2
    uts_confirmations: int = 2
    uts_rsi_length: int = 14
    uts_rsi_ob: float = 70.0
    uts_rsi_os: float = 30.0
    uts_rsi_lookback: int = 5
    uts_exit_ema: int = 5
    uts_wick_ratio: float = 0.35
    # This index feed appends a 15:15 settlement/frozen bar absent from the
    # uploaded chart. Exclude it; including it erases today's 09:15 BUY.
    uts_session_end: str = "15:15"
    # The lane: UT entries with binding quality, own exits, every safety control.
    uts_lane_enabled: bool = True
    # The main timing toggle also controls UTS. Intrabar mode includes closed signals.
    uts_lane_allow_intrabar_entries: bool = True
    uts_lane_result_max_age_seconds: float = 30.0
    # Timeframes: the dashboard TF toggle (ut_timeframe_entry_policy), as for every engine
    # (2026-09-26 user). Tested: 15m is the tuned frame; 5m had 0 of 4,920 parameter sets
    # positive in both halves, 1m 0 of 4,920 (1m bursts only: 14 trades, -Rs.1.1k).
    # Entries (2026-09-26 user): normal UT SMART signals trade from 09:15 until the
    # system-wide last-entry times -- 15m 14:50, 5m 15:10, 1m 15:14 (ut_no_entry_after and
    # siblings, engine/entry_window.py) -- then the 15:15 hard exit. The lane's own early
    # cutoffs (15m 12:45 / 5m 14:00 bar start, from the replay study: 15m entries after
    # 13:00 were flat or negative) are gone; the user wants late big moves traded.
    # Late big-move check from this time, on every timeframe: the late_session_reclaim
    # test (3+ indices with the same late 1m and 5m move, volume and strong closes),
    # voting on the move itself. A confirmed move admits a 1m signal without the 1m frame
    # selected and labels 5m/15m entries.
    uts_lane_reclaim_start: str = "13:00"
    # Index stop ceiling for the final policy: the UT stop is key x ATR (median 0.29-0.41%
    # of the index on 15m at key 2.0).
    uts_lane_sl_max_pct: float = 1.0
    uts_lane_min_entry_rr: float = 0.5
    # The chart has no target: nominal only (RR checks); it never closes the trade.
    uts_lane_target_r: float = 3.0
    # Premium broker stop = the UT stop distance through delta x this multiple, clamped.
    # The chart reacts on the close only; its worst intrabar excursion was 1.89R (winners
    # 1.16R), and a 2R catastrophe stop cost nothing in the study.
    uts_lane_guard_mult: float = 2.0
    uts_lane_guard_min_pct: float = 10.0
    uts_lane_guard_max_pct: float = 15.0
    # Protect observed option gains before the index ATR exit can arm.
    uts_lane_lock_arm_r: float = 1.0
    uts_lane_lock_giveback_pct: float = 30.0
    # 5m UTS intel gate (2026-10-06 user): a plain 5m UT SMART flip enters only when the
    # market-intelligence read of that index agrees. 19-session replay (Sep 7-Oct 5):
    # ungated 5m UTS 480 trades, -17.1k at realistic cost; requiring volume >= 1.0x,
    # ADX >= 20 and directional RSI together kept 59 trades at +5.9k. Each leg alone did
    # not help (volume alone was worse), so the legs are joint. PCR uses the intel
    # engine's own PCR signal and vetoes only a clearly opposing read; replay has no
    # chain history, so the PCR leg is untested. Unmeasured volume fails the gate.
    uts_lane_5m_intel_gate_enabled: bool = True
    uts_lane_5m_min_volume_ratio: float = 1.0
    uts_lane_5m_min_adx: float = 20.0
    uts_lane_5m_require_directional_rsi: bool = True
    uts_lane_5m_pcr_veto_enabled: bool = True
    # When both UTS opt-in and the general intra-candle toggle are enabled, a
    # UT flip on the forming 5m/15m candle must hold this long before entry. Closed-candle
    # signals never wait.
    # 15s, 5s on 1min (2026-10-09, user; was 20s / 6s).
    uts_lane_intrabar_wait_seconds: float = 15.0
    uts_lane_intrabar_wait_seconds_1min: float = Field(default=5.0, ge=5.0, le=7.0)
    # After an intra-candle entry, re-check the flip this often until its candle closes;
    # it exits (UTS_REPAINT) if the flip is gone or the closed candle does not confirm it.
    uts_lane_repaint_check_seconds: float = 10.0
    # After a UTS_REPAINT exit, the same closed-bar signal may re-enter within this many
    # seconds of that exit (2026-10-08 audit: the bar-close freshness deadline blocked it).
    uts_lane_repaint_reentry_grace_seconds: float = 90.0
    # ── S1+S2 DUAL engine (scanner_engine="s1s2_dual"; engine/s12_engine.py) ──
    # The user's "S1 + S2 Dual Independent v1.2" Python port of their Pine v5.4 script,
    # vendored in engine/s12dual. Two independent lanes: D1 is the package's S1 and D2
    # its S2 (neither is the host's S1 lane). Bounds follow the package's Pine inputs;
    # defaults below are the selected bot policy. Intraday modes, no BTST and the engine's own 15:15 EOD are fixed
    # (s12_engine.FIXED_CONFIG). Editable from the dashboard.
    # D1 (the package's S1): confluence gates, % stop, weakness-confirmed early cut and R runner.
    s12_d1_min_score: int = 60
    s12_d1_conflict_margin: int = 20
    s12_d1_min_adx: float = 20.0
    s12_d1_use_15m_gate: bool = True
    s12_d1_cooldown_bars: int = 1
    s12_d1_stop_pct: float = 0.1
    s12_d1_exit_at_t3: bool = False
    s12_d1_trail: bool = True
    s12_d1_trail_start_r: float = 1.5
    s12_d1_trail_atr: float = 1.8
    s12_d1_lock_at_t1: bool = True
    s12_d1_exit_confirm_bars: int = 3
    s12_d1_patterns: bool = True
    s12_d1_pattern_scoring: bool = True
    # Shared by both lanes: the intraday hard stop cap, the R ladder and the loss circuit.
    s12_hard_stop_intraday_pct: float = 1.5
    s12_t1_r: float = 1.5
    s12_t2_r: float = 2.25
    s12_t3_r: float = 4.0
    s12_share_target_ladder: bool = True
    s12_consecutive_loss_limit: int = 3
    s12_circuit_pause_bars: int = 5
    # D2 (the package's S2): stop mode, target milestones, R runner/early cut, exits, burst/ORB/divergence.
    s12_d2_settings_mode: Literal["Auto", "Manual"] = "Auto"
    s12_d2_risk_mode: Literal["Dynamic", "ATR", "Percentage", "Swing"] = "Percentage"
    s12_d2_min_stop_pct: float = 0.1
    s12_d2_max_stop_pct: float = 0.0
    s12_d2_dynamic_sensitivity: float = 1.0
    s12_d2_atr_stop_mult: float = 2.0
    s12_d2_percent_stop_pct: float = 0.1
    s12_d2_t1_manual: float = 1.5
    s12_d2_t2_manual: float = 2.25
    s12_d2_t3_manual: float = 4.0
    s12_d2_circuit: bool = True
    s12_d2_require_15m_alignment: bool = False
    s12_d2_confidence_floor: int = 60
    s12_d2_exit_confirm_input: int = 3
    s12_d2_gap_skip: bool = True
    s12_d2_daily_trend_lock: bool = False
    s12_d2_max_bars_in_trade: int = 250
    s12_d2_macd_entry_filter: bool = True
    s12_d2_signal_gap: int = 0
    s12_d2_trailing: bool = True
    s12_d2_trail_start_manual: float = 1.5
    s12_d2_trail_atr_manual: float = 2.5
    s12_d2_lock_at_t1: bool = True
    s12_d2_profit_lock_r: float = 0.6
    s12_d2_aggressive_trail: bool = False
    s12_d2_tsl_use_close: bool = True
    s12_d2_early_be: bool = False
    s12_d2_early_be_r: float = 0.8
    s12_d2_exit_mode: Literal["Conservative", "Balanced", "Trend-Following"] = "Trend-Following"
    s12_d2_smart_exit: bool = False
    s12_d2_exit_weakness: bool = False
    s12_d2_burst: bool = True
    s12_d2_burst_bars: int = 0
    s12_d2_burst_index_pct: float = 0.25
    s12_d2_burst_volume_mult: float = 1.1
    s12_d2_burst_score_boost: int = 20
    s12_d2_patterns: bool = True
    s12_d2_orb: bool = True
    s12_d2_orb_minutes: int = 30
    s12_d2_divergence: bool = True
    s12_d2_div_lookback: int = 14
    s12_d2_div_score_boost: int = 15
    # The lanes (engine/s12_lane.py): 1m burst/spike, 5m and 15m by default, Auto from day
    # one, each index may hold one D1 and one D2 trade. Exits
    # are the engine's own (early-loss ratchet, R runner, its EXIT) plus the UTS premium-cap
    # tiers and the square-off.
    s12_timeframes: Literal[
        "5min,15min", "5min", "15min", "1min,5min,15min", "1min,5min", "1min,15min", "1min",
    ] = "1min,5min,15min"
    s12_d1_enabled: bool = True
    s12_d2_enabled: bool = True
    s12_signal_only: bool = False
    s12_square_off: str = "15:15"
    # S1+S2 option premium hard cap; global options_sl_pct may tighten it, never widen it.
    s12_options_operator_cap_pct: float = Field(default=10.0, ge=0.5, le=10.0)
    # Index stop ceiling at admission (the engine's own intraday hard cap is 1.5%).
    s12_sl_max_pct: float = 1.5
    s12_min_entry_rr: float = 0.5
    # Nominal premium target for RR checks only: T1-T3 are milestones and never close.
    s12_target_r: float = 4.0
    # Operator lots per index, saved when changed on the dashboard and restored at start
    # (they used to reset to 1 on every restart).
    user_lots_by_index: Dict[str, int] = Field(default_factory=dict)
    user_lots_fut_by_index: Dict[str, int] = Field(default_factory=dict)
    # Macro-event blackout: block fresh ENTRIES during user-listed event windows
    # (data_store/event_blackouts.json: [{date,start,end,label}], e.g. RBI policy,
    # budget). DEFAULT OFF — zero behavior change until enabled. Exits/trailing
    # on open trades are never affected.
    event_blackout_enabled: bool = False
    engine_lab_enabled: bool = True # Shadow-run all 12 engines for comparison; never submits orders
    engine_lab_real_mode_only: bool = False # False also shows lab data during historical/offline scans
    # Live/REAL mode refreshes the lab on a worker thread at most this often; the scan
    # loop always serves the cached snapshot so the 100ms trade monitor never stalls.
    # Shadow diagnostics do not participate in trading. A 10s refresh consumed about
    # one quarter of live samples and could overlap the minute history refresh, starving
    # all four scanner workers. One minute is current enough for this comparison panel.
    engine_lab_refresh_seconds: float = 60.0
    # ─── UT1 warmup stance seed ────────────────────────────
    # UTBot's flip needs price to CROSS the ATR trail. On the rare window where the
    # ATR warm-up parks the trail on the wrong side of a one-way market from the very
    # first bars (e.g. relentless gap-and-trend phase), no crossover ever happens and
    # UT1 sits FLAT while every bar trends. The seed emits ONE synthetic entry per
    # session+direction when ALL of: zero flips in the window, price pinned on one
    # side of the trail for N bars, ADX >= floor, EMA(9/21) agrees, and net price
    # change over the stance run agrees. Protective stop = the ATR trail (already on
    # the correct protective side). Downstream grading/gates apply unchanged.
    utbot_warmup_seed_enabled: bool = True
    utbot_warmup_seed_min_stance_bars: int = 15
    utbot_warmup_seed_min_adx: float = 20.0
    # auto mode: per-bar, use consensus (low-noise) when ADX confirms a trend, else EMA
    # (more adaptive in chop). Gets consensus's trend precision + EMA's chop-resilience.
    scanner_auto_trend_adx: float = 25.0
    # utbot_prime ranking: confidence boost added PER agreeing partner (EMA / Supertrend /
    # Consensus) at the UT1 entry bar, capped by the max. Opt-in mode only — no effect
    # unless scanner_engine == "utbot_prime". Conservative defaults; raise to weight
    # trend-agreement harder, set per-agreement to 0.0 to disable ranking (pure UTBot).
    utbot_prime_confidence_per_agreement: float = 0.03
    utbot_prime_max_confidence_bonus: float = 0.09
    # When a trend engine (ema/consensus) is active, its flips are already smoothed,
    # confirmed trend signals — not UTBot's noisy single-bar flips. So a 15m-aligned,
    # confidence-qualified trend-engine flip bypasses the impulse-context / choppy
    # confirmation gates (which exist to filter UT1 noise). The confidence floor and
    # 15m-agreement requirement still apply. No effect when scanner_engine == "utbot".
    # Combo engines only: a SOLO UT1 opposite-flip must not exit a live trade by itself —
    # UT1 is the jumpy source, so its exits need the partner (ema/consensus) to confirm.
    # The partner emits its own opposite signal when it flips, so the trade simply rides on
    # the partner's (slower, reliable) judgement + the protective stop until then. Entries
    # are unaffected (UTBot's frequency is the combo's entry edge). Toggle off for raw union.
    combo_partner_confirms_exit: bool = True

    # Option rollover now measured in TRADING days (holiday-aware), not calendar days.
    # Roll to next expiry when this many trading sessions (or fewer) remain to expiry.
    # 1 => roll only on the last trading day before expiry (+ expiry day); the system
    # trades the current expiry one session longer than the old T-2 calendar rule.
    option_rollover_working_days: int = 1

    # ─── Expiry-Day Gamma Mode ───────────────────────────────
    # Even with the 1-trading-day rollover above, the system still rolls to next week's
    # options on T-1 + expiry day — exactly when ATM gamma is highest and premiums can
    # 2-3x. When this is ON, on expiry day (and optionally T-1) the system trades the
    # CURRENT expiring ATM (max gamma) instead of rolling — but only
    # in the morning, before the afternoon theta-crush. Default OFF (unchanged rollover).
    expiry_day_trade_current_week: bool = False
    # Cutoff after which the expiry move is usually done and theta decay dominates. Before
    # the cutoff -> current-week gamma; at/after -> see expiry_day_after_cutoff_mode.
    expiry_day_gamma_cutoff: str = "13:00"
    # Also apply gamma mode on T-1 (the day before expiry), not just expiry day itself.
    expiry_day_include_t1: bool = False
    # After the cutoff on expiry day:
    #   "rollover"        -> safest: trade next week's options (the unchanged behaviour)
    #   "momentum_current"-> keep current-week ATM but only the system's momentum-qualified
    #                        entries pass (big, short-tenure moves) — others fall through
    #   "futures"         -> trade the index FUTURE (no theta) instead of options
    expiry_day_after_cutoff_mode: Literal["rollover", "momentum_current", "futures"] = "rollover"
    # ── Expiry-day theta-crush risk controls (apply only to current-expiry gamma trades) ──
    # Expiring ATM options decay to zero fast and can gap hard, so gamma trades get a
    # TIGHTER stop than the normal options_sl_pct (caps the theta-crush loss).
    expiry_day_gamma_sl_pct: float = 10.0
    # Force-square-off gamma trades by this time — never hold an expiring option into the
    # afternoon decay even if SL/target haven't hit. Should be <= expiry_day_gamma_cutoff.
    expiry_day_gamma_time_exit: str = "13:30"
    # Only enter a current-expiry gamma trade on strong momentum (a big short-tenure move);
    # weak/chop entries are skipped so theta doesn't bleed a directionless premium.
    expiry_day_gamma_require_momentum: bool = True
    # Expiry-day LATE window: the big settlement move often comes in the last hour, so on
    # expiry day the system RE-ENGAGES current-expiry gamma between these times (in either
    # direction) — separate from the morning window, skipping the midday lull. Entries are
    # still momentum-gated, and the EOD square-off still flattens everything by force-exit.
    expiry_day_late_window: bool = True
    expiry_day_late_window_start: str = "14:00"
    expiry_day_late_window_end: str = "15:15"
    # Time exit for the late cohort. The comment above said "the EOD square-off still
    # flattens everything by force-exit", and that was literally all that stood between a
    # late-window position and settlement: the 13:30 gamma exit is already past when this
    # window opens, and its entry-time filter drops everything entered after the 13:00
    # cutoff. Defaults ahead of ut_force_exit_time (15:15) so an expiring contract is
    # closed on its own schedule rather than in the final-minutes liquidity crunch,
    # alongside every other force-exited position. Moved 15:20 -> 15:10 with the
    # 2026-08-11 schedule change; at 15:20 it sat AFTER the new force-exit, which
    # inverts the ordering this setting exists to guarantee.
    expiry_day_late_window_time_exit: str = "15:10"

    # ─── UT Bot Core Settings ────────────────────────────────
    ut_preset: str = "AGGRESSIVE"
    ut_atr_period: int = 10
    ut_atr_multiplier: float = 1.0
    ut_use_heikin_ashi: bool = False
    ut_signal_mode: Literal["realtime", "confirmed"] = "realtime"
    ut_adx_filter: bool = True      # ADX is informational, not a signal blocker
    ut_adx_period: int = 14
    ut_adx_threshold: float = 25.0
    ut_sizing_max_base_multiplier: float = 5.0 # Cap raw risk sizing as a multiple of configured base quantity
    ut_strict_adx: bool = False      # Keep UT marker timing aligned with TradingView; ADX is graded later
    ut_smart_trailing: bool = True   # Trailing stop follows the trade's timeframe
    ut_regime_adaptation: bool = True # Enable dynamic regime adaptation for choppy markets
    ut_concurrency_guard: bool = True # Enable concurrency guard to prevent overlapping index trades
    ut_correlated_index_guard: bool = True # Limit same-direction exposure across highly correlated core indices
    ut_correlated_core_limit: int = 2 # Maximum simultaneous same-direction NIFTY/BANKNIFTY/SENSEX exposures
    ut_correlated_allow_aplus_override: bool = True # Permit a third core exposure only when every setup is A+
    ut_fractal_confluence: bool = True # Allow 1m entries if 5m trend aligns
    ut_session_filter: bool = True   # Enable session filtering in the engine
    ut_intel_early_exit: bool = True # Enable intelligence-based early exits
    # OI_PCR_REVERSAL must prove the book turned SINCE entry, not merely that it
    # leans against the trade now. Off = any qualifying reading exits, including the
    # first one an entry made before the day's option chain existed ever saw.
    intel_reversal_requires_new_evidence: bool = True
    ut_dynamic_confidence: bool = True # Enable dynamic confidence threshold
    ut_backtest_more_results: bool = True # Loosen historical-only gates for exploratory backtests
    futures_shadow_enabled: bool = False # Research-only FUT comparison rows for historical AUTO/Hybrid; never places live orders
    ut_session_start: str = "09:15"
    ut_session_end: str = "15:25"
    # ─── Operator session schedule (2026-08-11) ──────────────────────────────
    #   14:55  last 15min entry
    #   15:01  last 5min/1min entry, and the moment EOD protection engages
    #   15:15  full exit
    # The three are deliberately consistent: eod_protect_window_minutes is set so that
    # protection begins exactly when the last entry is allowed, rather than 30 minutes
    # before force-exit. Until 2026-08-11 the 5min cutoff (15:15) sat 20 minutes INSIDE
    # the protection window, so fresh positions were opened into a regime whose own EOD
    # logic closed them on the next tick — four such entries on 11 Aug round-tripped in
    # 0.06-0.19s, each paying ~Rs.80 of round-trip charges for a fraction of a second of
    # exposure, and across the stored book that window cost -Rs.8,864 over 13 trades.
    # System-wide last NEW entry per timeframe (2026-09-26 user): 15m 14:50 (entries stop
    # at 14:51), 5m 15:10, 1m 15:14, then the 15:15 hard exit. The lanes (UT SMART EXIT,
    # S1) apply them on the entry clock (engine/entry_window.py); the other engines keep
    # their bar-time check AND the EOD-protect clamp below, so their 5m/1m entries still
    # end when EOD protection starts (15:01).
    ut_no_entry_after: str = "14:50" # 15m last entry
    ut_5min_no_entry_after: str = "15:10" # 5m last entry
    ut_1min_no_entry_after: str = "15:14" # 1m last entry
    # Enforces the invariant above: whatever the two cutoffs say, no entry is admitted
    # once EOD protection is active. Off restores the previous behaviour of letting the
    # entry gate and the EOD exit logic contradict each other.
    block_entries_during_eod_protect: bool = True
    ut_force_exit_time: str = "15:15"
    # Upper bound on the force exit: from this IST time the exchange no longer accepts
    # F&O orders, so check_session_end and the expiry-day time exits stop sending
    # exits and the stuck position is escalated by name instead. Without it the square-off re-sent an order that could
    # only be rejected on every scan cycle for the rest of the evening. 15:40 since
    # SEBI's Closing Auction Session went live on 2026-08-03 (SEBI circular 16-Jan-2026
    # para 4.2.3: the equity derivatives segment runs to 3:40 p.m.); it was 15:30 before.
    fno_market_close_time: str = "15:40"
    # CARRYFORWARD -> Kite NRML, INTRADAY -> Kite MIS (_PRODUCT_MAP in kite_broker.py).
    # A single-member Literal until 2026-08-16, so MIS could not be selected at all even
    # though the broker mapped it. The DEFAULT is deliberately unchanged, because for a
    # long-premium book the choice is not about margin — buying an option costs the full
    # premium under either product, so MIS buys no leverage here. What differs is the
    # broker's own square-off: Zerodha force-closes MIS positions in the 15:20 window,
    # NRML it does not.
    #   NRML  no broker square-off, so no surprise market order into a thin 15:20 book —
    #         but if this system's own 15:15 force-exit fails to complete (crash, dead
    #         session, IP block) the position carries OVERNIGHT with gap risk.
    #   MIS   the broker becomes a 5-minute safety net behind ut_force_exit_time, at the
    #         cost of that net being a market order at whatever the book then shows.
    # Both are defensible; making it configurable is the fix, not picking for you.
    trade_product_type: Literal["CARRYFORWARD", "INTRADAY"] = "CARRYFORWARD"
    # Adaptive hold for bought options (engine/adaptive_hold.py). OFF by default: at
    # introduction (2026-09-11) its leave-one-day-out walk-forward LOST more than the
    # current exits -- 227 accepted trades over 35 days, -Rs.233/trade vs -Rs.153
    # (difference CI -153..-14) -- so re-run that test on newer data before enabling.
    # When on: cut a trade whose best gain is < never_green_min_r x initial risk at
    # never_green_minutes; once gain >= activate_r x risk, trail the stop at
    # peak - trail_atr_mult x premium ATR (widened by momentum_widen_atr while the index
    # keeps extending) and stand the profit-taking exits down. It releases the trade to
    # end-of-day handling at the EOD-protect start and never touches the 15:15 exit.
    # Refuse entries the whole option book already agrees with (no OI/PCR/flow pillar
    # against the trade). 2,348 live signals, 2026-07-20..09-10: those moved the trade's
    # way 45.8% of the time vs 54.1% for the rest, and lost Rs.301/trade on executable
    # option prices (CI -704..-72). Neutral in July-mid Aug, strong since. OFF by
    # default: it is anti-trend, and on the 2026-09-11 rally it would have refused
    # the 13:25 BANKNIFTY/SENSEX/MIDCPNIFTY LONGs and the 13:45/14:00 SENSEX 15min
    # LONGs while passing the 13:40 SHORTs that lost Rs.1.6-2.0k each.
    entry_block_unanimous_book: bool = False
    adaptive_hold_enabled: bool = False
    adaptive_hold_never_green_minutes: float = 5.0
    adaptive_hold_never_green_min_r: float = 0.15
    adaptive_hold_activate_r: float = 0.5
    adaptive_hold_trail_atr_mult: float = 3.5
    adaptive_hold_momentum_widen_atr: float = 1.5
    adaptive_hold_atr_period: int = 14
    live_filter_leniency_pct: float = 0.0 # Keep live gates strict; loosen only during deliberate diagnostics
    intelligence_cache_ttl_seconds: float = 90.0 # Option-chain cache TTL for live intelligence
    # How long a COMPUTED intelligence bundle stays usable. This used to be the line
    # above, which is a network budget: the option chain is a broker round trip, so
    # re-fetching it every few seconds is wasteful and 90s is right for it. The analysis
    # on top is candle math against an already-cached chain, and it was inheriting the
    # network's budget for no reason — leaving the evidence behind an entry decision up
    # to 90 seconds old, which on a 1min setup is a bar and a half.
    # Measured: 48ms per instrument-timeframe with a 40-strike chain, 17ms without. Only
    # the pair actually being read recomputes, not all twelve. At 30s that is well under
    # 1% duty. Lower it if you want fresher evidence and can spare the CPU; a genuinely
    # per-tick bundle is not on the table (all twelve pairs is ~580ms).
    intelligence_analysis_ttl_seconds: float = 30.0
    # Background refresh of each index's 1m/5m/15m bundles, all three together on one
    # spot/chain snapshot. Capped at 80% of the TTL above so the scan never finds an
    # expired bundle and recomputes it inline. In REAL mode a bar close on any of the
    # three frames also refreshes at once, and the scan treats a bundle computed before
    # its frame's last bar close as stale.
    intelligence_background_refresh_seconds: float = 20.0
    # Ceiling on the profit-retention ladder, as a fraction of peak P&L. The major-win
    # rung used to run to 1.0 at a Rs.5000 peak, which places the protective floor ON the
    # peak price — breached by the very next tick, so the "trailing lock" became a fixed
    # take-profit at Rs.5000. Bounded to [0.50, 0.99] in code; 1.0 is not selectable.
    # Matters more as size grows: at 10 NIFTY lots Rs.5000 is under Rs.7 of premium.
    # Breathing space the ladder leaves on a peak, as a band. The operator's rule:
    # 20-25% on the current tiers. The tier progression still decides WHERE in the band a
    # trade sits — more gain, tighter lock — but it can no longer squeeze past 20% or
    # loosen past 25%.
    #
    # Before this the ladder ran 41% breathing at a Rs.800 peak down to 5% at Rs.4,000+,
    # so a big winner was held on a 5% leash and a small one was given nearly half its
    # gain back. Both ends are now inside the band.
    # Outer bounds of the operator's breathing schedule (2026-08-08). These are a CLAMP,
    # not the schedule itself — the tiers in _profit_lock_floor set the shape:
    #   < Rs.1,000/lot   50% breathing   (a young winner needs room)
    #   Rs.1,000-2,000   35% -> 30%
    #   Rs.2,000-3,000   25% -> 20%
    #   Rs.3,000+        15% -> 10%      (protect a big gain)
    # The previous 0.75/0.80 pair flattened every tier to 20-25%, which is far too tight
    # on an index option at a Rs.600 peak.
    profit_lock_max_ratio: float = 0.90      # 10% breathing — the tightest allowed
    profit_lock_min_ratio: float = 0.50      # 50% breathing — the loosest allowed
    # Let the option STOP flex with the intelligence score, the way RR, target and exit
    # already do. Of those four only the stop ignored intelligence entirely, moving with
    # regime and ATR alone. Above the neutral 50 the stop widens slightly (corroborated
    # conviction earns room to be right); below it, it tightens.
    #
    # Default OFF: this changes the risk taken on EVERY option entry, which is a strategy
    # decision, not a defect being repaired. The shift is capped so a stop chosen by ATR
    # and the per-timeframe rules is nudged, never redefined.
    dynamic_stop_from_intelligence: bool = False
    dynamic_stop_intel_max_shift_pct: float = 20.0
    # Let measured market conviction move lot SIZE, not just lot COUNT. Cascade and
    # heavy-trend already scale the number of trades, but size itself was a
    # user-configured constant with only those coarse on/off multipliers.
    # Options only — FUT is hard-capped at fut_lots_hard_cap by operator rule. Only
    # ABOVE-neutral conviction adds size; shrinking is the entry filters' job.
    # Default OFF: sizing up is how a good idea becomes an expensive one.
    market_driven_lot_size: bool = False
    market_lot_size_max_boost_pct: float = 50.0
    market_lot_size_max_lots: int = 4
    # Let MACD acceleration decide how patient an ENTRY bid can be. The broker chooses
    # MARKET vs LIMIT on confidence alone — how good the setup is — and never on how fast
    # the premium is leaving the bid, so a strong signal into an accelerating move rested
    # a limit behind the market. Urgency RAISES the confidence bar a limit must clear; it
    # never lowers it, so this can only make the system less willing to wait, never more.
    # Default OFF: it changes fills, and a missed fill and a paid spread are both real.
    macd_aware_entry_pricing: bool = False
    macd_entry_urgency_max_penalty: float = 0.15
    # Treat intelligence that is DECAYING as a loss-guard signal, not just intelligence
    # that is low. Every pillar the guard reads is a point-in-time value; the snapshot
    # store has held the whole session's history and only a chart overlay ever read it.
    # A trade whose evidence has been sliding for a quarter hour is deteriorating even
    # while price sits still. Default OFF: it makes exits stricter.
    loss_guard_intel_decay_enabled: bool = False
    loss_guard_intel_decay_window_seconds: float = 900.0
    loss_guard_intel_decay_min_slope: float = -1.0   # points per minute
    loss_guard_intel_decay_severity: int = 1
    # ── scale-in (adding lots to a position that is already working) ─────────────────
    # Until now nothing anywhere increased an open trade's quantity: "cascade lot-adding"
    # opens SEPARATE trades across timeframes, which is a different thing.
    #
    # Default OFF. Turning it on changes how much money is at risk per idea, so it is an
    # explicit decision. The four design choices behind it, each made deliberately:
    #   * ONE shared stop, re-armed for the full quantity. Every exit path in this system
    #     assumes one position with one stop; giving a trade two stops would mean teaching
    #     all of them about legs.
    #   * The profit ladder reads the COMBINED position — so run this with
    #     profit_lock_per_lot_basis on, or adding size fakes a bigger win and the ladder
    #     scalps a trade that has barely developed.
    #   * WEIGHTED-AVERAGE entry price, with the original rupee risk FROZEN. Averaging up
    #     moves the stop distance, which would silently redefine 1R and change every
    #     R-based exit rule because of a sizing decision rather than a market move.
    #   * An add-on does NOT consume a fresh position slot (it is not a new trade) but it
    #     IS new money at risk, so it must pass the rupee-risk check and is refused when
    #     it does not fit.
    scale_in_enabled: bool = False
    scale_in_max_adds_per_trade: int = 2
    scale_in_lots_per_add: int = 1
    # An add-on is only allowed on a trade that is ALREADY working: this much of the
    # original risk in open profit, measured in R. Adding to a loser is averaging down.
    scale_in_min_profit_r: float = 0.75
    # ...and only while the regime still points the trade's way.
    scale_in_requires_trend: bool = True
    # Hard ceiling on total lots after adding, whatever the adds-per-trade cap allows.
    scale_in_max_total_lots: int = 6
    auto_option_chain_min_strikes: int = 3 # Minimum fresh active-expiry strikes required for OPT Auto
    # A confident TREND_DAY session plus an aligned breadth vote overrule a regime
    # label pointing the other way. On 2026-09-15 the label stayed TRENDING_UP until
    # 11:00 on a ~2% down day, costing with-trend SHORTs 0.55 of regime score and
    # refusing 36 of them below the B+ floor.
    regime_trend_day_contradiction_guard_enabled: bool = True
    regime_trend_day_contradiction_min_confidence: float = 0.70
    regime_trend_day_contradiction_min_breadth_votes: int = 3
    vix_percentile_min_sessions: int = 20 # Minimum daily India VIX observations before volatility routing is trusted
    vix_live_cache_seconds: float = 60.0 # Avoid repeated India VIX REST calls across index scans
    signal_vix_high_threshold: float = 20.0 # Configurable high-volatility grading threshold
    signal_vix_low_threshold: float = 12.0 # Configurable low-volatility grading threshold
    option_premium_cache_ttl_seconds: float = 2.0 # Open option premium refresh TTL
    index_ltp_cache_max_age_seconds: float = 6.0 # Scanner-loop index quotes must be this fresh; REST refresh happens off-loop
    contract_ltp_cache_max_age_seconds: float = 10.0 # Scanner-loop derivative quotes must be this fresh
    contract_ltp_background_workers: int = 2 # Bound daemon REST refreshes when live caches miss
    contract_ltp_background_retry_cooldown_seconds: float = 2.0 # Avoid 100ms retry storms after quote failures
    contract_ltp_background_result_limit: int = 256 # Bound completed background quote state
    fyers_chain_background_workers: int = 1 # Bound no-timeout Fyers option-chain SDK refreshes
    option_chain_background_workers: int = 2 # Bound full PCR/OI chain refreshes outside ScannerEventLoop
    option_chain_background_retry_cooldown_seconds: float = 5.0 # Avoid cold-chain retry storms after provider failure
    dynamic_subscription_background_workers: int = 2 # Bound slow WebSocket subscribe sends outside ScannerEventLoop
    history_cache_ttl_seconds: float = 240.0 # Broker history cache TTL for candles/volume
    # The REAL-session 1m history worker runs once a minute. Reusing the general
    # four-minute cache there makes every "fresh" backfill arrive 3-5 minutes late,
    # which the ghost guard correctly refuses. Keep the deep/history cache long, but
    # let the live 1m index edge refresh on each worker pass.
    # 45s against a bar that can only change once a minute meant ~1.15 history fetches
    # per index per minute. Those calls are serialized behind the single Fyers flight
    # (the SDK is not thread-safe and fyers.history has no transport timeout), and slow
    # scan minutes carried 1.57 of them against 0.91 on fast minutes. 60s aligns the
    # cache with the bar period. The entry-gate freshness window is unaffected: it is
    # max(180.0, ttl * 3.0), which is 180s at either value.
    live_1min_history_cache_ttl_seconds: float = 60.0
    # Minimum seconds between full chart rebuilds when only the forming bar ticked.
    # Structural changes (new candle, new signal, trade open/close) ignore this.
    chart_rebuild_min_interval_seconds: float = 2.0
    # Live index LTP fallback is AngelOne. Yahoo is delayed and therefore disabled by
    # default; retain an explicit disaster-recovery opt-in for operators who accept a
    # reference-only badge. Historical candle backfill is configured separately.
    yahoo_live_price_fallback_enabled: bool = False
    # Broker history request pacing (requests/second). Too aggressive a rate trips
    # AngelOne's history limit, which forces the Yahoo fallback — and Yahoo reports
    # volume=0 for Indian index tickers, killing every volume-gated entry path.
    history_rate_limit_per_second: float = 0.35
    option_history_miss_ttl_seconds: float = 900.0
    live_index_timeout_seconds: float = 9.0 # Per-index live scan budget before serving a stale fallback that cycle. Raised 6->9 so heavy indices (SENSEX/BANKNIFTY, large option chains) finish during volatile periods. NOTE: this is a CAP, not a wait — fast indices still return early, so the higher cap costs nothing on healthy cycles.
    # Keep the 1s heartbeat and trade monitor, but run full signal analysis only when
    # candle/data/settings inputs change or an explicit replay step requests it.
    event_driven_scan_enabled: bool = True
    kite_http_timeout_seconds: float = 7.0
    # Scanner-runtime and replay-robustness knobs moved to typed subsystem groups in
    # config.runtime_groups. Retired flat .env names remain readable there during the
    # compatibility window, but no longer inflate this global model.
    # Background broker position-reconciliation cadence. Base rate applies while
    # positions are open (need close-reconciliation); idle rate when flat (nothing
    # to reconcile); exponential backoff up to max when the broker API fails
    # (rate-limit/timeout). Stops the every-5s get_positions() from spamming a
    # rate-limited broker and starving the data-API budget.
    position_sync_interval_seconds: float = 5.0
    position_sync_idle_interval_seconds: float = 30.0
    position_sync_max_backoff_seconds: float = 60.0
    live_calculation_lock_timeout_seconds: float = 12.0 # Live lock recovery budget; historical work gets a larger derived budget
    scanner_exception_backoff_seconds: float = 1.0 # Fast recovery after an unexpected scanner-loop exception
    morning_refresh_parallelism: int = 3 # Bound broker pressure while warming independent indices
    # Morning refresh quality gate: only mark the day "complete" when thresholds pass.
    # Incomplete days keep missing_keys and can partial re-warm + 09:15 catch-up.
    morning_refresh_require_history: bool = True
    morning_refresh_require_expiry: bool = True
    morning_refresh_require_all_quotes: bool = True
    morning_refresh_require_all_chains: bool = True
    morning_refresh_require_vix: bool = False  # VIX often flaky pre-open; optional
    morning_refresh_catchup_at_live_open: bool = True  # second pass at/after 09:15 if incomplete
    morning_refresh_max_attempts: int = 3  # full+partial retries per session day
    runtime_cache_prune_interval_seconds: float = 60.0 # Physically evict expired quote/decision cache entries
    trade_state_checkpoint_seconds: float = 1.0 # Coalesce mark-to-market persistence while keeping crash recovery fresh
    trade_stagnation_minutes: float = 20.0 # Minimum profitable hold before stagnation exit can trigger
    trade_stagnation_max_extra_minutes: float = 10.0 # Extra breathing time for volatile indices/regimes
    trade_stagnation_min_peak_r: float = 0.50 # Require meaningful peak profit relative to initial risk
    circuit_breaker_slippage_bps: float = 10.0 # Conservative projected exit slippage for open-position loss checks
    position_sizing_min_atr_fraction: float = 0.10 # Stop-distance floor used for sizing only
    position_sizing_min_price_fraction: float = 0.00005 # Absolute fallback when ATR is unavailable
    live_signal_stabilization_seconds: float = 15.0 # Legacy/default anti-flicker window for non-5m/15m paths (15s, 2026-10-09 user)
    # Floor for option signals in a risky regime (volatile/ranging); was a hard-coded 30s.
    live_signal_stabilization_risky_floor_seconds: float = 15.0
    # Anti-repaint buffer before a forming signal may be entered. 30.0 -> 15.0 on
    # 2026-09-01.
    #
    # Mined from 24 sessions of live logs, 3,604 recorded repaints ("UT1 arrow vanished
    # after Ns"), the distribution is tight and stable:
    #
    #     p50 3.7s   p90 16.0s   max 62.9s
    #      5s window catches 61.2%      20s catches 93.6%
    #     10s catches 80.9%             30s catches 98.8%   <- the old value
    #     15s catches 88.8%
    #
    # Every one of the 24 sessions lands between 84% and 94% caught at 15s -- no session
    # is an outlier, which is the session-clustered robustness that two earlier threshold
    # proposals in this codebase failed to show.
    #
    # THE COST, stated because only the timing half is measured: repaints reaching the
    # downstream gates rise from ~1.9/session at 30s to ~16.8/session at 15s, against
    # roughly 10 accepted entries per session. Most are refused further down, but how
    # many become bad ENTRIES is NOT measured -- no outcome data exists for signals the
    # repaint gate refused. 10s was requested and is one edit away (80.9%,
    # ~28.8/session); it was not taken because it costs another 12 slipped repaints per
    # session for 5 more seconds, with the cost side still unmeasured.
    #
    # Measured effect on the wait: the floor was 33s bar-open-to-entry (30s window +
    # scan); it becomes ~18s. Median was 90s first-appearance-to-accept.
    live_signal_stabilization_seconds_1min: float = 5.0 # Repaint-check buffer before a qualifying 1min live entry (2026-10-09 user)
    live_signal_stabilization_seconds_5min: float = 15.0 # Repaint-check buffer before 5min live entry filtering
    live_signal_stabilization_seconds_15min: float = 15.0 # Repaint-check buffer before 15min live entry filtering
    # Re-score a buffered signal's intelligence and regime pillars when it matures, on
    # the bundle that exists then, instead of letting the live gate read the grade the
    # signal was born with 20-60s earlier. The technical pillar is not re-scored: it
    # describes the signal bar, which has not changed. Audited 2026-09-18 as the gap
    # closest to "analyse during the buffer, then grade".
    stabilization_regrade_enabled: bool = True
    # The operator's 1-minute fast lane. A raw 1min arrow normally needs a volume burst,
    # a post-win snapback or a confirmed cascade, and with that requirement only 4 raw
    # 1min signals reached the buffer in 52 sessions - so there is no sample on which to
    # judge the open lane, which is why this ships OFF. When ON, such a signal buffers on
    # the short 1m window (volume_spike_entry_stabilization_seconds, 3-15s), not the 30s
    # 5m/15m wait, and every downstream gate still applies.
    one_min_raw_entry_enabled: bool = False
    midcap_15min_risky_stabilization_seconds: float = 60.0 # Extra buffer for MIDCPNIFTY 15min risky option signals in chop/weak tape
    midcap_15min_risky_stabilization_min_scans: int = 3 # Require repeated confirmation when MIDCPNIFTY 15min volume/intel is weak
    midcap_15min_risky_stabilization_min_volume_ratio: float = 1.05 # Basic volume floor for MIDCPNIFTY 15min risky stabilization
    midcap_15min_risky_stabilization_min_intel_score: float = 0.35 # Absolute signed-intel floor to skip the extra MIDCPNIFTY buffer
    live_15min_entry_cutoff_grace_seconds: float = 60.0 # 15m entries stop one minute after ut_no_entry_after (14:51)
    live_intrabar_signal_entries: bool = True # Detect forming-candle UT1 arrows before candle close
    # ─── Late-Session Power-Move Override ────────────────────
    # A data-confirmed "big move" may bypass the grade floor (any time) and the
    # 15:00 no-entry cutoff (after 15:00, up to the hard cutoff), with tight scalp-lock risk.
    power_move_override_enabled: bool = True
    power_move_volume_mult: float = 2.5      # candle volume >= this x recent average
    power_move_range_atr_mult: float = 1.8   # candle true range >= this x ATR (p95 of measured 5-min range/ATR)
    power_move_close_location: float = 0.70  # directional close >= this fraction of candle range
    power_move_adx_threshold: float = 20.0   # momentum: ADX >= this (lower than general 25; ADX lags spikes)
    power_move_momentum_volume_mult: float = 3.5 # momentum substitute: volume >= this x avg counts as momentum
    power_move_volume_lookback: int = 20     # candles for the volume average
    # A power move is the one lane that BYPASSES ut_no_entry_after entirely (see
    # power_move_cutoff_ok in signal_processor), so this is its only cutoff. Held at the
    # 5min entry time: at 15:15 it would have admitted a fresh entry at the very moment
    # of force-exit, which is the instant round-trip the 2026-08-11 schedule change
    # exists to stop — the bypass has to respect the schedule or it re-opens the hole.
    power_move_late_hard_cutoff: str = "15:01" # no power-move entry after this (= the 5min cutoff)
    # Fast 1min/5min volume-spike entry. 1min scalp entries can be disabled
    # separately without disabling 1min tracking/trailing/emergency context.
    # When 1min scalping is OFF, post-exit reversion can still admit 1min rows
    # only inside an open reversion window (decoupled toggles).
    volume_spike_entry_enabled: bool = True
    one_min_scalping_enabled: bool = True
    # 1min continuous scalps require high-burst/spike context (ADX/intel/power)
    # plus volume/range/close floors — not weak 1min noise.
    one_min_scalping_high_burst_only: bool = True
    # A live 1m entry must be backed by the current bar and a recent candle receipt.
    # Historical REST rows may remain useful for baselines, but cannot impersonate the
    # live edge merely because they were downloaded a moment ago.
    one_min_entry_max_bar_age_seconds: float = 90.0
    one_min_entry_max_receipt_age_seconds: float = 30.0
    one_min_entry_max_provider_lag_seconds: float = 90.0
    # A cash index publishes no traded volume in real time: measured 13 of the last 20
    # 1min bars carry volume 0 on every index (5min 1/20, 15min 0/20), because it only
    # arrives with the periodic Fyers backfill. The signal bar of a fresh 1min entry is
    # therefore almost always unmeasurable, and the surge ratio read 0.00x — which is
    # why no 1min trade has ever fired. tick_count is the only live participation
    # measure available. This is enabled because the threshold below was calibrated on
    # recorded live tick data. Tick counts are far less dispersed than traded quantity,
    # so the traded-volume ratio floor is deliberately NOT reused on them.
    surge_tick_proxy_enabled: bool = True
    # Calibrated on the real 1min candles, not guessed: ratio>=2.2 admits 4.73% of
    # measurable traded-volume bars and rank>=97 admits 4.98% — matched strictness in a
    # form that survives the change of measurement basis.
    surge_tick_percentile_floor: float = 97.0
    volume_spike_entry_min_volume_ratio: float = 2.5
    # Per-index volume-ratio floors, because a flat 2.5x is not one filter but four.
    # Measured over 24 sessions (2026-07-14..08-14, 1,650 five-minute bars per index)
    # against the profiler's own time-of-day baseline, 2.5x sits somewhere completely
    # different in each index's distribution:
    #
    #     index        median   p90    p95    p99  | flat 2.0x fires on
    #     NIFTY          0.94   1.42   1.63   2.03 |   1.3% of bars
    #     BANKNIFTY      0.83   1.70   2.14   3.76 |   6.2% of bars
    #     MIDCPNIFTY     0.88   1.70   2.13   4.07 |   6.3% of bars
    #     SENSEX         0.74   2.35   3.71   9.85 |  12.4% of bars
    #
    # (2.0x is the live .env value; the 2.5 default above is the shipped one. The shape
    # is the same either way — at 2.5x it is 0.5% against 9.3%.)
    #
    # Nine to eighteen times more often on SENSEX than on NIFTY, from one number. SENSEX is the
    # thinnest of the four by absolute volume (median 137k a bar against NIFTY's 3.5m)
    # and thin books print lumpily, so its ratios scatter far wider.
    #
    # These defaults are each index's own 95th percentile, putting the spike lane on the
    # top 5.0% of bars on every index rather than 1.3% on one and 12.4% on another.
    #
    # NIFTY is held at the live 2.0 rather than its p95 of 1.63, because nothing here
    # should make this lane MORE active. Tested in the other direction — starting from
    # the moves rather than from the volume — a spike does not precede a big move on any
    # of the four. Median ratio of the top 1% of forward moves:
    #
    #     NIFTY 1.01   BANKNIFTY 0.92   SENSEX 0.74   MIDCPNIFTY 0.73
    #
    # At or BELOW average volume. Sweeping every threshold from 0.8x to 2.5x, the share
    # of big moves retained divided by the share of all bars retained sits at ~1.00
    # throughout: no better than random. It degrades at the top — MIDCPNIFTY 0.19 at
    # 2.0x and 0.00 at 2.5x, NIFTY 0.00 at 2.5x, SENSEX below 1.00 at every level.
    #
    # So this is not a filter that catches real moves, and loosening it would only add
    # entries with no measured edge. It stays selective for that reason, not because
    # volume is informative. Volume never blocks a normal entry in this system — the
    # stabilization thrust only SHORTENS a wait and is OR'd with range expansion — so a
    # tight bar here costs no setups.
    #
    # The one-directional read (volume -> move) gave BANKNIFTY 1.64x lift at >=2.5x, but
    # the reverse read (move -> volume) gives it nothing, so treat even that as
    # unconfirmed until live fills say otherwise.
    #
    # Empty dict = nothing is per-index and the flat setting applies everywhere, which
    # is the pre-2026-08-15 behaviour.
    volume_spike_min_ratio_by_index: Dict[str, float] = Field(
        default_factory=lambda: {
            "NIFTY": 2.00,
            "BANKNIFTY": 2.14,
            "MIDCPNIFTY": 2.13,
            "SENSEX": 3.71,
        }
    )
    volume_spike_entry_strong_1min_min_volume_ratio: float = 2.5
    volume_spike_entry_strong_1min_adx_threshold: float = 25.0
    volume_spike_entry_strong_1min_intel_threshold: float = 0.70
    volume_spike_entry_min_range_atr_ratio: float = 1.5
    volume_spike_entry_min_close_location: float = 0.65
    volume_spike_entry_futures_sl_pct: float = 0.05
    volume_spike_entry_options_sl_pct: float = 5.0
    volume_spike_entry_min_rr: float = 2.5
    volume_spike_entry_1min_scalp_rr: float = 1.5
    # 1M SPIKE anti-repaint buffer. Aggressive floor = 3s (configurable in advanced
    # settings). 5m/15m use live_signal_stabilization_seconds_* (30–60s) separately.
    volume_spike_entry_stabilization_seconds: float = 3.0
    volume_spike_entry_stabilization_seconds_min: float = 3.0
    volume_spike_entry_stabilization_seconds_max: float = 15.0
    # When L1 depth/OFR is usable, 1min spikes must align with book pressure
    # (BUY needs OFR >= bullish floor; SELL needs OFR <= bearish ceiling).
    # If depth is unavailable, spike still qualifies on candle thrust alone.
    volume_spike_depth_confirm_enabled: bool = True
    volume_spike_depth_ofr_bullish: float = 1.08
    volume_spike_depth_ofr_bearish: float = 0.92
    stabilization_momentum_override_enabled: bool = True # Re-admit flickered signals when thrust keeps confirming direction
    # Drop the weakest branch of that override without disabling the whole mechanism.
    # The override is a chain of ALTERNATIVES (first match wins), and "regime flip" —
    # trending regime pointing the same way as the signal, with no price, volume or
    # structure test — accounted for 653 of 1327 re-admissions across every session log
    # (49%), versus 3 from the strict directional-thrust branch. Turning this ON keeps
    # thrust, cascade, late-reclaim, market-wide-reversal, VWAP and structure while
    # removing the branch that is nearly always true in a trending tape.
    # Default True: regime alignment alone is lagging context, not reversal proof.
    # Strict thrust, cascade, late-reclaim, market-wide, VWAP and structure routes stay.
    stabilization_override_drop_regime_flip: bool = True
    # Refuse a LONG while a SHORT is live on the same index (and vice versa). The rest of
    # the concurrency guard reasons about timeframe, lane, instrument type and count and
    # never once about DIRECTION, so with ut_concurrency_guard off the only limit on
    # holding both sides is max_trades_per_index. Options here are always BOUGHT, so a
    # long CE beside a long PE is two premiums and two spreads with theta against both —
    # a straddle assembled by accident from two signals that each looked fine alone.
    # Default ON: a reversal that arrives AFTER the first trade exits is untouched, and a
    # deliberately armed leg (pullback hedge, dual-lane reversal) is exempt. Turn it off
    # only to deliberately run both sides of an index at once.
    block_opposite_direction_stack: bool = True
    # Read the profit-retention ladder PER LOT instead of in absolute rupees. Its tiers
    # (500 / 1000 / 2000 / 3000) were tuned on single-lot trades, so on 3 lots a Rs.3000
    # peak — Rs.1000 per lot, a modest win — lands on the Rs.3000 MAJOR_WIN_GUARD rung and
    # locks 90%, scalping a position that has barely developed.
    # Per-lot is LOOSER, not safer: a multi-lot winner lands on a lower rung and is
    # allowed to give back more before the lock fires. That is a real risk increase on
    # live money, which is why it sat OFF and was left to the operator.
    #
    # Turned ON 2026-08-14, by the operator, because the pairing it was waiting for is
    # already live and the mismatch was costing the exact trades it was meant to protect:
    # cascade_lot_scale_enabled defaults True with cascade_lot_multiplier_opt 3.0 (floor
    # 2.0), so cascade OPTION signals size 2-3 lots today. With the ladder reading
    # absolute rupees against rungs tuned on single-lot trades, a 3-lot position peaking
    # at Rs.3,000 -- Rs.1,000 per lot, a modest win -- landed on the Rs.3,000
    # MAJOR_WIN_GUARD rung and locked 90%, scalping the high-conviction multi-lot
    # entries. The scale-in note at line 1623 already said to run adds with this on.
    #
    # Coherent by construction: _profit_ladder_pnl divides the peak by lots and
    # _profit_lock_arm_cap_rs multiplies the arm ceiling by lots, both under this one
    # switch, so the tier and the threshold feeding it stay on the same scale. Setting
    # it False restores the previous behaviour exactly.
    profit_lock_per_lot_basis: bool = True
    # Refuse an entry that would push the book past its stop-risk budget. The daily
    # circuit breaker measures a book that already exists and halts after the fact; this
    # refuses the entry that would create the breach, using the SAME segment and global
    # budgets so the two cannot drift apart.
    # Off by default because at one or two concurrent positions it duplicates the breaker.
    # It earns its keep once several lots run across several indices with cascade scaling
    # on — which is exactly when the breaker's after-the-fact halt arrives too late.
    entry_risk_budget_enabled: bool = False  # Optional for simulation. Required for live broker exposure.
    historical_quad_source_attribution: bool = False
    # Hold a MARGINAL profit-taking option exit for a few seconds while the quoted spread
    # is wider than the edge being captured — on an index option, exiting at the bid
    # routinely converts a small win into a small loss.
    # Off by default: it DELAYS an exit, and every deferral is a risk the position moves
    # against you while it waits. Bounded by spread_exit_deferral_max_seconds, restricted
    # to profitable positions and to the reasons in Scanner._MARGINAL_EXIT_REASONS, and
    # bypassed entirely by every protective/loss/forced/expiry/broker/manual exit and by
    # an exchange stop that is already armed and crossed.
    # Have the 15:25 force exit also square off BROKER-ONLY positions this system did not
    # open. Off by default: an orphan may be a position taken manually at the Zerodha
    # terminal, and squaring one off uninvited is worse than the exposure it removes.
    # Left off, the session end escalates instead — the position still carries overnight,
    # which CARRYFORWARD/NRML means the broker will not prevent either.
    # Turn it on only if you never hold manual positions in this account.
    force_exit_flattens_orphans: bool = False
    spread_aware_exit_enabled: bool = False
    spread_exit_deferral_min_spread_pct: float = 0.35   # below this the book is fine
    spread_exit_min_net_profit_rs: float = 250.0        # bid-side profit worth waiting for
    spread_exit_max_cost_to_profit_ratio: float = 0.35  # spread cost vs the edge
    spread_exit_deferral_max_seconds: float = 8.0       # hard ceiling on any hold
    # Tape hygiene: a 1min signal can ONLY enter via the volume-spike/scalp
    # filter, so 1min signals that never cleared it cannot trade. When True, the
    # raw-detection and rejection lines for such no-spike 1min signals are kept off
    # the live tape (downgraded to a debug file log); reject diagnostics still count.
    live_1min_log_only_volume_spike: bool = True
    # Late-session synchronized move: catches broad 1m+5m directional flips after
    # a range/chop tape only when the basket proves reclaim/rollover structure.
    # This is narrow by time, direction, breadth, and scalp-risk settings.
    late_session_reclaim_override_enabled: bool = True
    late_session_reclaim_start: str = "14:25"
    late_session_reclaim_end: str = "15:10"
    late_session_reclaim_core_votes: int = 3
    late_session_reclaim_total_votes: int = 3
    late_session_reclaim_reclaim_votes: int = 2
    late_session_reclaim_max_skew_seconds: float = 240.0
    late_session_reclaim_min_close_location: float = 0.55
    late_session_reclaim_min_volume_ratio: float = 1.10
    late_session_reclaim_volume_votes: int = 3
    late_session_reclaim_options_sl_pct: float = 5.0
    late_session_reclaim_futures_sl_pct: float = 0.05
    late_session_reclaim_rr: float = 1.5

    repaint_check_interval_minutes: int = 2 # (legacy/unused) periodic interval for repaint guard checking
    repaint_check_interval_seconds: float = 20.0 # Intra-candle repaint re-check cadence (5min & 15min) until candle close
    repaint_close_check_grace_seconds: float = 90.0 # Post-close window in which a repaint may still abort; after this the candle is confirmed and the trade is tracked normally
    repaint_recheck_seconds: float = 20.0 # Post-entry repaint re-check cadence until the source candle closes (same for 5min and 15min)
    # ── Opposite-flip confirm window (2026-07-10) ──
    # 5min/15min flips persist 30–60s (clamped) AND pass an intel/move sanity read.
    # 1min reversals cap both the repaint window and extended intel hold at 15s.
    # before it exits a live trade and hands the slot to the reverse entry.
    # Multi-index LTP shock force-exit bypasses this window by design.
    opposite_flip_confirm_min_seconds: float = 30.0
    # Normal QP reversals belong to the entry timeframe. Keep the old policy
    # available for controlled policy comparison; emergency shock exits are independent.
    qp_cross_timeframe_opposite_exits: bool = False
    opposite_flip_confirm_max_seconds: float = 60.0
    opposite_flip_intel_filter_enabled: bool = True
    # Hold confirm while the flip's own move has gone adverse by more than this
    # (%) — a flip that immediately mean-reverts is not a real reversal. At the
    # max window the hold expires: adverse ⇒ cancel timer, else confirm.
    opposite_flip_max_adverse_move_pct: float = 0.10
    # Strong opposing intelligence (aligned score <= -threshold) delays confirm
    # until the max window, then price action decides.
    opposite_flip_intel_block_threshold: float = 0.55
    live_signal_recheck_seconds: float = 5.0 # Re-grade a rejected forming arrow at a controlled cadence, plus once at candle close
    last_minute_extension_seconds: int = 60 # Check last minute signal entry extension
    live_restart_recovery_grace_seconds: float = 120.0 # Fallback restart-recovery window (non-5/15min TFs)
    live_restart_recovery_grace_seconds_5min: float = 180.0  # Restart recovery window for 5min last-candle signal
    live_restart_recovery_grace_seconds_15min: float = 360.0 # Restart recovery window for 15min last-candle signal
    live_signal_post_close_grace_seconds_5min: float = 300.0 # Let an already-running scanner finish slow 5min refreshes
    live_signal_post_close_grace_seconds_15min: float = 300.0 # Let an already-running scanner finish slow 15min refreshes
    live_choppy_gate_confidence: float = 0.90 # Choppy/ranging setups need exceptional confidence
    # ─── Live quality-gate confidence floors (promoted from getattr-defaults to real fields) ───
    live_15min_min_confidence: float = 0.62
    live_15min_trending_momentum_adx_threshold: float = 24.0
    live_15min_trending_momentum_intel_threshold: float = 0.60
    live_15min_trending_momentum_min_confidence: float = 0.58
    live_15min_momentum_override_enabled: bool = True
    live_15min_momentum_min_confidence: float = 0.20
    live_15min_momentum_volume_ratio: float = 2.0
    live_15min_momentum_range_atr_ratio: float = 1.25
    live_15min_momentum_close_location: float = 0.65
    live_15min_momentum_adx_threshold: float = 24.0
    live_15min_momentum_intel_threshold: float = 0.65
    fifteen_min_choppy_scalp_guard_enabled: bool = True
    fifteen_min_choppy_scalp_min_intel_score: float = 0.35
    fifteen_min_choppy_scalp_min_volume_ratio: float = 1.05
    fifteen_min_choppy_scalp_micro_max_skew_seconds: float = 240.0
    fifteen_min_choppy_scalp_min_close_location: float = 0.55
    fifteen_min_choppy_scalp_profit_lock_r_multiple: float = 0.20
    fifteen_min_choppy_scalp_min_peak_rs: float = 250.0
    live_5min_bplus_min_confidence: float = 0.70
    live_5min_choppy_min_confidence: float = 0.70
    live_5min_special_adx_threshold: float = 30.0
    live_5min_special_bplus_min_confidence: float = 0.60
    live_5min_special_intel_threshold: float = 0.75
    live_5min_special_max_iv_percentile: float = 70.0
    live_5min_special_min_chain_score: float = 50.0
    reversal_entry_relief_enabled: bool = True
    reversal_entry_min_confidence: float = 0.20
    # Stale-regime reversal (2026-09-17). A 5m/15m signal AGAINST a trending regime label
    # whose own session extreme is at least this many minutes old: the label is still
    # "up" but price stopped making highs half an hour ago. Every grading pillar lags in
    # that state, so these signals grade C and every veto reads them as counter-trend.
    # Measured on 1,519 non-repainted live signals over 22 sessions (entry at the signal
    # bar close, index points): +0.77 ATR at 30m, 59% positive, 13/21 days; both
    # walk-forward halves positive; +0.47 ATR de-duplicated. The same staleness filter
    # adds nothing to ranging (+0.11) or aligned (-0.12) signals.
    #
    # SWITCHED OFF 2026-09-18 after one live session. The backtest above is not wrong, but
    # it measured the SIGNAL, not the route, and the route as shipped carried no floor of
    # its own: it supersedes the B+ gate, so a signal that every other check refuses is
    # admitted on the staleness fact alone. Live result, MIDCPNIFTY 15min, all three
    # through route=stale_regime_reversal, all three losses totalling -Rs.2,478:
    #   10:32 C 19%   11:05 C 19%   11:17 C 32%
    #   confirmation_count 0 on all three, every leg false (trend_15m, impulse, volume,
    #   multi_source), volume 0.43-0.64x against a 1.2x floor, source_count 1,
    #   market_shift_core_votes 0, price_action_verdict ADVERSE on two, ADX 32-36 —
    #   measured, strong, and pointing the OTHER way.
    # The staleness test is also a one-way latch: MIDCPNIFTY set its session high at 09:24,
    # so "extreme older than 30 minutes" was true from 09:54 until the close and never
    # expired. It is a statement about the CLOCK, not about price reversing. All three
    # shorts landed within 6 points of the session low in a 14,456-14,474 chop, and price
    # was higher 15 minutes later every time (+6.25, +3.90, +3.70).
    # Do not switch this on again without: a confirmation floor inside the route, a
    # staleness test that requires price to have actually turned, and a re-entry cooldown.
    stale_regime_reversal_enabled: bool = False
    stale_regime_reversal_min_minutes: float = 30.0
    reversal_entry_thrust_min_range_atr_ratio: float = 1.20
    reversal_entry_thrust_min_close_location: float = 0.62
    reversal_entry_thrust_min_volume_ratio: float = 1.20
    reversal_entry_vwap_min_dev_pct: float = 0.03
    directional_census_reversal_relief_enabled: bool = True
    directional_census_reversal_min_confidence: float = 0.20
    late_session_choppy_entry_guard_enabled: bool = True
    late_session_choppy_min_runway_minutes: float = 30.0
    market_wide_reversal_enabled: bool = True
    market_wide_reversal_core_votes: int = 2
    market_wide_reversal_total_votes: int = 2
    market_wide_reversal_min_range_atr_ratio: float = 1.10
    market_wide_reversal_min_range_atr_ratio_5min: float = 1.10
    market_wide_reversal_min_range_atr_ratio_15min: float = 0.50
    market_wide_reversal_midcap_min_range_atr_ratio_5min: float = 0.65
    market_wide_reversal_midcap_min_range_atr_ratio_15min: float = 0.45
    market_wide_reversal_min_close_location: float = 0.65
    market_wide_reversal_max_skew_seconds_5min: float = 180.0
    market_wide_reversal_max_skew_seconds_15min: float = 900.0
    # Completed bars of the SAME session that must precede the bar a vote measures.
    #
    # A vote is "range >= k x ATR14 of the prior bars, close in the top 35%". At the open
    # the prior bars are the previous session's closing bars, which are the quietest of the
    # day, and the measured bar is the opening auction that absorbed the overnight gap. So
    # any gap that has not faded yet scores as a market-wide thrust. 2026-09-18 09:17:30:
    # SENSEX gapped +260 (74,314.59 -> 74,575.24), the 2.5-minute-old 09:15 15min bar read
    # 1.39x ATR / close 0.69, BANKNIFTY 1.99x / 0.98, MIDCPNIFTY 2.39x / 0.96, and "core 3/3"
    # carried a RANGING, ADX-14.9 B+ LONG past the floor that had refused the same signal
    # two minutes earlier. The gap faded 77 points the next minute; cut at 69s, -Rs.599,
    # never green. Market-shift entries before 09:30 in the trade table: 6, 1 winner,
    # -Rs.1,525. Three bars gives the ATR a same-session reference: 15min votes resume at
    # the 10:00 bar, 5min at 09:30.
    market_wide_reversal_min_session_bars: int = 3
    momentum_override_threshold: float = 25.0 # ADX threshold to bypass choppy market block
    max_trades_per_index: int = 5
    # Dynamic daily trade budget: scale max_trades_per_index by the current
    # intraday phase. A confirmed trend phase earns continuation re-entries;
    # a confirmed range/chop phase trims the budget whether it lasts all day,
    # an hour, or only the current 40-minute pocket.
    dynamic_trade_budget_enabled: bool = True
    dynamic_trade_budget_trend_bonus: int = 3
    dynamic_trade_budget_range_malus: int = 2
    dynamic_trade_budget_min_day_confidence: float = 0.60
    # Multi-entry engines (quad_prime/quad_union/adaptive_hybrid) legitimately accept more entries
    # per day in trend/normal phases. Never offsets the range-phase malus: raw
    # arrow density in chop is churn for every engine. Hard ceiling stays 20.
    dynamic_trade_budget_multi_engine_bonus: int = 4
    # Session trend-day guard: block weak 1min/5min counter-trend scalps while a
    # high-confidence trend day is in force (fast-thrust and reversal-relief
    # entries are exempt; guard disarms once price closes across session VWAP).
    session_context_trend_day_guard_enabled: bool = True
    session_context_trend_day_guard_min_confidence: float = 0.70
    # Census-contradiction yield: on 2026-07-08 the day-type label stayed
    # TREND_DAY_UP and vetoed 6 midcap SELLs (11:19-12:36) while the session's
    # own directional census was already 5L/10S bearish; midcap then fell 1.9%.
    # When the census opposes the label this strongly, the label is stale —
    # the guard yields instead of vetoing.
    session_trend_guard_census_yield_enabled: bool = True
    session_trend_guard_census_min_total: int = 6
    session_trend_guard_census_dominance: float = 1.5
    # Bidirectional late-session market LTP shock (dump AND rally).
    # Multi-index ≥0.20–0.35% move in ≤3–5 min + volume → short stab OPT path;
    # after session_override_start the sticky TREND_DAY_* label yields so the
    # regime panel is not stuck on the full-day open classification.
    market_ltp_shock_enabled: bool = True
    # Measured on 19,984 three-minute windows, four indices, 13 sessions:
    #   median 0.022%   p90 0.065%   p99 0.161%   max 0.978%
    # The old 0.20% floor sat ABOVE the 99th percentile, so this lane fired zero
    # times on 2026-08-13 — a session in which NIFTY moved +0.32% and SENSEX
    # +0.35%. A detector that never fires is an absence, not protection. 0.12% is
    # the p97-98 of that distribution: still unusual, not noise.
    #
    # Safe only because the multi-index requirement below bounds it — >=2 core
    # indices moving the same way inside the same window:
    #   >=0.20%  ~2.3 openings/day     >=0.12%  ~6.3 openings/day
    # and each still faces the quality floor, volume ratio and grade gates.
    market_ltp_shock_min_move_pct: float = 0.12
    market_ltp_shock_strong_move_pct: float = 0.35
    market_ltp_shock_lookback_minutes: int = 5
    market_ltp_shock_min_lookback_minutes: int = 3
    market_ltp_shock_min_indices: int = 2
    market_ltp_shock_min_core_indices: int = 2
    market_ltp_shock_min_volume_ratio: float = 1.10
    # Minimum canonical confidence for a strict multi-index market-shift route
    # to supersede stale regime/15-minute direction. Ordinary signals do not use it.
    quality_policy_market_shift_min_confidence: float = 0.60
    # Whether the confirmed_market_shift route exists at all. OFF since 2026-09-23: 12 real
    # trades through it, 3 winners, -Rs.2,447; on the executable shadow ledger its accepted
    # signals averaged -277 at 30 min and -574 at 60 min (negative in both halves) against
    # +906 / +440 for the normal route. See engine/quality_verdict.py.
    quality_policy_market_shift_route_enabled: bool = False
    # Measured-volume floor the route applies when it is switched on (RP50's value). The
    # route waives the regime/ADX/confirmation fence, so it must at least see real
    # participation; tick proxies and unavailable volume never count.
    quality_policy_market_shift_min_volume_ratio: float = 1.05
    # A fully synchronous 5m reversal across all three core indices has stronger
    # contemporaneous evidence than a generic market shift.  It may use this lower
    # floor only when its detector recorded all core votes and measured ADX.  This
    # does not relax ordinary range/chop arrows or 1m/15m entries.
    quality_policy_full_core_market_shift_min_confidence: float = 0.35
    quality_policy_full_core_market_shift_min_adx: float = 18.0
    # Shock engine full-sensitivity IST window (normal min_move applies inside).
    market_ltp_shock_session_start: str = "13:00"
    market_ltp_shock_session_end: str = "15:15"
    # Backward-compat alias used by older guards (same as session_start).
    market_ltp_shock_session_override_start: str = "13:00"
    # Outside the window: "strong" = only moves >= outside_strong_move_pct qualify
    # (genuine crash/melt-up bursts); "off" = engine idle outside the window.
    market_ltp_shock_outside_window_mode: str = "strong"
    market_ltp_shock_outside_strong_move_pct: float = 0.35
    market_ltp_shock_opposite_trade_relax: bool = True
    # When True and multi-index LTP shock is confirmed, immediately force-exit
    # open opposite positions (skip ~20s opposite-flip repaint wait) so reverse
    # OPT/FUT can enter the same scan cycle with opposite_trade_relax.
    market_ltp_shock_force_exit_opposite: bool = True
    market_ltp_shock_stabilization_seconds: float = 8.0
    market_ltp_shock_allow_1min: bool = True
    market_ltp_shock_allow_5min: bool = True
    # Rolling mood supersedes sticky full-day day_type for trend-phase guard.
    session_mood_dynamic_enabled: bool = True
    session_mood_min_confidence: float = 0.55
    # Previously referenced via getattr defaults only — declared so the operator
    # can actually toggle/tune them (audit 2026-07-03 found them undeclared).
    ut_market_location_gate: bool = True
    live_5min_special_min_confidence: float = 0.60
    ut_historical_verbose_audit_logs: bool = False
    # Angel REST exponential backoff after throttle/failure responses:
    # delay = min(max_seconds, seconds * 2^(failures-1)).
    angel_rest_backoff_seconds: float = 12.0
    angel_rest_backoff_max_seconds: float = 60.0
    # Kite REST throttle backoff, same ladder. Much shorter than the Angel figures above
    # because Kite's published limits are per-SECOND (historical 3/s, quote 1/s), so a
    # 12s floor would idle the data path for whole scan cycles to clear a sub-second
    # window. Applied per endpoint, and only on a recognised rate-limit refusal.
    kite_rest_backoff_seconds: float = 1.0
    kite_rest_backoff_max_seconds: float = 8.0
    # Candle REST is an independent capability from the live quote socket. A
    # successful call older than this is stale evidence for historical-dependent
    # live decisions and is reported unhealthy even while LTP ticks remain fresh.
    kite_historical_rest_max_age_seconds: float = 90.0
    # ── 2026-07-03 session-audit fixes ──
    # Bounded option-quote retry before dropping a live candidate (transient
    # chain hiccups cost 16 tradeable candidates that session).
    option_quote_retry_attempts: int = 2
    option_quote_retry_delay_seconds: float = 0.5
    # Fyers self-heal: re-create the client after a persistent option-chain
    # failure streak (the manual midday restart that cleared ~1600 chain errors
    # was effectively this).
    fyers_chain_self_heal_enabled: bool = True
    fyers_chain_reinit_failure_threshold: int = 8
    fyers_chain_reinit_cooldown_seconds: float = 300.0
    # Fyers WS silent-disconnect detection: reconnect after this many seconds of
    # no ticks while connected. Keep the live default at 90s: calm index sessions
    # can legitimately go quiet long enough that 45s causes false reconnect churn.
    # The detector remains settings-driven for emergency tightening.
    fyers_ws_silent_disconnect_seconds: float = 90.0
    # Divergence warnings: once per 5min bar instead of every intel cycle.
    divergence_warning_throttle_seconds: float = 300.0
    # ── 2026-07-03 lifecycle forensics fixes ──
    # Non-trend give-back scope: on 2026-07-03 four option trades peaked
    # +0.32R..+0.49R and round-tripped to full stops (~Rs.3,900) because normal
    # mid-day trades are excluded from the give-back guard and the Rs.500 floor
    # exceeds 0.40R at 1-lot size. When the session is NOT a confirmed trend day,
    # chop round-trips are the norm — every trade gets give-back protection at a
    # lower arm. Confirmed trend days keep full let-winners-run behavior.
    profit_giveback_nontrend_all_trades: bool = True
    profit_giveback_nontrend_arm_r_multiple: float = 0.30
    profit_giveback_nontrend_min_peak_rs: float = 300.0
    # Exhaustion honesty: ADX>40 entries carried "Risk of exhaustion" in their own
    # reasons yet still graded A+ 0.95 (both 10:25 losers). Cap confidence so an
    # exhaustion-flagged setup cannot wear the top label.
    ut_exhaustion_confidence_cap: float = 0.82
    # Pre-candle flip veto: the predictor projected the 15m flip that stopped the
    # 14:53 A+ short, but is wired to nothing. Default OFF — logs would-veto
    # telemetry so live sessions accumulate evidence before it gates anything.
    pre_candle_flip_veto_enabled: bool = False
    pre_candle_flip_veto_min_confidence: float = 0.70
    # Live 15m positions are monitored by BOTH their own 15m structure and the fast 1m
    # tape. The first minute after a fill is shock-only. After that, a completed 1m
    # reversal needs independent 5m/15m or market-wide evidence plus trade-state proof.
    # All state thresholds are in R / peak-retention units, so the same policy scales
    # across indices, futures and option premiums and can be A/B replayed with one toggle.
    fifteen_min_one_min_support_exit_enabled: bool = True
    fifteen_min_one_min_support_min_hold_seconds: float = 15.0
    fifteen_min_one_min_support_fresh_trade_seconds: float = 60.0
    fifteen_min_one_min_support_loss_min_r: float = 0.35
    fifteen_min_one_min_support_stop_near_r: float = 0.35
    fifteen_min_one_min_support_win_giveback_pct: float = 0.25
    fifteen_min_one_min_support_lock_buffer_r: float = 0.15
    fifteen_min_one_min_support_premium_adverse_min_r: float = 0.10
    fifteen_min_one_min_support_min_confirmations: int = 2
    # During the first minute, admit a second protective lane below the catastrophic
    # shock tier only when price risk is already meaningful and three independent,
    # current sources agree: market breadth, directional volume, and either fresh L1
    # order flow or an option-chain snapshot captured after the fill.  This keeps the
    # 60-second candle-noise guard while removing its blind spot for a genuine fast turn.
    fifteen_min_fast_intelligence_enabled: bool = True
    fifteen_min_fast_intelligence_min_sources: int = 3
    fifteen_min_fast_chain_min_votes: int = 2
    fifteen_min_fast_depth_max_age_seconds: float = 2.0
    fifteen_min_fast_order_flow_bullish_ratio: float = 1.08
    fifteen_min_fast_order_flow_bearish_ratio: float = 0.92
    # Refresh a non-probabilistic live outlook from the immutable entry MFE forecast,
    # realized underlying move, premium R state and current evidence.  Entry forecast
    # probability is retained as a baseline; no uncalibrated probability is invented.
    fifteen_min_live_expected_move_enabled: bool = True
    # A candle must have both a recent transport receipt and a bar clock compatible
    # with the live evaluation time before it can drive an early exit.
    early_exit_candle_max_age_seconds_1min: float = 90.0
    early_exit_candle_max_age_seconds_5min: float = 90.0
    early_exit_candle_max_age_seconds_15min: float = 90.0
    early_exit_timeframe_alignment_seconds: float = 90.0
    # Range-day regime-pillar damp: on 2026-07-03 the 5min regime detector read
    # TRENDING 54-60% of a day the session context correctly classified RANGE_DAY
    # all day; the grading regime pillar's +0.30 "alignment" bonus on those
    # phantom trends manufactured six A+ labels that went 0/6. When the session
    # is a confirmed range day, cap the regime pillar's trend-alignment bonus.
    range_day_regime_pillar_damp_enabled: bool = True
    range_day_regime_pillar_bonus_cap: float = 0.10
    range_day_regime_pillar_min_confidence: float = 0.60
    # Weak-market scalp mode: on a confirmed range day a non-burst entry arms
    # scalp-lock trailing from entry and caps the RR target — book small gains
    # that clear charges instead of holding chop toward runner targets. Bursts
    # (thrust / market-wide reversal / explosive) are exempt and run free.
    weak_market_scalp_mode_enabled: bool = True
    weak_market_scalp_min_day_confidence: float = 0.60
    weak_market_scalp_max_rr: float = 1.5
    # 3 halted an index for the rest of the day on a 3-loss run. Simulated over the
    # measured 59.8% win rate and v7's unconstrained arrival rate (3.61 trades/index/day,
    # 30 days x 4 indices): k=3 blocks ~13 trades and fires ~10 halts, k=4 blocks ~3 and
    # fires ~2.6, k=5 blocks ~1. Raising to 4 recovers ~10 trades (~Rs 7,800 at the
    # measured Rs 784/trade) and still halts an index that is genuinely broken, one trade
    # later. k=5 makes the protection close to vacuous.
    #
    # Per-index remains correct: the old global tape interleaved four indices so the halt
    # never fired at all. This tunes a working guard rather than reverting to a broken one.
    max_consecutive_losses: int = 4
    index_cooldown_minutes: float = 4.0
    # ── Regime-scoped cooldown (2026-07-10) ──
    # UT1 is a flip engine: after an opposite-signal exit the reverse entry IS
    # the trade. A blanket post-exit cooldown blocked those re-entries, so the
    # cooldown now applies ONLY when the index is in a chop/range/no-movement
    # pocket (revenge-churn protection); confirmed trend/normal tape skips it.
    index_cooldown_regime_scoped: bool = True
    # ── FUT position discipline (2026-07-10) ──
    # Futures are ALWAYS 1 lot and at most one concurrent FUT position per
    # index, regardless of user lot settings (options keep user sizing).
    fut_lots_hard_cap: int = 1
    fut_one_position_per_index: bool = True
    ut_option_history_mode: Literal["stored_or_synthetic", "fetch_or_synthetic", "fetch_or_skip"] = "fetch_or_synthetic"
    # REMOVED 2026-08-06: the eleven market_intel_pullback_* / market_intel_loss_* fields
    # that used to sit here had no consumer anywhere in the codebase — a whole guard that
    # was designed, configured (guard_enabled=True in .env) and never built. Keeping them
    # meant eleven knobs an operator could set and watch do nothing.
    #
    # The job they describe is done by the loss-guard family below and in scanner.py:
    # _profit_guard_reversal_context classifies the pullback (noise / small_pullback /
    # reversal / big_reversal / structure_break / premium_melt) from candle structure,
    # _loss_guard_severity scores it with loss_guard_rejection_wick_*, loss_guard_grind_*,
    # loss_guard_opt_premium_melt_* and the OI/PCR vote settings, and _loss_guard_decision
    # turns that severity into the R multiple at which the trade is cut. Those settings
    # are read and tested; these were neither.
    #
    # model_config uses extra='ignore', so the keys may stay in .env harmlessly.
    pcr_near_strikes: int = 5 # Strikes on either side of ATM for near-money PCR

    # ─── Logging ─────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    activity_event_dedupe_seconds: float = 2.0 # Suppress identical dashboard-tape events emitted by overlapping callers
    log_path: str = "./logs"
    # rtracker = regime tracker. Shadow-only evidence store for regime conviction,
    # accepted/rejected signals, and weekly study reports. It never changes live
    # entries, exits, stops, or broker behavior.
    rtracker_enabled: bool = True
    rtracker_data_dir: str = "data_store/rtracker"
    rtracker_report_weeks: int = 8
    # Days of raw event day-files to keep on disk. 0 keeps everything, which is the
    # behaviour that let the archive reach 83 MB in 61 days. Any value below
    # rtracker_report_weeks * 7 is raised to it, so pruning can never delete evidence
    # the summary still reports on.
    rtracker_retention_days: int = 0

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    @model_validator(mode="after")
    def resolve_angelone_pin_fallback(self):
        if not self.angelone_password.strip() and self.angelone_pin.strip():
            self.angelone_password = self.angelone_pin
        return self

    @field_validator('active_indices')
    @classmethod
    def validate_active_indices(cls, values: list[str]) -> list[str]:
        """Keep the runtime universe explicit; FINNIFTY is intentionally out of scope."""
        supported = {"NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"}
        result: list[str] = []
        for value in list(values or []):
            name = str(value or "").strip().upper()
            if name in supported and name not in result:
                result.append(name)
        return result

    @field_validator('entry_policy_min_rr')
    @classmethod
    def validate_entry_policy_min_rr(cls, value: float) -> float:
        """The live fresh-exposure boundary may be stricter, never below 1.5R."""
        return max(1.5, float(value or 1.5))

    @field_validator('max_daily_loss_pct')
    @classmethod
    def validate_max_daily_loss(cls, v: float) -> float:
        if not 0 < v <= 10:
            raise ValueError('max_daily_loss_pct must be between 0 and 10')
        return v

    @field_validator('futures_sl_pct')
    @classmethod
    def validate_futures_stop(cls, v: float) -> float:
        if not 0 < v <= 5:
            raise ValueError('futures_sl_pct must be between 0 and 5')
        return v

    @field_validator('options_sl_pct')
    @classmethod
    def validate_options_stop(cls, v: float) -> float:
        if not 0 < v <= 50:
            raise ValueError('options_sl_pct must be between 0 and 50')
        return v

    @field_validator('max_trade_loss_abs')
    @classmethod
    def validate_max_trade_loss(cls, v: float) -> float:
        if v < 0:
            raise ValueError('max_trade_loss_abs must be >= 0 (0 disables the per-trade rupee backstop)')
        return v

    @field_validator('entry_margin_buffer_pct')
    @classmethod
    def validate_entry_margin_buffer(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError('entry_margin_buffer_pct must be between 0 and 100')
        return v

    @field_validator('entry_margin_cache_seconds')
    @classmethod
    def validate_entry_margin_cache(cls, v: float) -> float:
        if not 0 <= v <= 30:
            raise ValueError('entry_margin_cache_seconds must be between 0 and 30')
        return v

    def risk_configuration_summary(self) -> Dict[str, Any]:
        fut_budget = self.capital_fut * self.risk_fut_pct / 100.0
        opt_budget = self.capital_opt * self.risk_opt_pct / 100.0
        issues = []
        if self.capital_total <= 0 or self.capital_fut <= 0 or self.capital_opt <= 0:
            issues.append("capital values must be positive")
        if self.capital_fut + self.capital_opt > self.capital_total:
            issues.append("segment capital exceeds total capital")
        warnings = []
        # A segment whose ENTIRE daily budget cannot pay one round trip is disabled in
        # practice: every entry is refused on budget wherever the entry-risk gate runs,
        # and the same figure becomes that segment's circuit-breaker threshold. The only
        # capital test above is `<= 0`, so the live CAPITAL_FUT=1.0 (Rs.0.015 of budget
        # at RISK_FUT_PCT=1.5) certified clean — ok=True with no warnings at all, and
        # the dashboard reported a segment nobody can trade as healthy.
        #
        # Whether that value is a typo or a deliberate way to ban futures is the
        # operator's call; this only makes the condition visible. Warning and not issue
        # for the same reason: `ok` gates startup surfaces, and a segment disabled on
        # purpose must not break the run.
        for _label, _budget, _cost in (
            ("FUT", fut_budget, float(getattr(self, "fut_cost", 200.0) or 0.0)),
            ("OPT", opt_budget, float(getattr(self, "opt_cost", 80.0) or 0.0)),
        ):
            if _cost > 0 and _budget < _cost:
                warnings.append(
                    f"{_label} daily budget Rs.{_budget:,.2f} is below one round trip "
                    f"(Rs.{_cost:,.0f}) — the segment cannot fund a single trade"
                )
        # max_trade_loss_abs is advisory / circuit-breaker projection only when
        # auto risk sizing is OFF — it does not shrink lots. 0 = disabled.
        if self.auto_risk_sizing_enabled and self.max_trade_loss_abs > 0:
            if self.max_trade_loss_abs > fut_budget:
                warnings.append(
                    f"max_trade_loss_abs {self.max_trade_loss_abs:.0f} exceeds FUT daily budget {fut_budget:.0f}"
                )
            if self.max_trade_loss_abs > opt_budget:
                warnings.append(
                    f"max_trade_loss_abs {self.max_trade_loss_abs:.0f} exceeds OPT daily budget {opt_budget:.0f}"
                )
        return {
            "ok": not issues,
            "issues": issues,
            "warnings": warnings,
            "capital_total": self.capital_total,
            "capital_fut": self.capital_fut,
            "capital_opt": self.capital_opt,
            "futures_sl_pct": self.futures_sl_pct,
            "options_sl_pct": self.options_sl_pct,
            "max_trade_loss_abs": self.max_trade_loss_abs,
            "fut_daily_budget": fut_budget,
            "opt_daily_budget": opt_budget,
            "auto_risk_sizing_enabled": bool(self.auto_risk_sizing_enabled),
            "cascade_lot_scale_enabled": bool(self.cascade_lot_scale_enabled),
            "cascade_lot_multiplier": float(self.cascade_lot_multiplier),
            "cascade_lot_multiplier_fut": float(self.cascade_lot_multiplier_fut),
            "cascade_lot_multiplier_opt": float(self.cascade_lot_multiplier_opt),
            "cascade_lot_multiplier_opt_min": float(self.cascade_lot_multiplier_opt_min),
            "heavy_trend_lot_scale_enabled": bool(self.heavy_trend_lot_scale_enabled),
            "heavy_trend_lot_multiplier_opt": float(self.heavy_trend_lot_multiplier_opt),
            "heavy_trend_min_aligned_indices": int(self.heavy_trend_min_aligned_indices),
            "cascade_multi_tf_stack_enabled": bool(self.cascade_multi_tf_stack_enabled),
            "cascade_midcap_solo_max_tf": int(self.cascade_midcap_solo_max_tf),
            "cascade_midcap_dual_main_max": int(self.cascade_midcap_dual_main_max),
            "cascade_midcap_dual_addon_max": int(self.cascade_midcap_dual_addon_max),
            "cascade_core_bucket_max_positions": int(self.cascade_core_bucket_max_positions),
            "cascade_midcap_bucket_max_positions": int(self.cascade_midcap_bucket_max_positions),
            "cascade_max_concurrent_positions": int(self.cascade_max_concurrent_positions),
            "dual_clone_reduction_enabled": bool(self.dual_clone_reduction_enabled),
            "risk_per_trade_daily_fraction": self.risk_per_trade_daily_fraction,
            "sizing_mode": "fixed_lots" if not self.auto_risk_sizing_enabled else "risk_budget",
        }

    def effective_trade_loss_cap(self, inst_type: str) -> float:
        """Never let one trade consume more than its segment's full daily budget.

        max_trade_loss_abs == 0 disables the extra rupee backstop: the cap is
        then the segment's own daily budget (the user's visible settings)."""
        segment = str(inst_type or "FUT").upper()
        segment_budget = max(0.0, float(
            self.capital_opt * self.risk_opt_pct / 100.0
            if segment == "OPT"
            else self.capital_fut * self.risk_fut_pct / 100.0
        ))
        abs_cap = float(self.max_trade_loss_abs or 0.0)
        if abs_cap <= 0:
            return segment_budget
        return min(abs_cap, segment_budget)

    def per_trade_risk_budget(self, inst_type: str, quality_multiplier: float = 1.0) -> float:
        """Per-trade sizing budget (₹). Daily % knobs stay circuit-breaker only."""
        from engine.position_sizing import per_trade_risk_budget as _budget

        return _budget(self, inst_type, quality_multiplier=quality_multiplier)


    def get_signal_fire_mode(self) -> str:
        """Return the canonical scanner entry-timing mode.

        Both legacy settings remain readable so existing .env files that used
        either switch continue to work. New UI writes synchronize both values.
        """
        intrabar_enabled = _setting_enabled(
            getattr(self, "live_intrabar_signal_entries", True),
            True,
        )
        legacy_mode = str(getattr(self, "ut_signal_mode", "realtime") or "realtime").lower()
        return "INTRACANDLE" if intrabar_enabled and legacy_mode != "confirmed" else "CANDLE_CLOSE"

    def set_signal_fire_mode(self, mode: str) -> str:
        """Synchronize the canonical mode with both legacy engine settings."""
        normalized = str(mode or "INTRACANDLE").strip().upper().replace("-", "_")
        aliases = {
            "REALTIME": "INTRACANDLE",
            "INTRA_CANDLE": "INTRACANDLE",
            "LIVE": "INTRACANDLE",
            "CONFIRMED": "CANDLE_CLOSE",
            "CLOSED_CANDLE": "CANDLE_CLOSE",
            "CLOSE": "CANDLE_CLOSE",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"INTRACANDLE", "CANDLE_CLOSE"}:
            raise ValueError(f"Unsupported signal fire mode: {mode}")
        intrabar_enabled = normalized == "INTRACANDLE"
        self.live_intrabar_signal_entries = intrabar_enabled
        self.ut_signal_mode = "realtime" if intrabar_enabled else "confirmed"
        if str(getattr(self, "scanner_engine", "")).lower() in {"ut_smart", "ut_smart_s1"}:
            self.uts_lane_allow_intrabar_entries = intrabar_enabled
        return normalized


    def get_ut_engine_params(self) -> dict:
        preset = getattr(self, "ut_preset", "AGGRESSIVE").upper()
        kv = 1.0
        atr = 10
        sig_mode = "realtime" if self.get_signal_fire_mode() == "INTRACANDLE" else "confirmed"
        if preset == "BALANCED":
            kv = 1.3
            atr = 10
        elif preset == "CONSERVATIVE":
            kv = 1.6
            atr = 14
            
        return {
            "key_value": kv,
            "atr_period": atr,
            "regime_adaptation": _setting_enabled(getattr(self, "ut_regime_adaptation", True), True),
            "use_heikin_ashi": self.ut_use_heikin_ashi,
            "signal_mode": sig_mode,
            "adx_filter": self.ut_adx_filter,
            "adx_period": self.ut_adx_period,
            "adx_threshold": self.ut_adx_threshold,
            "strict_adx": self.ut_strict_adx,
            "session_filter": self.ut_session_filter,
            "session_start": self.ut_session_start,
            "session_end": self.ut_session_end
        }
        

    def save_to_env(self):
        """Save current settings to .env file"""
        env_file = Path('.env')
        if not env_file.exists():
            env_file.touch()

        existing_values = dotenv.dotenv_values(env_file)
            
        # Serialize fields using pydantic's model_dump
        data = self.model_dump()
        # Never write back a setting nothing reads. This method dumps EVERY field, so
        # each save resurrected all 32 inert keys as live .env lines — undoing the
        # commenting that documents them as dead and making the preflight shout about
        # them again on the next start. Observed on 2026-08-08: a running scanner
        # persisted settings and every one came back.
        #
        # Their commented lines survive untouched, so the record of what was once
        # configured, and why it does nothing, stays in the file.
        try:
            from config.inert_settings import INERT as _INERT
        except Exception:
            _INERT = {}

        # A value that still equals its dataclass default is NOT a configuration
        # choice, and writing it makes it one. .env wins over the class default
        # (model_config env_file), so the moment a field is persisted at its default it
        # is pinned there for good: changing the default in code afterwards does nothing,
        # because the file shadows it. Measured 2026-08-11 — 681 of 741 keys in .env sat
        # at their default, i.e. 92% of the file was pinning nobody had asked for, and a
        # setting added in code was frozen at its first-ever default the moment the
        # running scanner next persisted (surge_tick_proxy_enabled did exactly that).
        #
        # Keys ALREADY in .env are still refreshed, so nothing an operator has set — or
        # that a previous build pinned — silently changes underneath them. This only
        # stops NEW defaults being written, which is what makes a default upgradeable.
        model_fields = getattr(type(self), "model_fields", {}) or {}

        def _is_untouched_default(field_name: str, value) -> bool:
            field = model_fields.get(field_name)
            if field is None:
                return False
            try:
                default = field.get_default(call_default_factory=True)
            except TypeError:          # older pydantic signature
                try:
                    default = field.get_default()
                except Exception:
                    return False
            except Exception:
                return False
            try:
                return bool(value == default)
            except Exception:
                return False

        for key, val in data.items():
            if key in _INERT:
                continue
            # OAuth registration belongs to deployment configuration. A cached
            # settings object must not overwrite a credential rotation.
            if key in {"upstox_api_key", "upstox_api_secret", "upstox_redirect_uri", "upstox_user_id"}:
                continue
            env_key = key.upper()
            # Auto execution authority is deliberately process-local.  Persisting
            # True lets a restarted REAL process inherit order authority before a
            # human can review the new process's feed/broker readiness.  The runtime
            # value remains untouched; only the next-start value is forced manual.
            if key == "auto_mode":
                val = False
            if env_key not in existing_values and _is_untouched_default(key, val):
                continue
            # Handle list/dict serialization (e.g. active_indices)
            if isinstance(val, (list, dict)):
                str_val = json.dumps(val)
            else:
                str_val = str(val)
            if env_key in PROTECTED_SECRET_ENV_KEYS and not _clean_env_secret(str_val):
                existing_secret = existing_values.get(env_key)
                if _clean_env_secret(existing_secret):
                    str_val = str(existing_secret)
            _set_env_key_with_retry(env_key, str_val)

    def get_instruments(self) -> Dict[str, Any]:
        """Load instrument configuration from instruments.json"""
        config_path = Path(__file__).parent / "instruments.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def ensure_directories(self):
        """Create required directories"""
        dirs = [self.log_path, "data_store", "data_store/candles"]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton instance
_settings = None


def get_settings() -> Settings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
