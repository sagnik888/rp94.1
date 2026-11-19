"""
Intelligence Aggregator — Combines all intelligence into a unified score
═══════════════════════════════════════════════════════════════
"""
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from loguru import logger
from intelligence.volume_profiler import VolumeProfiler
from intelligence.oi_tracker import OITracker
from intelligence.pcr_engine import PCREngine
from intelligence.greeks_engine import GreeksEngine
from intelligence.regime_detector import RegimeDetector
from intelligence.order_flow import OrderFlowAnalyzer
from intelligence.derivative_context import derivative_corroboration


#: Evidence mass a HEALTHY live read carries -- the sum of the maximum vote of the pillars
#: that are present whenever the feed and the option chain are both working:
#:
#:     volume 0.35 + oi 0.10 + pcr 0.20 + oi_wall 0.20 + regime 0.25 + order_flow 0.20
#:
#: top_down (0.35) needs session/anchor/breadth context and divergence (0.30) only fires on a
#: detected trap, so neither belongs in the baseline; a read carrying them is BETTER than
#: fully informed and divides by its own larger mass.
#:
#: This is the FLOOR on the score's denominator. It exists because dividing by the evidence
#: actually present let one pillar publish full conviction -- see the note in analyze().
FULL_EVIDENCE_MASS = 1.30


class IntelligenceAggregator:
    """Combines all market intelligence sources into a single score"""

    def __init__(self):
        self.volume = VolumeProfiler()
        self.oi = OITracker()
        self.pcr = PCREngine()
        self.greeks = GreeksEngine()
        self.regime = RegimeDetector()
        self.order_flow = OrderFlowAnalyzer()
        self._last_divergence_warning = {}

    def _warn_divergence_once(self, instrument: str, kind: str, message: str, throttle_seconds: Optional[float] = None) -> None:
        if throttle_seconds is None:
            # Default to one warning per 5min bar: the old 15s default re-fired
            # on every ~30s intel cycle (57 repeats in one session) and drowned logs.
            try:
                from config.settings import get_settings
                throttle_seconds = float(
                    getattr(get_settings(), "divergence_warning_throttle_seconds", 300.0) or 300.0
                )
            except Exception:
                throttle_seconds = 300.0
        now = time.time()
        key = (str(instrument or ""), str(kind or ""))
        last = float(self._last_divergence_warning.get(key, 0.0) or 0.0)
        if now - last >= max(1.0, float(throttle_seconds or 300.0)):
            self._last_divergence_warning[key] = now
            logger.warning(message)

    def _implied_iv_from_price(self, spot: float, strike: float, days: float, premium: float, opt_type: str) -> Optional[float]:
        if spot <= 0 or strike <= 0 or days <= 0 or premium <= 0:
            return None
        intrinsic = max(0.0, spot - strike) if opt_type == "call" else max(0.0, strike - spot)
        if premium + 0.05 < intrinsic:
            return None
        low, high = 0.01, 3.0
        for _ in range(40):
            mid = (low + high) / 2.0
            model_price = float(self.greeks.calculate(spot, strike, days, mid, opt_type).get("price", 0.0) or 0.0)
            if model_price < premium:
                low = mid
            else:
                high = mid
        solved = max(0.01, min(3.0, (low + high) / 2.0))
        if solved <= 0.0101 or solved >= 2.999:
            return None
        return solved

    def _chain_iv_context(self, options_chain: pd.DataFrame, spot_price: float, days_to_expiry: float) -> Dict:
        if options_chain is None or options_chain.empty or "strike" not in options_chain.columns:
            return {"call_iv": 0.15, "put_iv": 0.15, "source": "default"}
        row = options_chain.iloc[(options_chain["strike"].astype(float) - float(spot_price)).abs().argsort()[:1]]
        atm_row = row.iloc[0]
        call_iv = None
        put_iv = None
        for key in ("call_iv", "ce_iv", "iv_call"):
            if key in atm_row and pd.notna(atm_row.get(key)):
                call_iv = float(atm_row.get(key))
                break
        for key in ("put_iv", "pe_iv", "iv_put"):
            if key in atm_row and pd.notna(atm_row.get(key)):
                put_iv = float(atm_row.get(key))
                break
        if call_iv and call_iv > 1:
            call_iv /= 100.0
        if put_iv and put_iv > 1:
            put_iv /= 100.0
        source = "chain_iv" if call_iv or put_iv else "default"
        strike = float(atm_row.get("strike") or 0.0)
        # Prefer a live two-way mid over last-traded price. IV — and therefore every
        # Greek — is only as real as the premium it is solved from, and an LTP on an
        # illiquid strike can be minutes old. `*_mid` is present when the broker
        # supplied a usable book; it falls back to LTP when it did not.
        def _premium(side: str) -> float:
            mid = float(atm_row.get(f"{side}_mid", 0.0) or 0.0)
            return mid if mid > 0 else float(atm_row.get(f"{side}_ltp", 0.0) or 0.0)

        call_premium, put_premium = _premium("call"), _premium("put")
        premium_basis = "mid" if float(atm_row.get("call_mid", 0.0) or 0.0) > 0 else "ltp"
        if not call_iv and call_premium > 0:
            call_iv = self._implied_iv_from_price(spot_price, strike, days_to_expiry, call_premium, "call")
            source = f"premium_implied_iv_{premium_basis}" if call_iv else source
        if not put_iv and put_premium > 0:
            put_iv = self._implied_iv_from_price(spot_price, strike, days_to_expiry, put_premium, "put")
            source = f"premium_implied_iv_{premium_basis}" if put_iv else source
        if call_iv is None and put_iv is not None:
            call_iv = put_iv
            source = f"{source}_put_proxy"
        elif put_iv is None and call_iv is not None:
            put_iv = call_iv
            source = f"{source}_call_proxy"
        elif call_iv is None and put_iv is None:
            source = "default_low_quality"
        return {
            "call_iv": call_iv or 0.15,
            "put_iv": put_iv or 0.15,
            "source": source,
            "quality_ok": source != "default_low_quality",
        }

    def analyze(
        self,
        instrument: str,
        timeframe: str,
        candle_df: pd.DataFrame,
        candle_1min_df: Optional[pd.DataFrame] = None, # Added 1m support
        options_chain: Optional[pd.DataFrame] = None,
        spot_price: float = 0,
        strike_interval: int = 50,
        days_to_expiry: Optional[float] = 7,
        price_change_pct: float = 0,
        chain_quality: Optional[Dict] = None,
        volatility_value: Optional[float] = None,
        volatility_percentile: Optional[float] = None,
        volatility_source: str = "",
        instrument_iv_observations: Optional[List[float]] = None,
        instrument_iv_min_sessions: int = 20,
        session_context: Optional[Dict[str, Any]] = None,
        anchor_regime: Optional[Dict[str, Any]] = None,
        breadth_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        # Run all intelligence modules and produce aggregate score
        results = {}
        score = 0.0  # -1.0 (bearish) to +1.0 (bullish)
        components = 0
        # Sum of the MAXIMUM vote each counted pillar could have cast. The published
        # score divides by this rather than by the component COUNT, so it is a weighted
        # average of conviction on a true -1..+1 scale instead of a mean of small numbers.
        #
        # Measured 2026-08-08, mean vs weight-normalised over the same pillar votes:
        #   adding a CONFIRMING pillar LOWERED the score  4 of 7 times (mean) / 0 (this)
        #   range actually reachable                      0.24         / 1.00
        #   discrimination, strong-bull vs neutral        3.66 sd      / 3.74 sd
        #
        # The mean was not merely mis-scaled, it was non-monotone: more agreement could
        # mean less conviction, because a confirming pillar whose vote sat below the
        # running mean dragged it down. It also meant a better-informed read scored LOWER
        # -- with the option chain present the ceiling fell from 0.29 to 0.178 -- so every
        # threshold in the system, including this module's own 0.3/0.6 BUY/STRONG_BUY,
        # was unreachable. On this scale those read as written again.
        #
        # CORRECTED 2026-08-09. Dividing by the evidence actually present made the score a
        # measure of DIRECTION but not of how much was known, so thin evidence produced full
        # conviction. Regime is the only pillar that counts unconditionally, so whenever the
        # option chain was unauthoritative, order flow abstained and volume was unavailable --
        # the 09:15-09:19 state, and any feed outage -- regime alone owned the denominator:
        # 0.25/0.25 = 1.0, published as 100.0 / STRONG_BUY on the least-informed read of the
        # session. Reproduced: one pillar, components_analyzed=1, score=100.0.
        #
        # The denominator now has a FLOOR of the evidence mass a fully-informed read carries,
        # so a pillar set thinner than that cannot inflate its own conviction. Above the floor
        # nothing changes, which is why the measured ceiling and every gate are untouched.
        max_possible = 0.0
        regime_score_multiplier = 1.0

        actual_strike_count = int(len(options_chain)) if options_chain is not None else 0
        if chain_quality is None:
            chain_quality = {
                "source": "unknown" if actual_strike_count else "none",
                "strike_count": actual_strike_count,
                "score": 95 if actual_strike_count >= 10 else (60 if actual_strike_count >= 3 else 0),
                "fallback": actual_strike_count < 3,
                "expiry_aligned": actual_strike_count >= 3,
                # No freshness claim. This branch runs when the CALLER supplied no
                # chain_quality, so nothing here has measured an age -- and
                # "stale": False with "age_seconds": 0 asserted the strongest
                # possible freshness from nothing. The option-chain age gate in
                # scanner._option_chain_entry_freshness_blocker already fails
                # closed on a missing age, which is the correct answer here.
                "age_measured": False,
            }
        else:
            chain_quality = dict(chain_quality)
        reported_strikes = int(chain_quality.get("strike_count", actual_strike_count) or 0)
        chain_authoritative = bool(
            chain_quality.get(
                "intelligence_usable",
                min(actual_strike_count, reported_strikes) >= 3
                and not bool(chain_quality.get("fallback", False))
                and not bool(chain_quality.get("stale", False))
                and bool(chain_quality.get("expiry_aligned", True)),
            )
        )
        chain_authoritative = bool(chain_authoritative and actual_strike_count >= 3)
        chain_quality["intelligence_usable"] = chain_authoritative
        chain_quality["contract_lookup_only"] = bool(
            actual_strike_count and not chain_authoritative
        )
        results["data_quality"] = chain_quality
        intelligence_chain = options_chain if chain_authoritative else None
        # Separate contract lanes are never poured into cash-index candle volume.
        # Snapshot intervals without exchange clocks remain descriptive, not entry
        # confirmations. As-of queries cannot see later observations in replay.
        derivatives = {"timeframe": timeframe, "futures": [], "options": []}
        derivative_as_of = None
        derivative_source = getattr(self, "derivative_source", None)
        if callable(derivative_source) and candle_df is not None and not candle_df.empty:
            try:
                edge = pd.Timestamp(candle_df.index[-1])
                if edge.tzinfo is None:
                    edge = edge.tz_localize("Asia/Kolkata")
                edge += pd.Timedelta(minutes={"1min":1,"5min":5,"15min":15}[timeframe])
                now = pd.Timestamp.now(tz="Asia/Kolkata")
                derivative_as_of = min(edge,now).timestamp()
                derivatives = derivative_source(instrument, timeframe, derivative_as_of)
            except Exception as exc:
                derivatives["reason"] = str(exc)
        results["derivative_participation"] = derivatives
        if not chain_authoritative:
            chain_quality.setdefault("reason", (
                "Option chain unavailable" if actual_strike_count == 0 else
                "Option chain is not authoritative (fallback, stale, expiry mismatch, or insufficient strikes)"
            ))

        # The requested analysis timeframe owns its volume/order-flow measurement.
        # A 1m frame may be used by explicit microstructure features elsewhere, but
        # silently scoring it inside a 5m/15m bundle made the bundle's labels untrue.
        intel_df = candle_df

        # 1. Volume
        vol = self.volume.analyze(intel_df, instrument)
        if isinstance(vol, dict):
            # Name the frame the ratio was measured on. A 5m bar is the sum of its five
            # 1m bars, so the RAW volume agrees across timeframes while the ratio cannot:
            # each frame compares a bar with its own 20-bar history. Without the label a
            # reader cannot tell which yardstick produced the number on screen.
            vol["timeframe"] = str(timeframe or "")
        results["volume"] = vol
        profile_ctx = vol.get("market_structure_profile", {}) if isinstance(vol, dict) else {}
        aroon_ctx = vol.get("aroon", {}) if isinstance(vol, dict) else {}
        wavetrend_ctx = vol.get("wavetrend", {}) if isinstance(vol, dict) else {}
        results["market_structure"] = profile_ctx if isinstance(profile_ctx, dict) else {}
        results["aroon_context"] = aroon_ctx if isinstance(aroon_ctx, dict) else {}
        results["wavetrend"] = wavetrend_ctx if isinstance(wavetrend_ctx, dict) else {}
        volume_available = vol.get("available") is True
        if volume_available:
            if vol["signal"] == "BULLISH_VOLUME":
                score += 0.2 # Increased from 0.15
            elif vol["signal"] == "BEARISH_VOLUME":
                score -= 0.2 # Increased from 0.15
            if vol["buy_sell_ratio"] > 1.5: # Increased threshold
                score += 0.15
            elif vol["buy_sell_ratio"] < 0.6: # Lowered threshold
                score -= 0.15
        profile_signal = str(results["market_structure"].get("signal") or "")
        if profile_signal in {"ACCEPTING_HIGHER", "UPSIDE_ACCEPTANCE"}:
            score += 0.04
        elif profile_signal in {"ACCEPTING_LOWER", "DOWNSIDE_ACCEPTANCE"}:
            score -= 0.04
        elif profile_signal == "REJECTING_HIGHER":
            score -= 0.03
        elif profile_signal == "REJECTING_LOWER":
            score += 0.03

        aroon_trend = str(results["aroon_context"].get("trend") or "")
        aroon_late = bool(results["aroon_context"].get("late_trend_caution"))
        if aroon_trend == "UP" and not aroon_late:
            score += 0.03
        elif aroon_trend == "DOWN" and not aroon_late:
            score -= 0.03
        # Price-structure/Aroon remain valid without traded volume, but the
        # volume pillar itself exists only when a real/proxy source opted in.
        if volume_available:
            components += 1
            max_possible += 0.35  # volume
        elif profile_signal or aroon_trend:
            # They still voted (up to +/-0.07 above) and the 0.35 volume budget that would
            # have covered them was not taken, so without this the numerator carried a term
            # the denominator never budgeted for and score could exceed max_possible --
            # measured 0.32 against 0.25, an averaged_score of 1.28 silently clamped to 1.0.
            max_possible += 0.07  # price-structure + aroon, standing alone

        # 2. OI
        if intelligence_chain is not None:
            if (chain_quality.get("source") == "upstox" and chain_quality.get("volume_complete") is True
                    and {"call_volume", "put_volume"}.issubset(intelligence_chain.columns)):
                volume = intelligence_chain[["call_volume", "put_volume"]].to_numpy(dtype=float)
                if np.isfinite(volume).all() and (volume >= 0).all():
                    results["option_volume"] = {
                        "available": True, "quantity": float(volume.sum()), "provider": "Upstox",
                        "expiry": intelligence_chain.attrs.get("expiry") or chain_quality.get("active_expiry"),
                        "observed_at": chain_quality.get("fetched_at_epoch") or chain_quality.get("recorded_at"),
                    }
            oi_result = self.oi.analyze(intelligence_chain, price_change_pct, spot_price, strike_interval,
                timeframe_seconds={"1min":60,"5min":300,"15min":900}.get(timeframe,300),
                derivative_features=derivatives)
            if not isinstance(oi_result, dict): oi_result = {}
            results["oi"] = oi_result
            if oi_result.get("available") is True:
                # Conservative vote from the options-specific call/put pressure
                # model; OI remains context, never a standalone entry trigger.
                if oi_result.get("signal") == "BULLISH":
                    score += 0.10
                elif oi_result.get("signal") == "BEARISH":
                    score -= 0.10
                components += 1
                max_possible += 0.1  # oi

            # 3. PCR
            pcr_result = self.pcr.analyze(intelligence_chain, instrument, spot_price, strike_interval)
            if not isinstance(pcr_result, dict):
                pcr_result = self.pcr._empty_result()
            results["pcr"] = pcr_result
            signal_map = {"STRONG_BULLISH": 0.2, "BULLISH": 0.1, "NEUTRAL": 0, "BEARISH": -0.1, "STRONG_BEARISH": -0.2}
            if pcr_result.get("available") is True:
                score += signal_map.get(pcr_result.get("signal", "NEUTRAL"), 0)
                components += 1
                max_possible += 0.2  # pcr

        # 4. Greeks & Gamma Awareness
        if intelligence_chain is not None and spot_price > 0:
            expiry_available = days_to_expiry is not None and pd.notna(days_to_expiry) and days_to_expiry >= 0
            if expiry_available:
                iv_ctx = self._chain_iv_context(intelligence_chain, spot_price, days_to_expiry)
                greeks_atm = self.greeks.analyze_atm(
                    spot_price, strike_interval, days_to_expiry,
                    iv_call=iv_ctx["call_iv"], iv_put=iv_ctx["put_iv"],
                )
                greeks_chain = self.greeks.analyze_chain(
                    spot_price, strike_interval, days_to_expiry,
                    iv_call=iv_ctx["call_iv"], iv_put=iv_ctx["put_iv"], iv_source=iv_ctx["source"],
                )
            else:
                # OI/PCR and measured VIX remain usable. A missing contract expiry
                # cannot become invented time-to-expiry for IV inversion or Greeks.
                iv_ctx = {"source": "default_expiry_unavailable"}
                greeks_atm = {"available": False, "is_fallback": True, "msg": "Contract expiry unavailable"}
                greeks_chain = {"strikes": []}
            # ── Volatility percentile for routing ──
            # Preference order: (1) instrument-specific option IV-Rank computed
            # from this instrument's own prior-session ATM IV, (2) India VIX
            # regime percentile as a broad fallback, (3) neutral 50. The caller
            # supplies only prior-session IV observations, so this stays
            # look-ahead-safe in historical routing.
            iv_ctx_source = str(iv_ctx.get("source", ""))
            iv_is_real = not iv_ctx_source.startswith("default")
            current_atm_iv = None
            if iv_is_real:
                current_atm_iv = round(
                    ((float(iv_ctx["call_iv"]) + float(iv_ctx["put_iv"])) / 2.0) * 100.0, 2
                )

            iv_obs: List[float] = []
            for raw_iv in (instrument_iv_observations or []):
                try:
                    parsed_iv = float(raw_iv)
                except (TypeError, ValueError):
                    continue
                if parsed_iv > 0:
                    iv_obs.append(parsed_iv)

            vix_available = volatility_percentile is not None
            vix_percentile = (
                max(0.0, min(100.0, float(volatility_percentile)))
                if vix_available
                else None
            )
            vix_source = (
                str(volatility_source or "india_vix_percentile")
                if vix_available
                else "unavailable_neutral"
            )

            min_iv_sessions = max(2, int(instrument_iv_min_sessions or 20))
            if current_atm_iv is not None and len(iv_obs) >= min_iv_sessions:
                rank = sum(1 for obs in iv_obs if obs <= current_atm_iv) / len(iv_obs) * 100.0
                iv_percentile = round(max(0.0, min(100.0, rank)), 1)
                iv_basis = "instrument_iv_rank"
            elif vix_available:
                iv_percentile = round(vix_percentile, 1)
                iv_basis = "india_vix_percentile"
            else:
                # Abstain. A published 50.0 was read as a MEASURED percentile by
                # every downstream gate, and it is the most permissive value in
                # the range: 50 < 70 selected the looser 120s option-chain age
                # limit over 90s, 50 < 80 kept FUT/OPT routing off the high-IV
                # branch, and 50 < the 70.0 special-5min ceiling turned that gate
                # into a pass. Publishing None makes the existing
                # ``if iv_percentile is None`` branches reachable, which is what
                # they were written for.
                iv_percentile = None
                iv_basis = "unavailable"

            # ``percentile_available``/``percentile_source`` keep their original
            # meaning (India VIX availability) for existing dashboard consumers.
            percentile_available = vix_available
            percentile_source = vix_source

            # ── Institutional Edge: Proximity to major OI levels ──
            oi_data = results.get("oi", {})
            res_level = oi_data.get("resistance_level", 0)
            sup_level = oi_data.get("support_level", 0)
            
            dist_res = 999.0
            dist_sup = 999.0
            
            # Normalize threshold so that 0.25% at 24000 NIFTY (~60 pts) translates
            # proportionally to other instruments (e.g. ~0.50% at 12000 MIDCPNIFTY).
            dynamic_threshold = 0.25 * (24000.0 / max(1.0, spot_price))
            
            if res_level > 0:
                dist_res = (res_level - spot_price) / spot_price * 100
                if 0 < dist_res < dynamic_threshold: # Approaching major resistance wall
                    score -= 0.2 # Penalize buying into a wall
            
            if sup_level > 0:
                dist_sup = (spot_price - sup_level) / spot_price * 100
                if 0 < dist_sup < dynamic_threshold: # Approaching major support floor
                    score += 0.2 # Boost buying from a floor

            results["entry_quality"] = "EXCELLENT" if (dist_sup < dynamic_threshold or dist_res < dynamic_threshold) else "GOOD"
            results["greeks"] = {
                **greeks_atm,
                "iv_percentile": (
                    round(iv_percentile, 1) if iv_percentile is not None else None
                ),
                "iv_percentile_basis": iv_basis,
                "instrument_iv": current_atm_iv,
                "instrument_iv_sessions": len(iv_obs),
                "volatility_percentile": (
                    round(vix_percentile, 1) if vix_percentile is not None else None
                ),
                "volatility_percentile_available": percentile_available,
                "volatility_percentile_source": percentile_source,
                "volatility_value": (
                    round(float(volatility_value), 3)
                    if volatility_value is not None
                    else None
                ),
                "chain": greeks_chain["strikes"],
                "iv_source": iv_ctx["source"],
            }
            # Claim the oi_wall weight only when a wall was actually measured.
            # Taking 0.2 of the denominator with no level on either side diluted
            # every real observation, because the divisor has a hard
            # FULL_EVIDENCE_MASS floor and a silent pillar therefore costs exactly
            # as much as one voting zero. This is also the branch reached when the
            # contract expiry is unavailable and greeks_atm is {"available": False}.
            if res_level > 0 or sup_level > 0:
                components += 1
                max_possible += 0.2  # oi_wall
        elif intelligence_chain is None and spot_price > 0:
            # Fallback for when options chain is missing but we have price
            results["oi"] = self.oi._empty_result()
            results["pcr"] = self.pcr._empty_result()
            results["greeks"] = {
                "is_fallback": True,
                "msg": "Options data unavailable, using price-only fallback"
            }
        else:
            results["oi"] = self.oi._empty_result()
            results["pcr"] = self.pcr._empty_result()
            results["greeks"] = {}
        try:
            vix_value = float(volatility_value)
        except (TypeError, ValueError):
            vix_value = 0.0
        if np.isfinite(vix_value) and vix_value > 0:
            results["greeks"].setdefault("volatility_value", round(vix_value, 3))
            results["greeks"].setdefault("volatility_percentile", volatility_percentile)
            results["greeks"].setdefault("volatility_percentile_available", volatility_percentile is not None)

        # 5. Regime
        try:
            regime_result = self.regime.detect(
                candle_df,
                instrument,
                timeframe,
                session_context=session_context,
                breadth_context=breadth_context,
            )
        except TypeError as exc:
            if "breadth_context" not in str(exc):
                raise
            regime_result = self.regime.detect(
                candle_df,
                instrument,
                timeframe,
                session_context=session_context,
            )
        results["regime"] = regime_result
        if regime_result["regime"] in ["TRENDING_UP", "BREAKOUT_UP"]:
            score += 0.25 # Increased from 0.15
        elif regime_result["regime"] in ["TRENDING_DOWN", "BREAKOUT_DOWN"]:
            score -= 0.25 # Increased from 0.15
        # RANGING/VOLATILE/UNKNOWN add no directional regime vote here. Do not
        # multiply every independent OI/PCR/flow/volume observation by that label;
        # SignalManager already applies the explicit regime risk penalty.
        #
        # RANGING and VOLATILE are measured readings and keep their weight: "no
        # direction" is information. An UNKNOWN regime is the ABSENCE of a reading,
        # and the detector says so itself through ready_for_grading=False -- a flag
        # it has always published and this aggregator never read. Claiming 0.25 of
        # the denominator for it diluted every other pillar against the
        # FULL_EVIDENCE_MASS floor.
        regime_measured = bool(
            regime_result.get("ready_for_grading", True)
            and str(regime_result.get("regime") or "UNKNOWN").upper() != "UNKNOWN"
        )
        if regime_measured:
            components += 1
            max_possible += 0.25  # regime

        # 6. Order Flow
        flow_result = self.order_flow.analyze(
            intel_df,
            options_chain=intelligence_chain if chain_quality.get("volume_complete", True) else None,
            spot_price=spot_price,
            strike_interval=strike_interval,
            market_context=chain_quality,
        )
        results["order_flow"] = flow_result
        # Order flow was the ONLY pillar that never abstained: it voted and took a
        # component unconditionally, unlike volume, OI and PCR just above. An unavailable
        # read publishes signal NEUTRAL with source "unavailable", so "unknown" was scored
        # as "balanced" AND added to the denominator, shrinking every other pillar's
        # contribution by ~1/N. intel_df is the 1-minute frame and the analyzer needs five
        # live bars, so this bit 09:15-09:19 every single session — and if the 1-minute
        # frame is empty the aggregator falls back to 5-minute and the dead window
        # stretches past the 09:20 entry cutoff.
        # Abstain only on POSITIVE evidence of absence. Volume/OI/PCR use a strict
        # `is True`, but their analyzers all publish the key; order flow's did not until
        # now, and a strict test here silently stopped every existing collaborator and
        # stub from contributing — which is how a helper ends up wired to nothing. Both
        # markers are checked so a future early-return that forgets the flag still
        # abstains on its "unavailable" source.
        flow_available = (
            flow_result.get("available") is not False
            and str(flow_result.get("source") or "") != "unavailable"
        )
        # Do not stamp a real-looking source onto a reading that did not happen.
        results["candle_pressure"] = {
            **flow_result,
            "source": (flow_result.get("source") or "ohlcv_close_position") if flow_available else "unavailable",
        }
        if flow_available:
            if flow_result["signal"] == "BULLISH":
                score += 0.2 # Increased from 0.15
            elif flow_result["signal"] == "BEARISH":
                score -= 0.2 # Increased from 0.15
            components += 1
            max_possible += 0.2  # order_flow

        # Price versus estimated pressure is a heuristic divergence, not proof
        # of manipulation or participant intent. Missing pressure cannot vote.
        divergence_detected = False
        try:
            flow_ratio = float(flow_result.get("ratio"))
            price_move = float(price_change_pct)
        except (TypeError, ValueError, OverflowError):
            flow_ratio, price_move = float("nan"), float("nan")
        divergence_usable = bool(flow_available and np.isfinite(flow_ratio)
                                 and flow_ratio >= 0 and np.isfinite(price_move))
        # Case A: Price Up but Order Flow Bearish (Distribution Trap)
        if divergence_usable and price_move > 0.1 and flow_ratio < 0.8:
            score -= 0.3
            divergence_detected = True
            self._warn_divergence_once(
                instrument,
                "distribution_trap",
                f"DIVERGENCE: {instrument} Price rising; estimated pressure suggests Distribution. Heuristic context only.",
            )
        # Case B: Price Down but Order Flow Bullish (Accumulation Trap)
        elif divergence_usable and price_move < -0.1 and flow_ratio > 1.25:
            score += 0.3
            divergence_detected = True
            self._warn_divergence_once(
                instrument,
                "accumulation_shakeout",
                f"DIVERGENCE: {instrument} Price falling; estimated pressure suggests Accumulation. Heuristic context only.",
            )
        
        if divergence_detected:
            # The divergence vote is +/-0.30 — LARGER than any counted pillar (regime is
            # +/-0.25) — but it was the only term of that size never to increment
            # `components`. Since averaged_score = score / components, that made its
            # influence depend on how many UNRELATED pillars happened to have data:
            # measured +0.043 with seven pillars available versus +0.100 with three, a
            # 2.3x swing driven by option-chain availability rather than by the market.
            # It is an independent stress-test vote, so it carries its own weight.
            components += 1
            max_possible += 0.3  # divergence

        results["divergence_alert"] = divergence_detected
        results["divergence"] = {
            "available": divergence_usable,
            "detected": divergence_detected,
            "price_change_pct": price_move if np.isfinite(price_move) else None,
            "pressure_ratio": flow_ratio if np.isfinite(flow_ratio) and flow_ratio >= 0 else None,
            "pressure_source": flow_result.get("source") or ("ohlcv_close_position" if flow_available else "unavailable"),
            "interpretation": "Price and estimated pressure disagree; this does not establish market manipulation.",
        }

        # ═══ STRESS TEST: System CA OI (Cumulative Analysis) ═══
        if intelligence_chain is not None and not intelligence_chain.empty:
            # Analyze distribution of OI across entire chain (Wide Format Support)
            if 'call_oi' in intelligence_chain.columns:
                total_call_oi = intelligence_chain['call_oi'].sum()
                total_put_oi = intelligence_chain['put_oi'].sum()
            else:
                total_call_oi = intelligence_chain[intelligence_chain['type'] == 'CE']['oi'].sum()
                total_put_oi = intelligence_chain[intelligence_chain['type'] == 'PE']['oi'].sum()
                
            oi_imbalance = (total_call_oi - total_put_oi) / (total_call_oi + total_put_oi) if (total_call_oi + total_put_oi) > 0 else 0
            
            # Calculate nearby strikes (+/- 5) for strike-specific analysis
            nearby_strikes = []
            if spot_price > 0:
                atm = round(spot_price / strike_interval) * strike_interval
                min_strike = atm - (5 * strike_interval)
                max_strike = atm + (5 * strike_interval)
                
                # Filter for wide format
                if 'strike' in intelligence_chain.columns:
                    nearby_chain = intelligence_chain[(intelligence_chain['strike'] >= min_strike) & (intelligence_chain['strike'] <= max_strike)]
                    nearby_strikes = nearby_chain[['strike', 'call_oi', 'put_oi', 'call_volume', 'put_volume']].to_dict('records')
            
            results["oi"]["cumulative_analysis"] = {
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
                "imbalance": round(oi_imbalance, 3),
                "sentiment": "BEARISH_PRESSURE" if oi_imbalance > 0.1 else ("BULLISH_PRESSURE" if oi_imbalance < -0.1 else "BALANCED"),
                "nearby_strikes": nearby_strikes
            }

            # ═══ MAX PAIN ═══ the expiry strike that minimises total option-buyer
            # payout (an expiry magnet for index options). Pure-pandas, wide format.
            try:
                if 'call_oi' in intelligence_chain.columns and 'strike' in intelligence_chain.columns and spot_price > 0:
                    oc = intelligence_chain.dropna(subset=['strike'])
                    strikes = oc['strike'].astype(float).to_numpy()
                    coi = oc['call_oi'].astype(float).fillna(0.0).to_numpy()
                    poi = oc['put_oi'].astype(float).fillna(0.0).to_numpy()
                    # One strike-by-strike matrix instead of a pandas loop. The loop ran
                    # two Series.clip per strike (122 on a 61-strike chain), and each
                    # pandas op also deep-copied chain.attrs: 50% of analyze() time,
                    # 117ms a call (2026-09-25). Row P is the payout if expiry settles
                    # at strike P; argmin keeps the old first-minimum tie-break.
                    best_k = None
                    if strikes.size:
                        settle_minus_strike = strikes[:, None] - strikes[None, :]
                        pain = (
                            (coi * np.clip(settle_minus_strike, 0, None)).sum(axis=1)
                            + (poi * np.clip(-settle_minus_strike, 0, None)).sum(axis=1)
                        )
                        best_k = float(strikes[int(np.argmin(pain))])
                    if best_k:
                        results["max_pain"] = {
                            "strike": best_k,
                            "distance_pct": round((float(spot_price) - best_k) / best_k * 100, 2),
                        }
            except Exception:
                pass

        top_down = self._top_down_context_adjustment(
            session_context=session_context,
            anchor_regime=anchor_regime,
            breadth_context=breadth_context,
        )
        if top_down["has_context"]:
            score += float(top_down["score_adjustment"])
            components += 1
            max_possible += 0.35  # top_down
            multiplier_cap = float(top_down.get("regime_multiplier_cap") or 1.0)
            if multiplier_cap < regime_score_multiplier:
                regime_score_multiplier = multiplier_cap

        results["session_context"] = session_context or {}
        results["anchor_regime"] = anchor_regime or {}
        results["breadth_context"] = breadth_context or {}
        results["top_down_context"] = top_down

        # Final Score Aggregation (Scaled to -100 to +100 for SignalManager)
        score_before_regime_dampening = score
        score *= regime_score_multiplier
        # Participation corroborates the existing conviction exactly once. It is
        # not another directional price/OI vote or a new gate, and missing inputs
        # leave the existing score unchanged.
        derivative_context = derivative_corroboration(
            derivatives, instrument, timeframe, derivative_as_of, score,
            spot_price=spot_price,
            option_expiry=(intelligence_chain.attrs.get("expiry", "")
                           if intelligence_chain is not None else ""),
        )
        results["derivative_context"] = derivative_context
        score *= derivative_context["multiplier"]
        # Divide by at least the evidence mass of a fully-informed read. Below that floor
        # this reduces to score / FULL_EVIDENCE_MASS, so a thin pillar set is scored on the
        # same yardstick as a complete one instead of being graded against itself.
        averaged_score = (
            (score / max(max_possible, FULL_EVIDENCE_MASS)) if max_possible > 0 else 0.0
        )
        # Normalize *before* scaling so the public contract can never escape
        # [-100, +100].  The old order computed final_score from an unclamped
        # value and clamped a different variable afterward.
        score = max(-1.0, min(1.0, averaged_score))
        final_score = score * 100.0

        # Final signal
        if score > 0.3:
            final_signal = "STRONG_BUY" if score > 0.6 else "BUY"
        elif score < -0.3:
            final_signal = "STRONG_SELL" if score < -0.6 else "SELL"
        else:
            final_signal = "HOLD"

        # Safely extract values for final results dict
        vol_data = results.get("volume", {})
        if not isinstance(vol_data, dict): vol_data = {}
        
        flow_data = results.get("order_flow", {})
        if not isinstance(flow_data, dict): flow_data = {}
        
        regime_data = results.get("regime", {})
        if not isinstance(regime_data, dict): regime_data = {}

        results["aggregate"] = {
            "score": round(final_score, 1),
            # The SAME conviction on the canonical -1..+1 scale. `score` above is
            # legacy -100..+100 and rounds to 1dp, so a neutral read publishes as
            # e.g. 0.9 -- indistinguishable by magnitude from 90% conviction. Readers
            # that need the number rather than the display string must use this one
            # (engine.intelligence_score.aggregate_intelligence_score).
            "score_norm": round(score, 4),
            # How much of a fully-informed read's evidence this score rests on. Diagnostic
            # only, deliberately: the SCORE is already damped by it, so no gate has to be
            # rewired to consult this. components_analyzed had the opposite problem -- it was
            # published for exactly this purpose and never read by anything.
            "evidence_mass": round(max_possible, 3),
            "evidence_ratio": round(min(1.0, max_possible / FULL_EVIDENCE_MASS), 3),
            "signal": final_signal,
            "components_analyzed": components,
            "instrument": instrument,
            "timeframe": timeframe,
            "measured_volume": vol_data.get("current_volume", 0),
            "measurement_timeframe": vol_data.get("measurement_timeframe", timeframe),
            "measurement_timestamp": vol_data.get("measurement_timestamp", ""),
            "measurement_basis": vol_data.get("measurement_basis", ""),
            "order_ratio": flow_data.get("ratio", 1.0),
            "regime": regime_data.get("regime", "RANGING"),
            "regime_conviction_label": regime_data.get("conviction_label", ""),
            "regime_conviction_summary": regime_data.get("conviction_summary", ""),
            "regime_conviction_vector": dict(regime_data.get("conviction_vector", {}) or {}),
            "score_before_regime_dampening": round(score_before_regime_dampening, 4),
            "regime_score_multiplier": regime_score_multiplier,
            "derivative_participation_multiplier": round(derivative_context["multiplier"], 4),
            "day_type": (session_context or {}).get("day_type", ""),
            "session_bias": (session_context or {}).get("bias", ""),
            "effective_day_type": (session_context or {}).get("effective_day_type", ""),
            "effective_bias": (session_context or {}).get("effective_bias", ""),
            "phase_confidence": (session_context or {}).get("phase_confidence", 0.0),
            "mood_horizons": (session_context or {}).get("mood_horizons", {}),
            "hourly_anchor": (anchor_regime or {}).get("regime", ""),
            "breadth_bias": (breadth_context or {}).get("bias", ""),
            "top_down_adjustment": round(float(top_down.get("score_adjustment") or 0.0), 4),
            "market_structure_signal": results["market_structure"].get("signal", ""),
            "market_structure_state": results["market_structure"].get("state", ""),
            "aroon_trend": results["aroon_context"].get("trend", ""),
            "aroon_freshness": results["aroon_context"].get("freshness", ""),
            "wavetrend_exhaustion": results["wavetrend"].get("exhaustion", ""),
        }

        return results

    def _top_down_context_adjustment(
        self,
        *,
        session_context: Optional[Dict[str, Any]],
        anchor_regime: Optional[Dict[str, Any]],
        breadth_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        adjustment = 0.0
        reasons: List[str] = []
        multiplier_cap = 1.0

        session_context = session_context or {}
        anchor_regime = anchor_regime or {}
        breadth_context = breadth_context or {}

        day_type = str(
            session_context.get("effective_day_type")
            or session_context.get("day_type")
            or ""
        ).upper()
        day_conf = self._bounded_float(
            session_context.get("effective_confidence")
            or session_context.get("phase_confidence")
            or session_context.get("confidence"),
            0.0,
            1.0,
        )
        if day_conf >= 0.45:
            if day_type == "TREND_DAY_UP":
                delta = 0.12 * day_conf
                adjustment += delta
                reasons.append(f"session trend-up prior {delta:+.3f}")
            elif day_type == "TREND_DAY_DOWN":
                delta = -0.12 * day_conf
                adjustment += delta
                reasons.append(f"session trend-down prior {delta:+.3f}")
            elif day_type == "RANGE_DAY":
                multiplier_cap = min(multiplier_cap, 0.85)
                reasons.append("range-day dampens directional conviction")

        delta_ctx = session_context.get("session_delta", {})
        if isinstance(delta_ctx, dict) and str(delta_ctx.get("source") or "").lower() == "volume":
            delta_bias = str(delta_ctx.get("bias") or "").upper()
            delta_ratio = abs(self._bounded_float(delta_ctx.get("delta_ratio"), -1.0, 1.0))
            if delta_bias == "BULLISH" and delta_ratio >= 0.15:
                delta = min(0.06, delta_ratio * 0.12)
                adjustment += delta
                reasons.append(f"session cumulative delta bullish {delta:+.3f}")
            elif delta_bias == "BEARISH" and delta_ratio >= 0.15:
                delta = -min(0.06, delta_ratio * 0.12)
                adjustment += delta
                reasons.append(f"session cumulative delta bearish {delta:+.3f}")

        vix_change = self._bounded_float(session_context.get("vix_change_pct"), -20.0, 20.0)
        session_bias = str(
            session_context.get("effective_bias")
            or session_context.get("bias")
            or ""
        ).upper()
        if abs(vix_change) >= 2.0:
            if vix_change > 0 and session_bias == "BULLISH":
                adjustment -= 0.04
                reasons.append("VIX rising against bullish session")
            elif vix_change > 0 and session_bias == "BEARISH":
                adjustment -= 0.04
                reasons.append("VIX rising confirms bearish session")
            elif vix_change < 0 and session_bias == "BULLISH":
                adjustment += 0.04
                reasons.append("VIX falling confirms bullish session")
            elif vix_change < 0 and session_bias == "BEARISH":
                adjustment += 0.04
                reasons.append("VIX falling softens bearish session")

        anchor_name = str(anchor_regime.get("regime") or "").upper()
        anchor_ready = bool(anchor_regime.get("ready_for_grading", True))
        anchor_conf = self._bounded_float(anchor_regime.get("confidence"), 0.0, 100.0) / 100.0
        if anchor_ready and anchor_conf >= 0.45:
            if anchor_name in {"TRENDING_UP", "BREAKOUT_UP"}:
                delta = 0.10 * anchor_conf
                adjustment += delta
                reasons.append(f"60min anchor up {delta:+.3f}")
            elif anchor_name in {"TRENDING_DOWN", "BREAKOUT_DOWN"}:
                delta = -0.10 * anchor_conf
                adjustment += delta
                reasons.append(f"60min anchor down {delta:+.3f}")

        breadth_bias = str(breadth_context.get("bias") or "").upper()
        breadth_votes = int(self._bounded_float(breadth_context.get("aligned_count"), 0.0, 4.0))
        if breadth_votes >= 3:
            if breadth_bias == "BULLISH":
                adjustment += 0.08
                reasons.append("cross-index breadth bullish")
            elif breadth_bias == "BEARISH":
                adjustment -= 0.08
                reasons.append("cross-index breadth bearish")

        adjustment = max(-0.35, min(0.35, adjustment))
        return {
            "has_context": bool(reasons),
            "score_adjustment": round(adjustment, 4),
            "regime_multiplier_cap": round(multiplier_cap, 3),
            "reasons": reasons,
        }

    @staticmethod
    def _bounded_float(value: Any, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        return max(low, min(high, parsed))
