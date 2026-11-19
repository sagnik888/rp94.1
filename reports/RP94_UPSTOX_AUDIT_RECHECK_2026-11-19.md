# RP94 Upstox Audit Recheck

The supplied forensic audit reviewed main at `ce792f12`. This recheck reviewed the workspace branch at `e725d67` and updated its working tree. Startup still constructed an Angel One execution fallback. Several data paths and the contract master depended on Kite or Angel One.

Source changes are ready for evaluation in a fresh process. The current processes and active trading configuration were left in place. Broker validation used read-only requests and a market-data subscription. At the account check, Upstox reported zero positions, orders and intraday fills. Live order acceptance, partial-fill handling against the exchange and prior-session contract-note reconciliation remain unverified.

## Findings and changes

| Audit finding | Result against current source |
| --- | --- |
| Quad position can move on one vote | Confirmed. Two matching primitive votes are required to set or reverse displayed position. Sparse votes hold prior state or show unknown during initialization. Quad still uses the union as its trigger policy. |
| Same-bar BUY and SELL survive the merge | Confirmed. Both events remain available in diagnostics. Entry selection and raw opposite-arrow exits abstain on the conflicted bar. |
| Alt engine exception silently produces UT entries | Confirmed. The scanner publishes a degraded error with no entry signals. Protective price context remains available and failed results are invalidated for admission. |
| Combined scanner hides an unhealthy lane at envelope level | Confirmed. The envelope now reports full, degraded or halted health and lane eligibility. The healthy lane keeps its own observation clock. New entries require recognized lane ownership. |
| Quad strips the combined `as_of` argument | Confirmed signature requirement. The combined adapter clips base and dependency frames at the observation cutoff before invoking native engines. A historical snapshot still needs the observations that were actually available at that time. |
| Supertrend overlay differs from the effective bands | Confirmed. The chart uses the same recursive bands as trend detection. The existing SMA true-range input stays in use. |
| Historical Quad expands one merged event into several lifecycles | Confirmed. The normal historical path keeps the merged event and its source list. `HISTORICAL_QUAD_SOURCE_ATTRIBUTION=True` restores the research-only split. Bar-based historical fills still need a frozen observation and quote tape for exact live parity. |
| Raw opposite exits can execute before entry approval | Confirmed. Discretionary Upstox opposite exits now require a fresh quote for the exact held contract. Emergency shock and exchange protection retain their existing paths. Entry-side grading and exit-side confirmation remain independent decisions. |
| Ordinary and exceptional entries have different evidence requirements | Confirmed and intentional. Derivative activity is now identified as such and kept out of cash-index volume. Exceptional routes still have to satisfy their required evidence. Live order admission checks the exact contract, lot size, price and applicable touch and risk rules. |
| A minimum lot can exceed planned risk while optional admission is off | Confirmed. New Upstox exposure and adds must fit per-trade, segment and global planned-risk limits. The estimate includes configured slippage and fees and the option stop-limit buffer. Recovery of existing positions bypasses new-exposure admission. |
| The fixed urgency bonus can dominate score | Confirmed. Its effect on realized returns needs paired replay. The live score weight remains in use. |
| PRIMARY_15 has legacy Quad and one-minute exceptions | Confirmed. `STRICT_ENTRY_TIMEFRAMES=True` enforces the configured frame set for new entries. Existing default retains the legacy exceptions. |
| Intrabar entries and v10 shadow mode are defaults | Confirmed in source. The report supplies no holdout evidence to justify promoting v10, disabling provisional signals or changing the entry thresholds. |
| Credential exposure | Confirmed in tracked history. Removed the hardcoded Upstox key, secret and PIN from current source. Rotation of the exposed credentials and any history rewrite are separate account and repository operations. |

## Upstox integration

The application constructs one execution broker and one live market-data provider. Upstox provides the active NSE and BSE index contract master, exact contract keys, dated and intraday candles, option chains, REST depth and market-data WebSocket ticks. The process uses the existing offline archives for historical work. The public instrument master was read successfully for NIFTY, BANKNIFTY, MIDCPNIFTY and SENSEX. The live read-only check matched chain contracts against all four master entries and returned depth for each.

The existing Upstox adapter inverted the caller's resolved exit and protective order sides and skipped its configured execution authority. The adapter now preserves side and checks mode, the account lease and the session account at the mutation boundary. New orders wait for a recent position reconciliation. Position bookkeeping uses the actual fill and order status fields. Protective stop discovery checks the exact contract, side, size, product and order type. The adapter distinguishes an unreadable book from an empty book and uses exact tags to recover ambiguous acknowledgements. Option protection uses a buffered stop-limit order. A fast gap can leave that order unfilled.

Prospective risk uses an estimate of the standard Upstox charge schedule. The statutory model applies the April 2026 STT rates to the sell leg. Live realized charges still require the account's contract notes. [Upstox pricing](https://upstox.com/brokerage-charges/) [NSE STT schedule](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)

API contracts checked against the public broker documentation. [Instrument master](https://upstox.com/developer/api-documentation/instruments/) [V3 market feed](https://upstox.com/developer/api-documentation/v3/get-market-data-feed/) [Full quotes](https://upstox.com/developer/api-documentation/get-full-market-quote-v3/) [Order details](https://upstox.com/developer/api-documentation/get-order-details/) [Order placement](https://upstox.com/developer/api-documentation/place-order/)

## Verification

The focused regression suite passed 309 tests across scanner, combined-lane cache, historical attribution, quote freshness, entry risk, broker authority, contract master and readiness. The production data-provider path also published a V3 index tick, accepted it through the source gate, resolved a matching SENSEX contract with usable depth and returned Upstox candle history. These checks ran independently of the running trading process.

## Research status

The archived 102-trade corrected simulation lost ₹2,116.37 after modeled costs. The 151-trade exit-policy holdout remained negative, with candidate profit factor 0.420. Both figures match the archived results and come from different strategy samples. Expected returns for this build remain unmeasured. Profit-lock floors, quote chase thresholds, Wilder ATR, majority-only entries, urgency reranking and source exclusions remain hypotheses. [Corrected option replay](paired_replay_expiry_analysis_20260914.json) [Exit-policy walk-forward](exit_policy_walk_forward_upgrade_20260927.json)

Full trading-clock parity, unseen-period expectancy, actual fill and fee reconciliation, measured order slippage and exchange protection acknowledgement still require evidence. An exact replay needs timestamped quote and order observations, matched contracts and a fixed holdout. The older shared chat and contract notes were absent from the supplied evidence.
