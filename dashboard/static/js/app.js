let lastSystemScanCount = -1;

// XSS guard: escape dynamic/external text before it is inserted via innerHTML.
function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
// Expose for the other dashboard modules (trade-table, etc.)
window.escapeHtml = escapeHtml;

function setHtml(el, html) {
    if (el.innerHTML !== html) el.innerHTML = html;
}

function patchRow(tr, rowHtml) {
    if (tr.innerHTML === rowHtml) return;
    const temp = document.createElement('tr');
    temp.innerHTML = rowHtml;
    const oldCells = Array.from(tr.children);
    const newCells = Array.from(temp.children);
    for (let i = 0; i < Math.max(oldCells.length, newCells.length); i++) {
        if (!oldCells[i]) {
            tr.appendChild(newCells[i].cloneNode(true));
        } else if (!newCells[i]) {
            tr.removeChild(oldCells[i]);
        } else if (oldCells[i].innerHTML !== newCells[i].innerHTML || oldCells[i].className !== newCells[i].className) {
            oldCells[i].innerHTML = newCells[i].innerHTML;
            oldCells[i].className = newCells[i].className;
        }
    }
}

const state = {
    currentInstrument: 'NIFTY',
    currentTimeframe: '5min',
    lastSimId: 0,
    latestData: {},
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    tsLineSeries: null,
    pcrLineSeries: null,
    ofrLineSeries: null,
    oiDeltaSeries: null,
    supportWallSeries: null,
    resistanceWallSeries: null,
    signalSeries: null,
    chartResizeFrame: null,
    chartDefaultVisibleRangeId: null,
    chartSnapshotPendingId: null,
    logFilter: 'all',
    activityView: 'logs',
    fullLog: [],
    lastRenderedLogData: '',
    lastRenderedFilter: 'all',
    lastRenderedEngineLabData: '',
    restHydrationTimer: null,
    quoteRefreshTimer: null,
    quoteRefreshInFlight: false,
    quoteRefreshUnavailable: false,
    chartRefreshTimer: null,
    lastRestHydrationAt: 0,
    lastWsUpdateAt: 0,
    wsPayloadOmitted: false,
    pendingChartRequest: null,
    chartRequestInFlight: false,
    chartStreamEnabled: localStorage.getItem('ut_chart_stream_enabled') !== 'false',
    lastEquityRenderKey: '',
    wsRequestSeq: 0,
    pendingCommands: new Map(),
    connectionControlsBound: false,
    tradePagination: {
        key: '',
        page: 1,
        pageSize: 100,
        inFlight: false
    }
};

const INDEX_NAMES = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'MIDCPNIFTY'];

const TICKER_MAP = {
    'NIFTY': 'NIFTY', 'NSEI': 'NIFTY', '^NSEI': 'NIFTY',
    'BANKNIFTY': 'BANKNIFTY', 'NSEBANK': 'BANKNIFTY', '^NSEBANK': 'BANKNIFTY',
    'SENSEX': 'SENSEX', 'BSESN': 'SENSEX', '^BSESN': 'SENSEX',
    'MIDCPNIFTY': 'MIDCPNIFTY', 'MIDCAPNIFTY': 'MIDCPNIFTY'
};

document.addEventListener('DOMContentLoaded', () => {
    setupIntelTabs();
    setupSettingsExperience();
    initChart();
    applyChartStreamState(state.chartStreamEnabled, { notifyBackend: false, requestSnapshot: false });
    connectWebSocket();
    startIndexQuoteRefresh();
    setupChartEventListeners();
    setupSettingsEventListeners();
    setupEventListeners();
    completeUpstoxRedirectIfPresent();
    updateMarketTime();
    setInterval(updateMarketTime, 1000);
});

async function refreshDashboardSnapshot(reason = 'manual', options = {}) {
    try {
        const now = Date.now();
        if (!options.force && now - state.lastRestHydrationAt < 1500) return;
        state.lastRestHydrationAt = now;

        const response = await authFetch('/api/state', { cache: 'no-store' });
        if (!response.ok) throw new Error(`State fetch failed: ${response.status}`);

        const snapshot = await response.json();
        if (!snapshot || snapshot.status === 'starting') return;

        const mergedData = mergeDashboardData(state.latestData, snapshot);
        state.latestData = mergedData;
        updateDashboard(mergedData);
    } catch (error) {
        console.warn(`Dashboard REST hydration failed (${reason})`, error);
    }
}

async function getDashboardToken() {
    if (window.dashboardAuthToken) return window.dashboardAuthToken;
    try {
        const res = await fetch('/api/client_config', { cache: 'no-store' });
        if (!res.ok) throw new Error(`Token fetch failed: ${res.status}`);
        const cfg = await res.json();
        const token = cfg.dashboard_token || '';
        if (token) window.dashboardAuthToken = token;
        return token;
    } catch (err) {
        console.warn('Dashboard token bootstrap failed; reconnect will retry.', err);
        return '';
    }
}

async function authFetch(url, options = {}) {
    const token = await getDashboardToken();
    const headers = { ...(options.headers || {}) };
    if (token) headers['X-Dashboard-Token'] = token;
    return fetch(url, { ...options, headers });
}

function startIndexQuoteRefresh() {
    if (state.quoteRefreshTimer) return;
    // Index badges use a cache-only endpoint that is independent of the full signal
    // scan. A slow option-chain/intelligence cycle therefore cannot freeze the spot
    // LTP for the 10-41 seconds observed in the live Aug-21 logs.
    void refreshIndexQuotes();
    state.quoteRefreshTimer = setInterval(refreshIndexQuotes, 500);
}

async function refreshIndexQuotes() {
    if (state.quoteRefreshInFlight || state.quoteRefreshUnavailable) return;
    state.quoteRefreshInFlight = true;
    try {
        const response = await authFetch('/api/quotes', { cache: 'no-store' });
        if (response.status === 404) {
            // Supports a dashboard page served briefly by an older backend during a
            // rolling restart. The normal websocket/scan snapshot remains the fallback.
            state.quoteRefreshUnavailable = true;
            if (state.quoteRefreshTimer) clearInterval(state.quoteRefreshTimer);
            state.quoteRefreshTimer = null;
            return;
        }
        if (!response.ok) return;
        const quoteData = await response.json();
        if (!quoteData?.instruments || Object.keys(quoteData.instruments).length === 0) return;
        const merged = mergeDashboardData(state.latestData, quoteData);
        state.latestData = merged;
        const normalized = {};
        Object.entries(merged.instruments || {}).forEach(([key, value]) => {
            normalized[TICKER_MAP[key] || key] = value;
        });
        updateInstrumentSelectors(normalized, merged);
        updateQuoteSessionLabel(merged);
    } catch (_) {
        // Websocket scan snapshots remain a safe fallback during a transient local
        // HTTP failure. Avoid a console flood from a 500ms best-effort refresh lane.
    } finally {
        state.quoteRefreshInFlight = false;
    }
}

async function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // /api/client_config seeds the HttpOnly /ws cookie. Never put the dashboard
    // credential in the WebSocket URL where proxies and access logs can retain it.
    await getDashboardToken();
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    ws.onopen = () => {
        window.wsReconnectDelay = 1000;
        if (statusDot) {
            statusDot.classList.add('connected');
            statusDot.classList.remove('error');
            statusDot.style.background = '';
        }
        if (statusText) statusText.textContent = 'Connected';
        ws.send(JSON.stringify({ cmd: 'get_state' }));
        ws.send(JSON.stringify({ cmd: 'set_chart_stream', enabled: state.chartStreamEnabled }));
        refreshDashboardSnapshot('websocket-open');
        startChartRefreshLoop();
        if (!state.restHydrationTimer) {
            state.restHydrationTimer = setInterval(() => {
                const wsIsStale = Date.now() - (state.lastWsUpdateAt || 0) > 10000;
                const hasState = state.latestData?.instruments && Object.keys(state.latestData.instruments).length > 0;
                if (state.wsPayloadOmitted || wsIsStale || !hasState) {
                    refreshDashboardSnapshot('periodic-stale');
                }
            }, 30000);
        }
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.request_id && state.pendingCommands.has(msg.request_id)) {
                const pending = state.pendingCommands.get(msg.request_id);
                state.pendingCommands.delete(msg.request_id);
                if (pending?.resolve) pending.resolve(msg);
            }
            if (msg.type === 'full_update' || msg.type === 'delta_update') {
                state.lastWsUpdateAt = Date.now();
                const payloadOmitted = msg.data?.ws_payload_omitted === true;
                if (payloadOmitted) {
                    state.wsPayloadOmitted = true;
                    void refreshDashboardSnapshot('websocket-payload-omitted', { force: true });
                } else if (msg.data?.instruments && Object.keys(msg.data.instruments).length > 0) {
                    state.wsPayloadOmitted = false;
                }
                const mergedData = mergeDashboardData(state.latestData, msg.data);
                state.latestData = mergedData;
                if (msg.type === 'delta_update' && isLightweightDashboardDelta(msg.data)) {
                    updateDashboardHeartbeat(mergedData);
                } else {
                    updateDashboard(mergedData);
                }
            } else if (msg.type === 'chart_snapshot') {
                applyChartSnapshot(msg.data);
            } else if (msg.type === 'chart_stream_updated') {
                applyChartStreamState(msg.data?.enabled !== false, { notifyBackend: false, requestSnapshot: true });
            } else if (msg.type === 'mode_updated') {
                if (msg.data?.mode) syncModeControls(msg.data.mode);
                refreshDashboardSnapshot('mode-updated', { force: true });
            } else if (msg.type === 'config_ack') {
                if (msg.data?.mode) syncModeControls(msg.data.mode);
                if (msg.data && msg.data.status === 'blocked') {
                    const reason = msg.data.reason ? `: ${msg.data.reason}` : '';
                    const label = msg.data.mode_command ? 'Mode switch blocked' : 'Settings blocked';
                    showToast(`${label}${reason}`, "warning");
                } else {
                    showToast("Settings Applied Successfully", "success");
                }
                refreshDashboardSnapshot('config-ack', { force: msg.data?.status === 'blocked' || msg.data?.mode_command === true });
            }
        } catch (e) {
            console.error("Data error:", e);
        }
    };
    
    ws.onclose = () => {
        if (statusDot) {
            statusDot.classList.add('error');
            statusDot.classList.remove('connected');
            statusDot.style.background = '';
        }
        if (statusText) statusText.textContent = 'Disconnected - Retrying...';
        
        window.wsReconnectDelay = (window.wsReconnectDelay || 1000) * 1.5;
        if (window.wsReconnectDelay > 30000) window.wsReconnectDelay = 30000;
        
        setTimeout(connectWebSocket, window.wsReconnectDelay);
    };
    
    window.sendCommand = (data) => {
        if (ws.readyState === WebSocket.OPEN) {
            const requestId = data.request_id || `cmd_${Date.now()}_${++state.wsRequestSeq}`;
            const payload = { ...data, request_id: requestId };
            let resolveCommand;
            const commandPromise = new Promise((resolve) => {
                resolveCommand = resolve;
            });
            commandPromise.requestId = requestId;
            state.pendingCommands.set(requestId, {
                cmd: payload.cmd,
                sentAt: Date.now(),
                resolve: resolveCommand,
            });
            ws.send(JSON.stringify(payload));
            setTimeout(() => {
                const pending = state.pendingCommands.get(requestId);
                if (pending && Date.now() - pending.sentAt > 15000) {
                    state.pendingCommands.delete(requestId);
                    console.warn(`Command timed out: ${pending.cmd}`, requestId);
                    if (pending.resolve) {
                        pending.resolve({
                            type: 'command_timeout',
                            request_id: requestId,
                            data: { status: 'timeout', reason: 'command response timed out' }
                        });
                    }
                }
            }, 16000);
            return commandPromise;
        } else {
            if (window.showToast) window.showToast("Connection lost. Reconnecting...", "error");
            return Promise.resolve({
                type: 'command_error',
                data: { status: 'error', reason: 'connection lost' }
            });
        }
    };
    if (state.connectionControlsBound) return;
    state.connectionControlsBound = true;

    // ═══ SYSTEM POWER CONTROL ═══
    const powerToggle = document.getElementById('powerToggle');
    if (powerToggle) {
        powerToggle.addEventListener('change', () => {
            const stateVal = powerToggle.checked ? 'ON' : 'OFF';
            window.sendCommand({ 
                cmd: 'set_power', 
                state: stateVal 
            });
            if (state.latestData) {
                state.latestData.system_power = stateVal;
            }
            showToast(`System Power: ${stateVal}`, stateVal === 'ON' ? 'info' : 'warning');
        });
    }
    
    // ═══ REGIME MODE SWITCH ═══
    const regimeModeSwitch = document.getElementById('regimeModeSwitch');
    if (regimeModeSwitch) {
        regimeModeSwitch.addEventListener('change', () => {
            const isChecked = regimeModeSwitch.checked;
            const stateVal = isChecked ? 'ON' : 'OFF';
            window.sendCommand({ 
                cmd: 'set_regime_adaptation', 
                state: stateVal 
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.ut_regime_adaptation = isChecked;
            }
            showToast(`Regime Adaptation: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ SESSION MOOD + LTP SHOCK TOGGLES (regime card) ═══
    const sessionMoodDynamicSwitch = document.getElementById('sessionMoodDynamicSwitch');
    if (sessionMoodDynamicSwitch && !sessionMoodDynamicSwitch.dataset.bound) {
        sessionMoodDynamicSwitch.dataset.bound = '1';
        sessionMoodDynamicSwitch.addEventListener('change', () => {
            const on = sessionMoodDynamicSwitch.checked === true;
            window.sendCommand({ cmd: 'configure', session_mood_dynamic_enabled: on });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.session_mood_dynamic_enabled = on;
            }
            showToast(`Dynamic mood: ${on ? 'ON' : 'OFF'}`, 'info');
        });
    }
    const marketLtpShockSwitch = document.getElementById('marketLtpShockSwitch');
    if (marketLtpShockSwitch && !marketLtpShockSwitch.dataset.bound) {
        marketLtpShockSwitch.dataset.bound = '1';
        marketLtpShockSwitch.addEventListener('change', () => {
            const on = marketLtpShockSwitch.checked === true;
            window.sendCommand({ cmd: 'configure', market_ltp_shock_enabled: on });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.market_ltp_shock_enabled = on;
            }
            showToast(`LTP shock engine: ${on ? 'ON' : 'OFF'}`, on ? 'warning' : 'info');
        });
    }

    // ═══ SL MODE SWITCH ═══
    const slModeSwitch = document.getElementById('slModeSwitch');
    if (slModeSwitch) {
        slModeSwitch.addEventListener('change', () => {
            const isChecked = slModeSwitch.checked;
            const stateVal = isChecked ? 'NATURAL' : 'HARDCODED';
            window.sendCommand({ 
                cmd: 'set_sl_mode', 
                state: stateVal 
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.sl_mode = stateVal;
            }
            showToast(`SL Mode: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ SCANNER ENGINE SELECTOR ═══
    const scannerEngineSelect = document.getElementById('scannerEngineSelect');
    if (scannerEngineSelect) {
        scannerEngineSelect.addEventListener('change', () => {
            const mode = scannerEngineSelect.value;
            window.sendCommand({ cmd: 'set_scanner_engine', state: mode });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.scanner_engine = mode;
            }
            const labels = {
                utbot: 'UT1',
                ema: 'EMA (9/21)',
                consensus: 'Consensus (ST+EMA)',
                utbot_consensus: 'UT1 + Consensus (union)',
                quad_union: 'Quad Union (UT+EMA+ST+CONS)',
                quad_prime: 'Quad Prime (all entries + confluence exits)',
                quad_prime_s1: 'Quad Prime + S1',
                ut_smart_s1: 'UT Smart Exit + S1',
                adaptive_dual: 'Adaptive Dual (Adaptive Hybrid + APEX)',
                s1: 'S1 Confluence (10-component score + 5 gates)',
                ut_smart: 'UT Smart Exit (UT flips + smart peak exit)',
            };
            showToast(`Signal Engine: ${labels[mode] || mode}`, mode === 'utbot' ? 'info' : 'warning');
        });
    }

    // ═══ EXPIRY-DAY GAMMA MODE SWITCH ═══
    const expiryGammaSwitch = document.getElementById('expiryGammaSwitch');
    if (expiryGammaSwitch) {
        expiryGammaSwitch.addEventListener('change', () => {
            const isChecked = expiryGammaSwitch.checked;
            const stateVal = isChecked ? 'ON' : 'OFF';
            window.sendCommand({ cmd: 'set_expiry_gamma', state: stateVal });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.expiry_day_trade_current_week = isChecked;
            }
            showToast(`Expiry-Day Gamma Mode: ${stateVal}`, isChecked ? 'warning' : 'info');
        });
    }

    // ═══ COMBO PARTNER-CONFIRMS-EXIT SWITCH ═══
    const comboExitConfirmSwitch = document.getElementById('comboExitConfirmSwitch');
    if (comboExitConfirmSwitch) {
        comboExitConfirmSwitch.addEventListener('change', () => {
            const isChecked = comboExitConfirmSwitch.checked;
            const stateVal = isChecked ? 'ON' : 'OFF';
            window.sendCommand({ cmd: 'set_combo_partner_exit', state: stateVal });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.combo_partner_confirms_exit = isChecked;
            }
            showToast(`Combo Partner-Confirms-Exit: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ 1MSCALP (continuous high-burst 1min spike entries) ═══
    const oneMinScalpSwitch = document.getElementById('oneMinScalpSwitch');
    if (oneMinScalpSwitch) {
        oneMinScalpSwitch.addEventListener('change', () => {
            const isChecked = oneMinScalpSwitch.checked;
            window.sendCommand({
                cmd: 'configure',
                one_min_scalping_enabled: isChecked
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.one_min_scalping_enabled = isChecked;
            }
            const stateVal = isChecked ? 'ON' : 'OFF';
            showToast(`1MSCALP: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ 1M EMERGENCY EXIT FOR 15M TRADES ═══
    const fifteenMinOneMinSupportExitSwitch = document.getElementById('fifteenMinOneMinSupportExitSwitch');
    if (fifteenMinOneMinSupportExitSwitch) {
        fifteenMinOneMinSupportExitSwitch.addEventListener('change', () => {
            const isChecked = fifteenMinOneMinSupportExitSwitch.checked;
            window.sendCommand({
                cmd: 'configure',
                fifteen_min_one_min_support_exit_enabled: isChecked
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.fifteen_min_one_min_support_exit_enabled = isChecked;
            }
            showToast(`1M Emergency Exit: ${isChecked ? 'ON' : 'OFF'}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ P-WIN SNAP (post-win 3m snapback; decoupled from 1mscalp) ═══
    const reversionScalpSwitch = document.getElementById('reversionScalpSwitch');
    if (reversionScalpSwitch) {
        reversionScalpSwitch.addEventListener('change', () => {
            const isChecked = reversionScalpSwitch.checked;
            window.sendCommand({
                cmd: 'configure',
                post_exit_reversion_enabled: isChecked
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.post_exit_reversion_enabled = isChecked;
            }
            const stateVal = isChecked ? 'ON' : 'OFF';
            showToast(`P-WIN SNAP: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ CONCURRENCY GUARD SWITCH ═══
    const concurGuardSwitch = document.getElementById('concurGuardSwitch');
    if (concurGuardSwitch) {
        concurGuardSwitch.addEventListener('change', () => {
            const isChecked = concurGuardSwitch.checked;
            const stateVal = isChecked ? 'ON' : 'OFF';
            window.sendCommand({ 
                cmd: 'set_concurrency_guard', 
                state: stateVal 
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.ut_concurrency_guard = isChecked;
            }
            showToast(`Concurrency Guard: ${stateVal}`, isChecked ? 'info' : 'warning');
        });
    }

    // ═══ SYSTEM RECALIBRATE / REFRESH BUTTON ═══
    const systemRefreshBtn = document.getElementById('systemRefreshBtn');
    if (systemRefreshBtn) {
        systemRefreshBtn.addEventListener('click', () => {
            const icon = systemRefreshBtn.querySelector('.logo-icon');
            if (icon) {
                icon.classList.remove('recalibrating');
                void icon.offsetWidth; // Force reflow to restart animation
                icon.classList.add('recalibrating');
                setTimeout(() => {
                    icon.classList.remove('recalibrating');
                }, 1000);
            }
            window.sendCommand({
                cmd: 'system_recalibrate'
            });
            showToast("System Recalibration initiated...", "info");
            
            // Hard refresh the UI as requested by user for bottleneck recovery
            setTimeout(() => {
                window.location.reload();
            }, 600);
        });
    }
}

function mergeDashboardData(current, incoming) {
    if (!incoming) return current;
    if (!current) return incoming;

    const merged = {
        ...current,
        ...incoming,
        instruments: { ...(current.instruments || {}) }
    };
    if (incoming.trades) {
        const currentTrades = current.trades || {};
        const incomingTrades = incoming.trades || {};
        merged.trades = { ...currentTrades, ...incomingTrades };
        const currentWindow = currentTrades.window || {};
        const incomingWindow = incomingTrades.window || {};
        const currentOffset = Number(currentWindow.offset || 0);
        const incomingOffset = Number(incomingWindow.offset || 0);
        const keepUserPage =
            currentOffset > 0 &&
            incomingOffset === 0 &&
            tradePayloadContextKey(current, currentTrades) === tradePayloadContextKey({ ...current, ...incoming }, incomingTrades);
        if (keepUserPage) {
            merged.trades.closed = currentTrades.closed || [];
            merged.trades.signals = currentTrades.signals || [];
            merged.trades.open = currentTrades.open || incomingTrades.open || [];
            merged.trades.window = currentTrades.window || incomingTrades.window;
        }
    }

    if (!incoming.instruments) {
        return merged;
    }

    Object.entries(incoming.instruments).forEach(([inst, incomingInst]) => {
        const currentInst = current.instruments?.[inst] || {};
        const nextInst = { ...currentInst, ...incomingInst };

        if (incomingInst.chart) {
            nextInst.chart = { ...(currentInst.chart || {}) };
            Object.entries(incomingInst.chart).forEach(([tf, incomingTf]) => {
                nextInst.chart[tf] = mergeChartData(currentInst.chart?.[tf], incomingTf);
            });
        }

        merged.instruments[inst] = nextInst;
    });

    return merged;
}

function isLightweightDashboardDelta(delta) {
    if (!delta || typeof delta !== 'object') return false;
    const heavyKeys = ['instruments', 'trades', 'activity_log', 'config', 'diagnostics', 'gateway_status', 'mode', 'runtime_mode'];
    return !heavyKeys.some(key => Object.prototype.hasOwnProperty.call(delta, key));
}

function updateDashboardHeartbeat(data) {
    if (!data) return;
    if (data.scan_count !== undefined) setText('scanCount', data.scan_count || 0);
    if (data.latency !== undefined) setText('scanLatency', (data.latency || 0).toFixed(0) + 'ms');
    const heartbeat = data.dashboard_heartbeat || {};
    const heartbeatTime = heartbeat.timestamp ? new Date(heartbeat.timestamp).getTime() : NaN;
    const heartbeatAgeMs = Number.isFinite(heartbeatTime) ? Math.max(0, Date.now() - heartbeatTime) : null;
    const scanAgeMs = Number.isFinite(Number(heartbeat.scan_age_ms)) ? Number(heartbeat.scan_age_ms) : null;
    const wsAgeMs = state.lastWsUpdateAt ? Math.max(0, Date.now() - state.lastWsUpdateAt) : null;
    const liveAgeMs = heartbeatAgeMs !== null ? heartbeatAgeMs : wsAgeMs;
    const liveSuffix = heartbeatAgeMs !== null
        ? (scanAgeMs !== null ? ` / scan ${(scanAgeMs / 1000).toFixed(1)}s` : '')
        : ' / ws';
    const liveText = liveAgeMs === null ? '--' : `${(liveAgeMs / 1000).toFixed(1)}s${liveSuffix}`;
    setText('liveRefreshAge', liveText);
    updateQuoteSessionLabel(data);

    // Sync signal button on heartbeat deltas so it doesn't stay stuck in state-analysing
    syncSignalState(data);
}

function updateQuoteSessionLabel(data) {
    const labelEl = document.getElementById('liveRefreshLabel');
    if (!labelEl) return;
    const status = String(data?.market_status || '').toUpperCase();
    let label = status === 'PRE-OPEN' ? 'PRE' : status === 'OPEN' || status === 'LIVE' ? 'LIVE' : 'REF';
    if (status === 'SYSTEM OFF') label = 'OFF';
    labelEl.textContent = label;
    labelEl.className = 'metric-lbl quote-session-label ' + label.toLowerCase();
    // Operator-facing reason (P2): hover shows WHY (e.g. "Market closed (weekend)").
    const detail = data?.market_status_detail;
    labelEl.title = (detail && detail.reason) ? detail.reason : (status || 'Market status');
}

function syncSignalState(data) {
    if (!data) return;
    const powerToggle = document.getElementById('powerToggle');
    const signalBtn = document.getElementById('analysingSignal');
    const analysisText = document.getElementById('analysisText');
    if (!signalBtn) return;

    const serverSaysOn = !powerToggle || data.system_power !== 'OFF';
    if (!serverSaysOn) {
        signalBtn.className = 'analysing-signal state-offline';
        if (analysisText) analysisText.textContent = 'OFFLINE';
    } else {
        // Track scan_count for the counter display
        if (data.scan_count !== undefined && data.scan_count > lastSystemScanCount) {
            lastSystemScanCount = data.scan_count;
        }
        // "RUNNING" describes the scanner process only. Calling this state LIVE made
        // closed/stale quotes look current even while the separate market label said
        // REF, which is unsafe operator feedback.
        // This prevents state-analysing (force-rescan pulsing) from getting stuck
        // permanently when scan_count updates arrive as lightweight deltas that
        // bypass updateDashboard.
        if (!signalBtn.classList.contains('state-running')) {
            signalBtn.className = 'analysing-signal state-running';
            if (analysisText) analysisText.textContent = 'RUNNING';
        }
    }
}

function syncModeControls(mode) {
    const normalizedMode = String(mode || '').toUpperCase();
    if (!normalizedMode) return;
    document.querySelectorAll('.mode-opt').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === normalizedMode);
        btn.disabled = false;
    });
    if (state.latestData) {
        state.latestData.mode = normalizedMode;
    }
}
window.syncModeControls = syncModeControls;

function updateDashboard(data) {
    if (!data) return;

    // ══ SYNC SYSTEM POWER & ANALYSING SIGNAL ══
    const powerToggle = document.getElementById('powerToggle');
    
    if (powerToggle) {
        const serverSaysOn = data.system_power !== 'OFF';
        if (powerToggle.checked !== serverSaysOn) {
            powerToggle.checked = serverSaysOn;
        }
    }
    syncSignalState(data);

    // ══ SYNC GATEWAY STATUS ══
    if (data.gateway_status) {
        updateGatewayStatus(data.gateway_status, data.mode || data.runtime_mode);
    }
    updateDiagnosticsPanel(data.diagnostics || {});
    
    // ══ SYNC REGIME MODE ══
    const regimeModeSwitch = document.getElementById('regimeModeSwitch');
    if (regimeModeSwitch && data.config && data.config.ut_regime_adaptation !== undefined) {
        const serverSaysOn = data.config.ut_regime_adaptation === true;
        if (regimeModeSwitch.checked !== serverSaysOn) {
            regimeModeSwitch.checked = serverSaysOn;
        }
    }
    const sessionMoodDynamicSwitchSync = document.getElementById('sessionMoodDynamicSwitch');
    if (sessionMoodDynamicSwitchSync && data.config && data.config.session_mood_dynamic_enabled !== undefined) {
        const on = data.config.session_mood_dynamic_enabled !== false;
        if (sessionMoodDynamicSwitchSync.checked !== on) sessionMoodDynamicSwitchSync.checked = on;
    }
    const marketLtpShockSwitchSync = document.getElementById('marketLtpShockSwitch');
    if (marketLtpShockSwitchSync && data.config && data.config.market_ltp_shock_enabled !== undefined) {
        const on = data.config.market_ltp_shock_enabled !== false;
        if (marketLtpShockSwitchSync.checked !== on) marketLtpShockSwitchSync.checked = on;
    }

    // ══ SYNC SL MODE ══
    const slModeSwitch = document.getElementById('slModeSwitch');
    if (slModeSwitch && data.config && data.config.sl_mode !== undefined) {
        const serverSaysNatural = data.config.sl_mode === 'NATURAL';
        if (slModeSwitch.checked !== serverSaysNatural) {
            slModeSwitch.checked = serverSaysNatural;
        }
    }

    // ══ SYNC SCANNER ENGINE ══
    const scannerEngineSelect = document.getElementById('scannerEngineSelect');
    if (scannerEngineSelect && data.config && data.config.scanner_engine !== undefined) {
        const serverEngine = String(data.config.scanner_engine || 'quad_prime');
        if (scannerEngineSelect.value !== serverEngine) {
            scannerEngineSelect.value = serverEngine;
        }
    }

    // ══ SYNC CONCURRENCY GUARD ══
    const concurGuardSwitch = document.getElementById('concurGuardSwitch');
    if (concurGuardSwitch && data.config && data.config.ut_concurrency_guard !== undefined) {
        const serverSaysOn = data.config.ut_concurrency_guard === true;
        if (concurGuardSwitch.checked !== serverSaysOn) {
            concurGuardSwitch.checked = serverSaysOn;
        }
    }

    // ══ SYNC EXPIRY-DAY GAMMA MODE ══
    const expiryGammaSwitch = document.getElementById('expiryGammaSwitch');
    if (expiryGammaSwitch && data.config && data.config.expiry_day_trade_current_week !== undefined) {
        const serverSaysOn = data.config.expiry_day_trade_current_week === true;
        if (expiryGammaSwitch.checked !== serverSaysOn) {
            expiryGammaSwitch.checked = serverSaysOn;
        }
    }

    // ══ SYNC COMBO PARTNER-CONFIRMS-EXIT ══
    const comboExitConfirmSwitch = document.getElementById('comboExitConfirmSwitch');
    if (comboExitConfirmSwitch && data.config && data.config.combo_partner_confirms_exit !== undefined) {
        const serverSaysOn = data.config.combo_partner_confirms_exit === true;
        if (comboExitConfirmSwitch.checked !== serverSaysOn) {
            comboExitConfirmSwitch.checked = serverSaysOn;
        }
    }

    // ══ SYNC 1-MINUTE SCALPING ENTRY MODE ══
    const oneMinScalpSwitch = document.getElementById('oneMinScalpSwitch');
    if (oneMinScalpSwitch && data.config && data.config.one_min_scalping_enabled !== undefined) {
        const serverSaysOn = data.config.one_min_scalping_enabled !== false;
        if (oneMinScalpSwitch.checked !== serverSaysOn) {
            oneMinScalpSwitch.checked = serverSaysOn;
        }
    }

    // ══ SYNC 1M EMERGENCY EXIT FOR 15M TRADES ══
    const fifteenMinOneMinSupportExitSwitch = document.getElementById('fifteenMinOneMinSupportExitSwitch');
    if (fifteenMinOneMinSupportExitSwitch && data.config && data.config.fifteen_min_one_min_support_exit_enabled !== undefined) {
        const serverSaysOn = data.config.fifteen_min_one_min_support_exit_enabled !== false;
        if (fifteenMinOneMinSupportExitSwitch.checked !== serverSaysOn) {
            fifteenMinOneMinSupportExitSwitch.checked = serverSaysOn;
        }
    }

    const topTimeframePolicy = document.getElementById('topTimeframePolicy');
    if (topTimeframePolicy && data.config && data.config.ut_timeframe_entry_policy) {
        const serverPolicy = data.config.ut_timeframe_entry_policy;
        if ([...topTimeframePolicy.options].some(option => option.value === serverPolicy)) {
            topTimeframePolicy.value = serverPolicy;
        }
    }

    // ══ SYNC REVERSION SCALPING MODE ══
    const reversionScalpSwitch = document.getElementById('reversionScalpSwitch');
    if (reversionScalpSwitch && data.config && data.config.post_exit_reversion_enabled !== undefined) {
        const serverSaysOn = data.config.post_exit_reversion_enabled;
        if (reversionScalpSwitch.checked !== serverSaysOn) {
            reversionScalpSwitch.checked = serverSaysOn;
        }
    }

    if (data.config && data.config.chart_stream_enabled !== undefined) {
        applyChartStreamState(data.config.chart_stream_enabled !== false, { notifyBackend: false, requestSnapshot: false });
    }
    
    state.latestData = data;
    updateEngineLabPanel(data.engine_lab || {}, data);

    // Safety Alert: If instruments are missing, notify user
    if (!data.instruments || Object.keys(data.instruments).length === 0) {
        console.warn("⚠️ Scanner is still initializing data...");
    }
    
    syncSettingsFromConfig(data);

    setText('scanCount', data.scan_count || 0);
    setText('scanLatency', (data.latency || 0).toFixed(0) + 'ms');
    const heartbeat = data.dashboard_heartbeat || {};
    const heartbeatTime = heartbeat.timestamp ? new Date(heartbeat.timestamp).getTime() : NaN;
    const heartbeatAgeMs = Number.isFinite(heartbeatTime) ? Math.max(0, Date.now() - heartbeatTime) : null;
    const scanAgeMs = Number.isFinite(Number(heartbeat.scan_age_ms)) ? Number(heartbeat.scan_age_ms) : null;
    const wsAgeMs = state.lastWsUpdateAt ? Math.max(0, Date.now() - state.lastWsUpdateAt) : null;
    const liveAgeMs = heartbeatAgeMs !== null ? heartbeatAgeMs : wsAgeMs;
    const liveSuffix = heartbeatAgeMs !== null
        ? (scanAgeMs !== null ? ` / scan ${(scanAgeMs / 1000).toFixed(1)}s` : '')
        : ' / ws';
    const liveText = liveAgeMs === null ? '--' : `${(liveAgeMs / 1000).toFixed(1)}s${liveSuffix}`;
    setText('liveRefreshAge', liveText);
    updateQuoteSessionLabel(data);

    // 2. Global Trades & Performance
    try { updateActivityLog(data.activity_log); } catch(e) { console.error("Log Error:", e); }
    const tradePayload = data.trades || {};
    syncTradePaginationState(data, tradePayload);
    try { updateTradesPanel(tradePayload, data.simulation_id); } catch(e) { console.error("Trades Error:", e); }
    try { updatePnlPanel(tradePayload); } catch(e) { console.error("P&L Error:", e); }

    try {
        const hasTradeRows =
            (tradePayload.open || []).length ||
            (tradePayload.closed || []).length ||
            (tradePayload.signals || []).length;
        if (!hasTradeRows) {
            updateSignalsFeed(data.instruments || {});
        }
    } catch(e) { console.error("Signals Error:", e); }
    try { updateAnalyticsStrip(data.trades); } catch(e) { console.error("Analytics Error:", e); }

    // 3. Instrument Normalization
    const normalizedInst = {};
    if (data.instruments) {
        for (const key in data.instruments) {
            const normKey = TICKER_MAP[key] || key;
            normalizedInst[normKey] = data.instruments[key];
        }
    }
    
    try { updateInstrumentSelectors(normalizedInst, data); } catch(e) { console.error("Tabs Error:", e); }
    
    // 4. Active Instrument Updates
    const inst = normalizedInst[state.currentInstrument];
    if (inst) {
        try { updateInfoPanel(inst); } catch(e) { console.error("Info Error:", e); }
        try { updateIntelPanel(inst, data); } catch(e) { console.error("Intel Error:", e); }
        try { updateChart(inst); } catch(e) { console.error("Chart Error:", e); }
    }
}


function setupIntelTabs() {
    const tabs = document.querySelectorAll('.intel-tab');
    if (tabs.length === 0) return;
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active'));
            e.currentTarget.classList.add('active');
            
            document.querySelectorAll('.intel-view-pane').forEach(pane => {
                pane.style.display = 'none';
                pane.classList.remove('active');
            });
            
            const targetId = e.currentTarget.dataset.target;
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.style.display = 'block';
                targetPane.classList.add('active');
            }
        });
    });
}

function formatLogTime(rawTime) {
    if (!rawTime) {
        return new Date().toLocaleTimeString('en-IN', {
            timeZone: 'Asia/Kolkata',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        }) + ' IST';
    }
    const text = String(rawTime).trim();
    if (/IST$/i.test(text)) return text;
    if (/^\d{1,2}:\d{2}(:\d{2})?$/.test(text)) return text.padStart(8, '0') + ' IST';
    const parsed = new Date(text);
    if (!Number.isNaN(parsed.getTime())) {
        return parsed.toLocaleTimeString('en-IN', {
            timeZone: 'Asia/Kolkata',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        }) + ' IST';
    }
    return text;
}

function cleanLogMessage(message) {
    let text = String(message || '');
    text = text
        .replace(/ðŸ[^\s]*/g, '')
        .replace(/âš[^\s]*/g, '')
        .replace(/âœ[^\s]*/g, '')
        .replace(/Ã¢[^\s]*/g, '')
        .replace(/Â/g, '')
        .replace(/[^\x20-\x7E₹]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

    text = text
        .replace(/\bManual\s+(LONG|SHORT)\s+Signal:/i, 'Manual $1 signal:')
        .replace(/\bPotential\s+(LONG|SHORT)\s+Signal\s+for\b/i, 'Potential $1 signal for')
        .replace(/\bSignal Rejected:/i, 'Signal rejected:')
        .replace(/\|\s*Reasons:/i, '. Reason:')
        .replace(/\bBelow Grade Pref\b/i, 'below selected grade')
        .replace(/\bMulti-TF\b/i, 'multi-timeframe')
        .replace(/\bRisk-Window\b/i, 'risk window')
        .replace(/\bRange-bound noise\b/i, 'range-bound noise')
        .replace(/\bSystem initialized\b/i, 'System initialised')
        .replace(/\bInitialized\b/i, 'Initialised');

    return text || 'System update received.';
}

function normalizeToken(text) {
    return String(text || '')
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function titleCaseToken(text) {
    return normalizeToken(text).toLowerCase().replace(/\b\w/g, ch => ch.toUpperCase());
}

function compactNumber(value, digits = 0) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '--';
    const abs = Math.abs(num);
    const sign = num < 0 ? '-' : '';
    if (abs >= 10000000) return `${sign}${(abs / 10000000).toFixed(1)}Cr`;
    if (abs >= 100000) return `${sign}${(abs / 100000).toFixed(1)}L`;
    if (abs >= 1000) return `${sign}${(abs / 1000).toFixed(1)}K`;
    return `${sign}${abs.toFixed(digits)}`;
}

function setElementTitle(id, title) {
    const el = document.getElementById(id)?.closest('.intel-card') || document.getElementById(id);
    if (el && title) el.title = title;
}

function formatSignedPct(value, digits = 2) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '--';
    return `${num >= 0 ? '+' : ''}${num.toFixed(digits)}%`;
}

function classifyTapeItem(item) {
    const msg = item.cleanMsg.toLowerCase();
    const type = String(item.type || '').toLowerCase();
    const hasVolume = /\b(volume|vol|power-move|pressure)\b/.test(msg);
    const isRejected = /\b(rejected|blocked|below selected grade|confidence below|grade rank below|discarded|ignored)\b/.test(msg);
    const isExit = /\b(exit|closed|target hit|sl hit|stop)\b/.test(msg);
    const isSignal = /\b(signal|matched|entry|buy|sell|long|short)\b/.test(msg) || type === 'trade';
    const isAlert = type === 'error' || type === 'warning' || /\b(alert|freeze|hang|stale|latency high)\b/.test(msg);
    const isData = type === 'data' || /\b(cache|sync|source|gateway|fyers|angel|yahoo)\b/.test(msg);
    if (hasVolume) return 'pressure';
    if (isAlert) return 'alert';
    if (isSignal && isRejected) return 'rejected';
    if (isExit) return 'exit';
    if (isSignal) return 'signal';
    if (isData) return 'data';
    return 'system';
}

function summarizeTapeMessage(item, kind) {
    let text = item.cleanMsg;
    if (kind === 'rejected') {
        text = text
            .replace(/^Signal rejected:\s*/i, '')
            .replace(/\s+-\s+/g, ' · ')
            .replace(/Confidence below 50%/i, 'confidence < 50%');
    } else if (kind === 'pressure') {
        text = text
            .replace(/^POWER-MOVE confirmed:\s*/i, '')
            .replace(/\boverride eligible\.?/i, 'override-ready')
            .replace(/\s+/g, ' ');
    } else if (kind === 'signal') {
        text = text.replace(/^Potential\s+/i, '').replace(/^Manual\s+/i, '');
    }
    return text;
}

function updateActivityLog(logs = []) {
    if (logs && !Array.isArray(logs) && typeof logs === 'object') {
        logs = Object.values(logs);
    }
    if (!Array.isArray(logs)) logs = [];
    const seen = new Set();
    state.fullLog = logs
        .filter(Boolean)
        .map(item => ({
            id: item.id || '',
            time: item.time || item.timestamp || '',
            msg: item.msg || item.message || '',
            type: item.type || 'system'
        }))
        .filter(item => {
            const key = item.id || `${item.time}|${item.type}|${cleanLogMessage(item.msg)}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    renderLogs();
}


function generateTapeCardHtml(item) {
    const msgLower = item.cleanMsg.toLowerCase();
    const kind = classifyTapeItem(item);
    const summary = summarizeTapeMessage(item, kind);
    const isLong = msgLower.includes('long') || msgLower.includes('buy');
    const isShort = msgLower.includes('short') || msgLower.includes('sell');
    
    if (kind === 'pressure') {
        const volClass = isLong ? 'vol-buy' : 'vol-sell';
        const volColor = isLong ? 'var(--buy)' : 'var(--sell)';
        const title = isLong ? 'BUY PRESSURE' : 'SELL PRESSURE';
        
        return {
            className: `tape-card tape-volume ${volClass}`,
            html: `
                <div class="tape-header">
                    <span class="tape-time">${item.timeText}</span>
                    <span class="tape-dir vol-badge" style="color:${volColor}">${title}</span>
                </div>
                <div class="tape-body">${summary}</div>
                <div class="tape-progress"><div class="tape-bar" style="background:${volColor}; width: 85%;"></div></div>
            `
        };
    } else if (kind === 'signal' || kind === 'rejected' || kind === 'exit') {
        const sigClass = isLong ? 'sig-long' : (isShort ? 'sig-short' : '');
        const tone = kind === 'rejected' ? 'REJECTED' : kind === 'exit' ? 'EXIT' : (isLong ? 'LONG SIGNAL' : (isShort ? 'SHORT SIGNAL' : 'SIGNAL'));
        const dirClass = kind === 'rejected' ? 'neutral' : isLong ? 'long' : isShort ? 'short' : 'neutral';
        const dirBadge = `<span class="tape-dir ${dirClass}">${tone}</span>`;
        
        return {
            className: `tape-card tape-signal tape-${kind} ${sigClass}`,
            html: `
                <div class="tape-header">
                    <span class="tape-time">${item.timeText}</span>
                    ${dirBadge}
                </div>
                <div class="tape-body">${summary}</div>
            `
        };
    } else {
        const label = kind === 'alert' ? 'ALERT' : kind === 'data' ? 'DATA' : item.typeLabel;
        return {
            className: `tape-card tape-system tape-${kind}`,
            html: `
                <div class="tape-header">
                    <span class="tape-time">${item.timeText}</span>
                    <span class="tape-time" style="opacity:0.6">[${label}]</span>
                </div>
                <div class="tape-body">${summary}</div>
            `
        };
    }
}

function renderLogs() {
    const logEl = document.getElementById('activityLog');
    if (!logEl) return;
    if (state.logFilter === 'engine_lab') {
        renderEngineLabTape(state.latestData?.engine_lab || {});
        return;
    }

    const filtered = state.logFilter === 'all' 
        ? state.fullLog 
        : state.fullLog.filter(item => {
            if (state.logFilter === 'error') return item.type === 'error' || item.type === 'warning';
            return item.type === state.logFilter;
        });

    const dataString = JSON.stringify(filtered);
    if (state.lastRenderedLogData === dataString) return;
    state.lastRenderedLogData = dataString;

    const newItems = filtered.map(item => {
        const type = item.type || 'system';
        const typeLabel = type === 'error' ? 'ALERT' : type.toUpperCase();
        const cleanMsg = escapeHtml(cleanLogMessage(item.msg));
        const timeText = formatLogTime(item.time);
        const key = item.id || `${timeText}|${type}|${cleanMsg}`;
        return { key, type, typeLabel, cleanMsg, timeText };
    });

    const currentChildren = Array.from(logEl.children);
    const filterChanged = state.lastRenderedFilter !== state.logFilter;
    state.lastRenderedFilter = state.logFilter;

    if (filterChanged || currentChildren.length === 0 || newItems.length === 0) {
        logEl.innerHTML = newItems.length ? newItems.map(item => {
            const card = generateTapeCardHtml(item);
            return `<div class="${card.className}" data-key="${item.key}">${card.html}</div>`;
        }).join('') : `
            <div class="tape-card tape-system" data-key="empty">
                <div class="tape-header"><span class="tape-time">${formatLogTime('')}</span></div>
                <div class="tape-body">Waiting for tape activity...</div>
            </div>
        `;
        return;
    }

    const c0_key = currentChildren[0].getAttribute('data-key');
    const matchIndex = newItems.findIndex(item => item.key === c0_key);

    if (matchIndex === -1) {
        logEl.innerHTML = newItems.map(item => {
            const card = generateTapeCardHtml(item);
            return `<div class="${card.className}" data-key="${item.key}">${card.html}</div>`;
        }).join('');
    } else if (matchIndex > 0) {
        // Prepend new items
        for (let i = matchIndex - 1; i >= 0; i--) {
            const item = newItems[i];
            const div = document.createElement('div');
            const card = generateTapeCardHtml(item);
            div.className = card.className;
            div.setAttribute('data-key', item.key);
            div.innerHTML = card.html;
            logEl.insertBefore(div, logEl.firstChild);
        }
    }

    // Trim extra items from bottom
    while (logEl.children.length > newItems.length) {
        logEl.lastChild.remove();
    }
}

function appendLogItem(container, item) {
    // Legacy - replaced by renderLogs
}


function updateInstrumentSelectors(instruments, dashboardContext = {}) {
    INDEX_NAMES.forEach(name => {
        const d = instruments[name];
        if (!d) return;
        
        // Update Price & Change
        const ltp = Number(d.ltp || d.spot_price || 0);
        const pct = Number(d.change_pct || 0);
        const changePoints = resolveIndexChangePoints(d, ltp, pct);
        setText(`price_${name}`, formatPrice(ltp));
        const changeEl = document.getElementById(`change_${name}`);
        if (changeEl) {
            changeEl.textContent = `${formatPointChange(changePoints)} (${pct.toFixed(2)}%)`;
            changeEl.className = 'inst-change ' + (changePoints > 0 ? 'green' : changePoints < 0 ? 'red' : 'neutral');
        }
        const sessionLabel = normalizeQuoteLabel(d, dashboardContext);
        const tabEl = document.getElementById(`tab_${name}`);
        if (tabEl) {
            tabEl.dataset.quoteSession = sessionLabel;
            tabEl.title = quoteTooltip(d, sessionLabel);
        }

        const quoteEl = document.getElementById(`quote_${name}`);
        if (quoteEl) {
            const age = Number(d.quote_age_seconds);
            const explicitlyStale = d.quote_fresh === false || d.quote_stale === true;
            // The backend dictates freshness based on the source's exchange capabilities.
            // BSE indices naturally tick every 5-6s, so hardcoding >3s here causes false STALE badges.
            const stale = sessionLabel === 'LIVE' && (explicitlyStale || !Number.isFinite(age));
            const displayLabel = stale ? 'STALE' : sessionLabel;
            quoteEl.textContent = `${displayLabel}·${formatQuoteAge(age)}`;
            quoteEl.className = 'inst-quote-status ' + quoteStatusClass(displayLabel);
            quoteEl.title = quoteTooltip(d, displayLabel);
            quoteEl.setAttribute('aria-label', `${name} quote ${displayLabel}, ${formatQuoteAge(age)} old`);
        }

        // Update Signal
        const signalEl = document.getElementById(`signal_${name}`);
        const mtf = d.mtf || {};
        const signal = (mtf.confluence_signal || 'SCANNING').toUpperCase();
        const stateClass = signal === 'BUY' ? 'state-buy' : signal === 'SELL' ? 'state-sell' : signal === 'SCANNING' ? 'state-scanning' : 'state-hold';
        if (tabEl) {
            tabEl.classList.remove('state-buy', 'state-sell', 'state-hold', 'state-scanning');
            tabEl.classList.add(stateClass);
        }
        if (signalEl) {
            signalEl.textContent = sessionLabel === 'LIVE' ? signal : sessionLabel;
            signalEl.className = 'inst-signal ' + (signal === 'BUY' ? 'buy' : signal === 'SELL' ? 'sell' : 'neutral');
            signalEl.classList.toggle('preopen', sessionLabel === 'PRE-OPEN' || sessionLabel === 'SETTLE');
            signalEl.classList.toggle(
                'reference',
                sessionLabel === 'REF' || sessionLabel === 'CLOSED' || sessionLabel === 'HISTORICAL'
            );
        }
    });
}

function normalizeQuoteLabel(d, dashboardContext = {}) {
    // Quote packets describe the exchange session, while the dashboard context also
    // describes whether this process is replaying history.  A cached packet can retain
    // quote_label=LIVE across either transition; never let that packet overrule the
    // current operator-facing runtime/session state.
    const runtimeMode = String(dashboardContext?.mode || '').toUpperCase();
    const marketStatus = String(dashboardContext?.market_status || '').toUpperCase();
    if (runtimeMode === 'HISTORICAL' || marketStatus === 'HISTORICAL') return 'HISTORICAL';
    if (marketStatus === 'CLOSED') return 'CLOSED';
    if (d?.preopen_settlement === true) return 'SETTLE';
    const raw = String(d.quote_label || d.quote_session || '').toUpperCase();
    if (raw.includes('PRE')) return 'PRE-OPEN';
    if (raw === 'LIVE' || raw === 'OPEN') return 'LIVE';
    if (raw === 'CLOSED') return 'CLOSED';
    return 'REF';
}

function quoteTooltip(d, label) {
    const source = d.quote_source || 'unknown';
    const age = Number(d.quote_age_seconds);
    const ageText = Number.isFinite(age) ? `${age.toFixed(1)}s old` : 'age unavailable';
    if (label === 'SETTLE') return `Pre-open settlement quote from ${source}; ${ageText}. Used as the index-card LTP before the normal session opens.`;
    if (label === 'PRE-OPEN') return `Pre-open quote from ${source}; ${ageText}. Indicative/reference values may not tick like normal market data.`;
    if (label === 'LIVE') return `Live quote from ${source}; ${ageText}.`;
    if (label === 'STALE') return `Stale live-session quote from ${source}; ${ageText}. Do not use this badge as an execution mark.`;
    return `Reference/closed-session quote from ${source}; ${ageText}.`;
}

function formatQuoteAge(age) {
    if (!Number.isFinite(age) || age < 0) return '--';
    if (age < 10) return `${age.toFixed(1)}s`;
    if (age < 60) return `${Math.round(age)}s`;
    if (age < 3600) return `${Math.floor(age / 60)}m`;
    return `${Math.floor(age / 3600)}h`;
}

function quoteStatusClass(label) {
    if (label === 'LIVE') return 'live';
    if (label === 'STALE') return 'stale';
    if (label === 'PRE-OPEN') return 'preopen';
    if (label === 'SETTLE') return 'settlement';
    if (label === 'CLOSED') return 'closed';
    return 'reference';
}

function resolveIndexChangePoints(d, ltp, pct) {
    const direct = [d.change_points, d.change_abs, d.net_change, d.points_change, d.change].find(v => Number.isFinite(Number(v)));
    if (direct !== undefined) return Number(direct);

    const prevClose = Number(d.prev_close || d.previous_close || d.close_previous || 0);
    if (Number.isFinite(prevClose) && prevClose > 0 && Number.isFinite(ltp)) return ltp - prevClose;

    if (!Number.isFinite(ltp) || !Number.isFinite(pct) || Math.abs(100 + pct) < 0.0001) return 0;
    return ltp - (ltp / (1 + pct / 100));
}

function formatPointChange(value) {
    if (!Number.isFinite(value)) return '--';
    const sign = value < 0 ? '-' : '';
    return sign + Math.abs(value).toLocaleString('en-IN', { useGrouping: false, minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function hasScannerTfState(tfState) {
    if (!tfState || typeof tfState !== 'object') return false;
    if (tfState.position_label) return true;
    return ['trailing_stop', 'last_entry_price', 'last_exit_price', 'stop_distance', 'adx_value', 'atr_value']
        .some(key => Number.isFinite(Number(tfState[key])) && Number(tfState[key]) !== 0);
}

function resolveScannerTfState(inst, timeframe) {
    const mtf = inst.mtf || {};
    const chart = inst.chart || {};
    const mtfState = mtf[`state_${timeframe}`] || {};
    const chartState = ((chart[timeframe] || {}).state) || {};
    // /api/state intentionally lightens chart payloads. The MTF state is the fresh
    // scanner-engine state for every index/timeframe; chart state is only a fallback.
    return hasScannerTfState(mtfState) ? mtfState : (hasScannerTfState(chartState) ? chartState : {});
}

function resolveScannerTfSource(inst, timeframe) {
    const mtf = inst.mtf || {};
    const chart = inst.chart || {};
    if (hasScannerTfState(mtf[`state_${timeframe}`] || {})) return 'MTF scanner';
    if (hasScannerTfState(((chart[timeframe] || {}).state) || {})) return 'Chart fallback';
    return 'Waiting';
}

function scannerEngineLabel(raw) {
    const mode = String(raw || 'utbot').toLowerCase();
    return {
        utbot: 'UT1',
        ema: 'EMA 9/21',
        consensus: 'Consensus ST+EMA',
        utbot_consensus: 'UT1 + Consensus',
        quad_union: 'Quad Union',
        quad_prime: 'Quad Prime',
        quad_prime_s1: 'Quad Prime + S1',
        ut_smart_s1: 'UT Smart Exit + S1',
        adaptive_dual: 'Adaptive Dual',
        s1: 'S1 Confluence',
        ut_smart: 'UT Smart Exit',
        s1s2_dual: 'S1+S2 Dual (Pine v5.4)',
        ut_ema: 'ut/ema',
    }[mode] || titleCaseToken(mode);
}

function formatSignedPoints(value) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return '0.00';
    const sign = num > 0 ? '+' : num < 0 ? '-' : '';
    return `${sign}${Math.abs(num).toFixed(2)}`;
}

function engineLabClass(value) {
    const num = Number(value || 0);
    return num > 0 ? 'pos' : num < 0 ? 'neg' : 'flat';
}

function resolveEngineLabActualSummary(data = {}) {
    const trades = data.trades || {};
    const summary = trades.summary || {};
    const monetary = trades.monetary || {};
    const headline = monetary.headline || {};
    const openRows = Array.isArray(trades.open) ? trades.open.length : 0;
    const closedTotal = Number(summary.total_trades ?? trades.closed_total ?? 0);
    const pnl = Number(summary.daily_pnl ?? summary.total_pnl ?? 0);
    return {
        mode: String(data.mode || '').toUpperCase(),
        trades: Number.isFinite(closedTotal) ? closedTotal : 0,
        open: Number(summary.open_count ?? openRows ?? 0),
        wins: Number(summary.wins ?? 0),
        losses: Number(summary.losses ?? 0),
        winRate: Number(summary.win_rate ?? headline.win_rate ?? 0),
        pnl,
        futPnl: Number(summary.fut_pnl ?? 0),
        optPnl: Number(summary.opt_pnl ?? 0),
        profitFactor: Number(summary.profit_factor ?? headline.profit_factor ?? 0),
        sharpe: Number(summary.sharpe_ratio ?? headline.sharpe_ratio ?? 0),
        sharpeDisplay: String(summary.sharpe_ratio_display || headline.sharpe_ratio_display || ''),
        maxDrawdown: Number(summary.max_drawdown ?? headline.max_drawdown ?? 0),
    };
}

function buildEngineLabDisplayRows(lab = {}, data = {}) {
    const rows = Array.isArray(lab.rows) ? lab.rows : [];
    const actual = resolveEngineLabActualSummary(data);
    const modeLabel = actual.mode === 'HISTORICAL' ? 'TRADE MANAGER BACKTEST' : actual.mode === 'REAL' ? 'TRADE MANAGER LIVE' : 'TRADE MANAGER';
    return rows.map(row => {
        const active = row.active === true;
        const hasReplayScore = (row.total_points !== undefined && row.total_points !== null)
            || (row.net_points !== undefined && row.net_points !== null);
        const hasActual = row.actual_result === true;
        if (active && hasActual) {
            return {
                ...row,
                display_basis: row.display_basis || modeLabel,
                raw_signals: row.raw_signals ?? row.signals ?? 0,
                signals: actual.trades,
                open: actual.open,
                closed: actual.trades,
                wins: actual.wins,
                losses: actual.losses,
                win_rate: actual.winRate,
                total_money: actual.pnl,
                fut_pnl: actual.futPnl,
                opt_pnl: actual.optPnl,
                profit_factor: actual.profitFactor,
                sharpe: actual.sharpe,
                sharpe_display: actual.sharpeDisplay,
                max_drawdown: actual.maxDrawdown,
                active_actual: null,
            };
        }
        return {
            ...row,
            display_basis: row.display_basis || (hasReplayScore ? 'SPOT REPLAY' : 'RAW SHADOW'),
            active_actual: active && !hasActual ? {
                label: modeLabel,
                trades: actual.trades,
                open: actual.open,
                wins: actual.wins,
                losses: actual.losses,
                win_rate: actual.winRate,
                pnl: actual.pnl,
                fut_pnl: actual.futPnl,
                opt_pnl: actual.optPnl,
                profit_factor: actual.profitFactor,
                sharpe: actual.sharpe,
                sharpe_display: actual.sharpeDisplay,
                max_drawdown: actual.maxDrawdown,
            } : null,
        };
    });
}

function updateEngineLabPanel(lab = {}, data = state.latestData || {}) {
    const summary = lab.summary || {};
    const rows = buildEngineLabDisplayRows(lab, data);
    const active = lab.active_engine || data?.config?.scanner_engine || state.latestData?.config?.scanner_engine || '';
    const badge = document.getElementById('engineLabTabBadge');
    if (badge) {
        badge.textContent = lab.enabled === false ? 'OFF' : (rows.length ? String(rows.length) : '--');
        badge.classList.toggle('has-data', lab.enabled !== false && rows.length > 0);
        badge.title = lab.enabled === false
            ? 'Engine Lab disabled'
            : `${rows.length || 0} engines · ${summary.signals || 0} signals · ${summary.fresh_signals || 0} fresh`;
    }
    if (state.logFilter === 'engine_lab') renderEngineLabTape(lab, data);
}

function renderEngineLabTape(lab = {}, data = state.latestData || {}) {
    const logEl = document.getElementById('activityLog');
    if (!logEl) return;
    const rows = buildEngineLabDisplayRows(lab, data);
    const summary = lab.summary || {};
    const active = lab.active_engine || data?.config?.scanner_engine || state.latestData?.config?.scanner_engine || '';
    const actualCount = rows.filter(row => row.actual_result === true).length;
    const replayCount = rows.filter(row =>
        (row.total_points !== undefined && row.total_points !== null)
        || (row.net_points !== undefined && row.net_points !== null)
    ).length;
    const staleShadow = rows.some(row =>
        row.actual_result !== true
        && (row.total_points === undefined || row.total_points === null)
        && (row.net_points === undefined || row.net_points === null)
    );
    const dataKey = JSON.stringify({
        enabled: lab.enabled,
        active,
        updated_at: lab.updated_at,
        trade_summary: data?.trades?.summary || {},
        rows: rows.map(row => [
            row.engine,
            row.active,
            row.actual_result,
            row.display_basis,
            row.signals,
            row.wins,
            row.losses,
            row.open,
            row.total_points,
            row.total_money,
            row.raw_signals,
            row.lane_A_pnl,
            row.lane_B_pnl,
            row.dual_concurrent_pnl,
            row.lane_jaccard,
            row.raw_lane_A_pnl,
            row.raw_lane_B_pnl,
            row.raw_dual_concurrent_pnl,
            row.raw_lane_jaccard,
            row.active_actual,
            row.sharpe,
            row.profit_factor,
        ]),
    });
    if (state.lastRenderedEngineLabData === dataKey) return;
    state.lastRenderedEngineLabData = dataKey;
    const previousTable = logEl.querySelector('.engine-lab-tape-table');
    const previousScrollTop = previousTable ? previousTable.scrollTop : 0;

    if (lab.enabled === false) {
        logEl.innerHTML = `
            <div class="engine-lab-tape-empty">
                <b>Engine Lab OFF</b>
                <span>Turn on 12-Engine Lab in Settings → System Controls.</span>
            </div>`;
        return;
    }
    if (!rows.length) {
        logEl.innerHTML = `
            <div class="engine-lab-tape-empty">
                <b>Waiting for Engine Lab scan</b>
                <span>Rows appear after the next scan/recalculation.</span>
            </div>`;
        return;
    }

    const header = `
        <div class="engine-lab-tape-head">
            <div><span>Active</span><b>${escapeHtml(scannerEngineLabel(active))}</b></div>
            <div><span>${actualCount ? 'Actual' : 'Replay'}</span><b>${actualCount || replayCount}/11</b></div>
            <div><span>Raw Signals</span><b>${summary.signals || 0}</b></div>
            <div><span>${staleShadow ? 'Status' : 'Speed'}</span><b>${staleShadow ? 'Restart' : (summary.latency_ms !== undefined ? `${summary.latency_ms}ms` : '--')}</b></div>
        </div>`;
    const body = rows.slice(0, 11).map((row, idx) => {
        const isActual = row.actual_result === true;
        const total = Number(row.total_points ?? row.net_points ?? 0);
        const money = Number(row.total_money || 0);
        const openPts = Number(row.open_points || 0);
        const signal = row.last_signal || {};
        const basis = row.display_basis || 'RAW SHADOW';
        const staleRow = !isActual
            && (row.total_points === undefined || row.total_points === null)
            && (row.net_points === undefined || row.net_points === null);
        const signalText = signal.type
            ? `${escapeHtml(signal.instrument || '--')} ${escapeHtml(signal.timeframe || '--')} ${escapeHtml(signal.type || '--')}`
            : '--';
        const openPreview = Array.isArray(row.open_rows) && row.open_rows.length
            ? row.open_rows.slice(0, 2).map(open => `${escapeHtml(open.instrument || '--')} ${escapeHtml(open.direction || '--')} ${formatSignedPoints(open.pnl_points || 0)}`).join(' · ')
            : 'No live shadow trade';
        const activeActual = row.active_actual || null;
        const actualText = activeActual
            ? ` · ${activeActual.label}: ${activeActual.trades || 0}T ${activeActual.wins || 0}/${activeActual.losses || 0} ${formatMoney(activeActual.pnl || 0, true)}`
            : '';
        const isAdaptiveDual = String(row.engine || '').toLowerCase() === 'adaptive_dual';
        const laneA = Number(row.raw_lane_A_pnl ?? row.lane_A_pnl ?? 0);
        const laneB = Number(row.raw_lane_B_pnl ?? row.lane_B_pnl ?? 0);
        const dualPts = Number(row.raw_dual_concurrent_pnl ?? row.dual_concurrent_pnl ?? 0);
        const laneJ = Number(row.raw_lane_jaccard ?? row.lane_jaccard ?? 0);
        const laneText = isAdaptiveDual
            ? ` · lanes A ${formatSignedPoints(laneA)} / B ${formatSignedPoints(laneB)} · dual ${formatSignedPoints(dualPts)} · J ${(laneJ * 100).toFixed(0)}%`
            : '';
        const detailText = isActual
            ? `${basis} · raw ${row.raw_signals || 0} signals${laneText} · OPT ${formatMoney(row.opt_pnl || 0, true)} / FUT ${formatMoney(row.fut_pnl || 0, true)}`
            : `${basis}${staleRow ? ' · backend restart pending' : ''} · ${signalText} · ${openPreview}${actualText}${laneText}`;
        const primaryValue = isActual ? formatMoney(money, true) : formatSignedPoints(total);
        const primaryLabel = isActual ? 'P&L' : 'Gain Pts';
        const primarySub = isActual
            ? `DD ${formatMoney(row.max_drawdown || 0, true)}`
            : `PF ${Number(row.profit_factor || 0).toFixed(2)}`;
        const sharpeText = row.sharpe_display || Number(row.sharpe || 0).toFixed(2);
        return `
            <div class="engine-lab-tape-row ${row.active ? 'active' : ''} ${isActual ? 'actual' : ''}">
                <div class="engine-rank">${idx + 1}</div>
                <div class="engine-main">
                    <b>${escapeHtml(scannerEngineLabel(row.engine))}</b>
                    <em>${escapeHtml(basis)}</em>
                    <span>${detailText}</span>
                </div>
                <div class="engine-stat">
                    <span>${isActual ? 'Trades' : 'Signals'}</span>
                    <b>${row.signals || 0}</b>
                    <small>${isActual ? `raw ${row.raw_signals || 0}` : `${row.buy || 0}B / ${row.sell || 0}S`}</small>
                </div>
                <div class="engine-stat">
                    <span>W / L</span>
                    <b>${row.wins || 0} / ${row.losses || 0}</b>
                    <small>${Number(row.win_rate || 0).toFixed(1)}%</small>
                </div>
                <div class="engine-stat ${engineLabClass(isActual ? 0 : openPts)}">
                    <span>${isActual ? 'Open' : 'Live'}</span>
                    <b>${row.open || 0}</b>
                    <small>${isActual ? 'actual' : formatSignedPoints(openPts)}</small>
                </div>
                <div class="engine-stat ${engineLabClass(isActual ? money : total)}">
                    <span>${primaryLabel}</span>
                    <b>${primaryValue}</b>
                    <small>${primarySub}</small>
                </div>
                <div class="engine-stat ${engineLabClass(row.sharpe)}">
                    <span>Sharpe</span>
                    <b>${escapeHtml(sharpeText)}</b>
                    <small>PF ${Number(row.profit_factor || 0).toFixed(2)}</small>
                </div>
            </div>`;
    }).join('');
    logEl.innerHTML = `<div class="engine-lab-tape">${header}<div class="engine-lab-tape-table">${body}</div></div>`;
    const nextTable = logEl.querySelector('.engine-lab-tape-table');
    if (nextTable) nextTable.scrollTop = previousScrollTop;
}

function formatOneDecimal(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(1) : '--';
}

function updateInfoPanel(inst) {
    if (!inst) return;
    
    const mtf = inst.mtf || {};
    const tf = state.currentTimeframe;
    const tfState = resolveScannerTfState(inst, tf);
    const engineLabel = scannerEngineLabel(state.latestData?.config?.scanner_engine);
    const sourceLabel = resolveScannerTfSource(inst, tf);

    setText('infoEngine', engineLabel);
    setText('infoTimeframe', tf);
    setText('infoStateSource', sourceLabel);

    const pos = tfState.position_label || 'FLAT';
    const posEl = document.getElementById('infoPosition');
    if (posEl) {
        posEl.textContent = pos;
        posEl.className = 'info-value ' + (pos === 'LONG' ? 'green' : pos === 'SHORT' ? 'red' : '');
        posEl.title = `${engineLabel} ${tf} scanner state from ${sourceLabel}`;
    }

    const intel = inst.intelligence || {};
    const agg = intel.aggregate || {};
    const tradeState = inst.trade_state || {};

    setText('infoEntry', formatPrice(tfState.last_entry_price));
    setText('infoExit', formatPrice(tfState.last_exit_price));
    setText('infoStop', formatPrice(tfState.trailing_stop));
    setText('infoStopDist', formatPrice(tfState.stop_distance) + ' pts');
    setText('infoADX', formatOneDecimal(tfState.adx_value));
    setText('infoConfluence', (mtf.confluence_score || 0).toFixed(2));
    
    setText('infoIntel', agg.score !== undefined ? agg.score.toFixed(0) : '--');
    
    setText('infoOpenMTM', tradeState.open_mtm !== undefined ? formatMoney(tradeState.open_mtm) : '--');
    if (tradeState.open_mtm > 0) {
        document.getElementById('infoOpenMTM').className = 'info-value green glow-text';
    } else if (tradeState.open_mtm < 0) {
        document.getElementById('infoOpenMTM').className = 'info-value red glow-text';
    } else {
        document.getElementById('infoOpenMTM').className = 'info-value glow-text';
    }
    
    setText('infoTimeInTrade', tradeState.time_in_trade || '--');
    setText('infoRisk', tradeState.risk_exposure !== undefined ? formatMoney(tradeState.risk_exposure) : '--');
    setText('infoNextTrigger', tradeState.next_trigger || '--');
    setElementTitle('infoNextTrigger', `Active scanner engine: ${engineLabel}`);
    
    setText('intelBreadth', tradeState.overbought_oversold || '--');
    setText('intelVwap', tradeState.vwap_deviation || '--');
    setText('intelATR', tradeState.atr_exhaustion || '--');

    updateMTFLabels(mtf);
}

function updateHardwarePanel(metrics = {}) {
    const ram = metrics.ram || {};
    const disk = metrics.disk || {};
    const gpu = metrics.gpu || {};
    const processRam = Number(metrics.process_ram_mb || 0);
    const processCpu = Number(metrics.process_cpu_pct || 0);
    const processSingleCoreCpu = Number(metrics.process_cpu_single_core_pct || 0);
    const threads = Number(metrics.threads || 0);
    const fmtGb = value => Number(value || 0).toLocaleString('en-IN', { maximumFractionDigits: 1 });
    const fmtMb = value => Number(value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });

    setText('hwRam', `${Number(ram.used_pct || 0).toFixed(1)}%`);
    setText('hwRamSub', `${fmtMb(ram.used_mb)} / ${fmtMb(ram.total_mb)} MB`);

    setText('hwProcess', `${fmtMb(processRam)} MB`);
    const cpuDetail = processSingleCoreCpu > 0
        ? `${processCpu.toFixed(1)}% CPU (${processSingleCoreCpu.toFixed(0)}% core)`
        : `${processCpu.toFixed(1)}% CPU`;
    setText('hwProcessSub', `${cpuDetail} / ${threads || '--'} workers`);

    setText('hwDisk', `${Number(disk.used_pct || 0).toFixed(1)}%`);
    setText('hwDiskSub', `${fmtGb(disk.free_gb)} GB free`);

    if (gpu.available) {
        const bench = gpu.benchmark || {};
        const speedLabel = bench.speedup > 0 ? ` / ${Number(bench.speedup).toFixed(2)}x bench` : '';
        const computeLabel = gpu.compute_available
            ? `compute ${gpu.compute_backend || 'ready'}${speedLabel}`
            : `${gpu.acceleration_mode || 'telemetry_only'}`;
        setText('hwGpu', `${Number(gpu.used_pct || 0).toFixed(1)}%`);
        setText('hwGpuSub', `${gpu.label || 'GPU'} / ${fmtMb(gpu.used_mb)} of ${fmtMb(gpu.total_mb)} MB / ${computeLabel}`);
    } else {
        setText('hwGpu', 'N/A');
        setText('hwGpuSub', gpu.acceleration_mode || 'No NVIDIA telemetry');
    }
}

function updateMTFLabels(mtf) {
    ['1min', '5min', '15min'].forEach(tf => {
        const s = mtf['state_' + tf] || {};
        const dotId = 'dot' + tf.replace('min', 'm');
        const el = document.getElementById(dotId);
        if (el) {
            const pos = (s.position_label || 'FLAT').toLowerCase();
            el.className = 'tf-pos-dot ' + pos;
        }
    });
}


function getSentimentBadge(text) {
    if (!text) return '';
    const txt = normalizeToken(text).toUpperCase();
    let colorClass = 'neutral';
    
    if (txt.includes('BEAR') || txt.includes('FEAR') || txt.includes('SHORT') || txt.includes('DOWN') || txt.includes('SELL')) colorClass = 'danger';
    else if (txt.includes('BULL') || txt.includes('GREED') || txt.includes('LONG') || txt.includes('UP') || txt.includes('BUY')) colorClass = 'success';
    else if (txt.includes('BREAKOUT') || txt.includes('SURGE')) colorClass = 'warning';
    
    return `<span class="sentiment-badge ${colorClass}">${txt}</span>`;
}

function updateIntelTile(id, value, label, title = '') {
    setText(id, value);
    if (label !== undefined && label !== null) setHtml(document.getElementById(`${id}Label`), label);
    setElementTitle(id, title);
}

function _normConf01(raw) {
    let conf = Number(raw);
    if (!Number.isFinite(conf)) return 0;
    if (conf > 1.5) conf = conf / 100.0;
    return Math.max(0, Math.min(1, conf));
}

/** Hover/title detail (full sentence) — short text stays in the cell. */
function _moodFullTooltip(h, role) {
    if (!h || typeof h !== 'object') return 'No session mood yet';
    const bias = String(h.bias || h.effective_bias || 'NEUTRAL').toUpperCase();
    const dt = String(h.day_type || h.effective_day_type || 'UNDECIDED').toUpperCase();
    const conf = _normConf01(h.confidence != null ? h.confidence : h.phase_confidence);
    const confPct = conf > 0.005 ? Math.round(conf * 100) : 0;
    const move = Number(h.move_pct);
    const mv = Number.isFinite(move) ? `${move >= 0 ? '+' : ''}${move.toFixed(2)}%` : 'n/a';
    const roleName = {
        day: 'Full session day type',
        pulse: 'Last 1–5 minutes',
        pocket: 'Rolling pocket',
        phase: 'Trade phase used by guards',
    }[role] || role;
    return `${roleName}: ${dt.replace(/_/g, ' ')} · bias ${bias}`
        + (confPct ? ` · ${confPct}% conf` : '')
        + ` · move ${mv}`;
}

/** Readable short labels (card is 2 rows tall — room for full words). */
function _moodPlainStory(h, role) {
    if (!h || typeof h !== 'object') return '—';
    const bias = String(h.bias || h.effective_bias || '').toUpperCase();
    const dt = String(h.day_type || h.effective_day_type || '').toUpperCase();
    const conf = _normConf01(h.confidence != null ? h.confidence : h.phase_confidence);
    const confPct = conf > 0.005 ? Math.round(conf * 100) : 0;
    const move = Number(h.move_pct);
    const hasMove = Number.isFinite(move) && Math.abs(move) >= 0.01;
    const mv = hasMove ? `${move >= 0 ? '+' : ''}${move.toFixed(2)}%` : '';

    let core = 'Wait';
    if (dt === 'TREND_DAY_UP' || (dt.includes('TREND') && dt.includes('UP'))) core = 'Bull trend';
    else if (dt === 'TREND_DAY_DOWN' || (dt.includes('TREND') && dt.includes('DOWN'))) core = 'Bear trend';
    else if (dt === 'MOOD_DUMP') core = 'Dump';
    else if (dt === 'MOOD_RALLY') core = 'Rally';
    else if (dt === 'MOOD_TRANSITION') core = 'Flip';
    else if (dt === 'RANGE_DAY' || dt === 'RANGING' || dt === 'CHOPPY') {
        core = role === 'phase' ? 'Range — wait break' : 'Range chop';
    } else if (bias === 'BULLISH') core = 'Bull lean';
    else if (bias === 'BEARISH') core = 'Bear lean';
    else if (dt && dt !== 'UNDECIDED') core = titleCaseToken(dt.replace(/_/g, ' ')).slice(0, 16);
    else if (!dt || dt === 'UNDECIDED') core = 'Forming';

    let out = core;
    if (mv) out += ` ${mv}`;
    if (confPct > 0) out += ` ${confPct}%`;
    return out;
}

function _moodToneClass(h) {
    const bias = String((h && (h.bias || h.effective_bias)) || '').toUpperCase();
    const dt = String((h && (h.day_type || h.effective_day_type)) || '').toUpperCase();
    if (bias === 'BULLISH' || dt.includes('UP') || dt.includes('RALLY')) return 'bullish';
    if (bias === 'BEARISH' || dt.includes('DOWN') || dt.includes('DUMP')) return 'bearish';
    return '';
}

function _regimePlainName(raw) {
    const r = String(raw || '').toUpperCase();
    if (!r || r === 'UNKNOWN') return 'Unknown';
    if (r.includes('TREND') && r.includes('UP')) return 'Trend Up';
    if (r.includes('TREND') && r.includes('DOWN')) return 'Trend Dn';
    if (r.includes('BREAKOUT')) return 'Breakout';
    if (r.includes('VOLATIL') || r.includes('VOLATILE')) return 'Volatile';
    if (r.includes('RANGE') || r === 'RANGING' || r === 'CHOPPY') return 'Ranging';
    if (r.includes('REVERSAL')) return 'Reversal';
    return titleCaseToken(r.replace(/_/g, ' '));
}

function _resolveInstrumentIntel(inst, data) {
    /** Prefer top-level intelligence map, fall back to instrument payload.
     *  Critical: empty `{}` map is truthy — must not block instrument.intelligence. */
    const name = String((state && state.currentInstrument) || '').toUpperCase();
    const map = data && data.intelligence;
    if (map && typeof map === 'object') {
        const direct = map[name] || map[state.currentInstrument];
        if (direct && typeof direct === 'object' && Object.keys(direct).length) {
            return direct;
        }
        const hit = Object.keys(map).find(
            (k) => String(k || '').toUpperCase() === name
        );
        if (hit && map[hit] && typeof map[hit] === 'object' && Object.keys(map[hit]).length) {
            return map[hit];
        }
    }
    const nested = inst && inst.intelligence;
    if (nested && typeof nested === 'object' && Object.keys(nested).length) {
        return nested;
    }
    return {};
}

function _regimeHorizonTone(code) {
    const r = String(code || '').toUpperCase();
    if (r.includes('UP') || r.includes('BULL')) return 'bullish';
    if (r.includes('DOWN') || r.includes('BEAR') || r.includes('DUMP')) return 'bearish';
    if (r.includes('VOLATIL') || r.includes('BREAK')) return 'volatile';
    return '';
}

function _fillRegimeHorizonRow(rowId, labelId, metaId, snap) {
    const row = document.getElementById(rowId);
    const lab = document.getElementById(labelId);
    const meta = document.getElementById(metaId);
    if (!lab) return false;
    const s = snap && typeof snap === 'object' ? snap : {};
    const code = String(s.display_regime || s.regime || 'UNKNOWN').toUpperCase();
    // Primary line = regime only (no ADX/ER/RSI — those live on other tiles).
    let text = s.label || _regimePlainName(code) || 'Warming up';
    // Strip accidental metric suffixes from backend labels.
    text = String(text).replace(/\s*·\s*dir\s+\w+/i, '').trim();
    lab.textContent = text;

    // Detail: hybrid day mix / thin-vol / conf — never ADX/ER/RSI/VWAP.
    const detail = [];
    if (s.detail) detail.push(String(s.detail));
    const vs = String(s.volume_state || '').toUpperCase();
    if (!s.detail) {
        if (vs === 'THIN' || vs === 'NO_VOLUME') detail.push('thin volume');
        else if (vs === 'SURGE') detail.push('vol surge');
        if (code.includes('BREAKOUT')) detail.push('breakout');
        if (code === 'CHOPPY' || (code === 'RANGING' && Number(s.adx) > 0 && Number(s.adx) < 16)) {
            if (!/choppy/i.test(text)) detail.push('chop');
        }
    }
    const conf = Number(s.confidence_pct);
    if (Number.isFinite(conf) && conf > 0) detail.push(`${conf}%`);
    if (meta) meta.textContent = detail.join(' · ');

    const tip = [
        s.title || '',
        code.replace(/_/g, ' '),
        s.direction ? `bias ${s.direction}` : '',
        vs && vs !== 'UNKNOWN' && vs !== 'NORMAL' ? vs : '',
        s.summary || '',
    ].filter(Boolean).join(' · ');
    lab.title = tip;
    if (row) {
        row.classList.remove('bullish', 'bearish', 'volatile');
        const tone = _regimeHorizonTone(code);
        if (tone) row.classList.add(tone);
        row.title = tip;
    }
    return Boolean(code && code !== 'UNKNOWN');
}

function updateRegimeMoodPanel(intel, config) {
    const regime = (intel && intel.regime) || {};
    const agg = (intel && intel.aggregate) || {};
    // Live multi-TF regimes only (1m / 5m / 15m engine) — no metric dump.
    const rh = (intel && intel.regime_horizons)
        || (agg && agg.regime_horizons)
        || {};
    const pulse = rh.pulse || {};
    const hour = rh.hour || rh.live || {};
    const day = rh.day || {};

    const primaryCode = regime.regime || regime.smoothed_regime || agg.regime || 'UNKNOWN';
    const fallbackSnap = {
        regime: primaryCode,
        display_regime: primaryCode,
        label: _regimePlainName(primaryCode),
        confidence_pct: (() => {
            const c = Number(regime.confidence);
            if (!Number.isFinite(c) || c <= 0) return 0;
            return c > 1.5 ? Math.round(c) : Math.round(c * 100);
        })(),
        volume_state: 'UNKNOWN',
        summary: regime.conviction_summary || '',
        title: 'Live 5m',
        direction: regime.direction || '',
    };
    _fillRegimeHorizonRow('rhPulse', 'moodMinute', 'moodMinuteMeta', pulse.regime ? pulse : fallbackSnap);
    _fillRegimeHorizonRow('rhHour', 'moodHour', 'moodHourMeta', hour);
    _fillRegimeHorizonRow('rhDay', 'moodDay', 'moodDayMeta', day);

    // Keep hidden stubs quiet if other code still writes them.
    setText('intelRegime', _regimePlainName(primaryCode));
    setText('intelRegimeLabel', String(regime.conviction_label || regime.direction || '').toUpperCase() || '—');
}

function updateDivergenceAlert(intel) {
    const alertEl = document.getElementById('manipulationAlert');
    if (!alertEl) return;
    const flow = (intel && intel.order_flow) || {};
    const rawRatio = flow.ratio;
    const ratio = Number(rawRatio);
    const usable = flow.available !== false && String(flow.source || '').toLowerCase() !== 'unavailable'
        && rawRatio !== null && rawRatio !== undefined && String(rawRatio).trim() !== ''
        && Number.isFinite(ratio) && ratio >= 0;
    const show = Boolean(intel && intel.divergence_alert === true && usable);
    alertEl.textContent = 'PRESSURE DIVERGENCE';
    alertEl.style.display = show ? 'block' : 'none';
    alertEl.title = show
        ? 'Price and estimated candle/depth pressure disagree. This heuristic does not establish market manipulation.'
        : '';
}

function intelNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function updateIntelPanel(inst, data) {
    if (!inst) return;
    const intel = _resolveInstrumentIntel(inst, data);
    updateDivergenceAlert(intel);

    const vol = intel.volume || {};
    const oi = intel.oi || {};
    const pcr = intel.pcr || {};
    const regime = intel.regime || {};
    const flow = intel.order_flow || {};
    const agg = intel.aggregate || {};
    const tradeState = inst.trade_state || {};
    const dataQuality = intel.data_quality || {};
    const profile = intel.market_structure || vol.market_structure_profile || {};
    const aroon = intel.aroon || vol.aroon || {};
    const wavetrend = intel.wavetrend || vol.wavetrend || {};
    const optionVolume = intel.option_volume || {};
    const optionQuantity = intelNumber(optionVolume.quantity);
    const optionVolumeAvailable = optionVolume.available === true && optionQuantity !== null && optionQuantity >= 0;
    const asOf = typeof intel.as_of === 'number' ? new Date(intel.as_of * 1000) : new Date(intel.as_of || '');
    const replay = String(data?.mode || '').toUpperCase() === 'HISTORICAL';
    const snapshotTime = Number.isFinite(asOf.getTime()) ? asOf.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
    }) : '';
    setText('intelDataAsOf', snapshotTime ? `${replay ? 'Replay' : 'As of'} · ${snapshotTime}` : 'Waiting for data');

    const hasTrustedRelativeVolume = Number(vol.volume_confidence || 0) >= 75 && Number(vol.time_of_day_avg_volume || 0) > 0;
    const rawVolRatio = vol.display_volume_ratio !== undefined
        ? vol.display_volume_ratio
        : (hasTrustedRelativeVolume ? vol.relative_volume_ratio : vol.volume_ratio);
    // Availability is opt-in.  An absent payload must never render as a real
    // zero because index spot candles commonly do not carry traded volume.
    const volAvailable = vol.available === true && String(vol.status || '').toUpperCase() !== 'UNAVAILABLE';
    const volRatio = Number(rawVolRatio !== undefined && rawVolRatio !== null ? rawVolRatio : NaN);
    setText('intelVolRatio', volAvailable && Number.isFinite(volRatio) ? `${volRatio.toFixed(1)}x`
        : (optionVolumeAvailable ? compactNumber(optionQuantity) : '--'));
    const volSurge = titleCaseToken(vol.surge_level || 'NORMAL');
    const buyPct = Number(vol.buy_pct !== undefined ? vol.buy_pct : 50);
    let volLabelText = volAvailable ? 'BALANCED' : (optionVolumeAvailable ? 'OPTIONS SESSION VOLUME' : 'VOLUME UNAVAILABLE');
    if (volAvailable && buyPct > 50.5) {
        volLabelText = `${volSurge} ${buyPct.toFixed(0)}% BUY`;
    } else if (volAvailable && buyPct < 49.5) {
        volLabelText = `${volSurge} ${(100 - buyPct).toFixed(0)}% SELL`;
    } else if (volAvailable) {
        volLabelText = `${volSurge} BALANCED`;
    }
    setHtml(document.getElementById('intelVolLabel'), getSentimentBadge(volLabelText));
    const volBasis = normalizeToken(vol.volume_ratio_basis || (hasTrustedRelativeVolume ? 'time_of_day' : 'rolling_avg'));
    const profileState = normalizeToken(profile.state || '');
    const profileSignal = normalizeToken(profile.signal || '');
    const profileSummary = profile.poc
        ? ` Profile ${profileSignal || profileState || 'neutral'}: POC ${formatPrice(profile.poc)}, VAH ${formatPrice(profile.vah)}, VAL ${formatPrice(profile.val)}.`
        : '';
    const volMetricName = vol.is_proxy ? 'Tick-activity proxy' : 'Volume';
    // Name the frame and the bar. A 5m bar is the sum of its five 1m bars, so the raw
    // volume agrees across timeframes while the ratio cannot: each frame compares a bar
    // against its own 20-bar history. Unlabelled, "0.7x" reads as a contradiction.
    const volFrame = String(vol.timeframe || vol.measurement_timeframe || '').toLowerCase().replace('min', 'm');
    const volBarStamp = String(vol.measurement_timestamp || '').slice(11, 16);
    // Say so when a vendor outlier was kept out of the 20-bar mean, otherwise the
    // average appears to move for no reason the operator can see.
    const volClipped = Number(vol.baseline_outliers_clipped || 0);
    const volClipNote = volClipped > 0
        ? ` ${volClipped} outlier bar${volClipped > 1 ? 's' : ''} above ${compactNumber(vol.baseline_outlier_cap)} excluded from the average (vendor spike).`
        : '';
    const volFrameEl = document.getElementById('intelVolFrame');
    if (volFrameEl) {
        volFrameEl.textContent = volAvailable && volFrame
            ? (volBarStamp ? `${volFrame} · bar ${volBarStamp}` : volFrame)
            : (optionVolumeAvailable ? 'Options' : '');
    }
    setElementTitle('intelVolRatio', volAvailable ? `${volMetricName} ratio ${volRatio.toFixed(2)}x (${volBasis}) on the ${volFrame || 'analysis'} frame${volBarStamp ? `, last complete bar ${volBarStamp}` : ''}. Current ${compactNumber(vol.current_volume)} vs avg ${compactNumber(vol.avg_volume)} for that frame. Trend: ${normalizeToken(vol.volume_trend || 'unknown')}. Confidence: ${vol.volume_confidence || 0}.${volClipNote}${profileSummary}`
        : (optionVolumeAvailable ? `Upstox call and put traded quantity for expiry ${optionVolume.expiry || 'shown'}.`
            : `Volume unavailable: ${vol.reason || 'no recorded volume'}.`));
    
    const volBar = document.getElementById('intelVolBar');
    if (volBar) {
        volBar.style.width = volAvailable && Number.isFinite(buyPct)
            ? Math.max(0, Math.min(100, buyPct)) + '%'
            : '0%';
        if (volAvailable && buyPct > 50.5) {
            volBar.style.background = 'var(--gradient-bull)';
        } else if (volAvailable && buyPct < 49.5) {
            volBar.style.background = 'var(--gradient-bear)';
        } else {
            volBar.style.background = 'var(--border)';
        }
    }
    
    const oiAvailable = oi.available === true && String(oi.status || '').toUpperCase() !== 'UNAVAILABLE';
    const absOi = intelNumber(oi.absolute_oi_change);
    const callOi = intelNumber(oi.total_call_oi);
    const putOi = intelNumber(oi.total_put_oi);
    const oiMeasuredChange = oi.change_comparable === true && absOi > 0;
    const oiVal = !oiAvailable ? '--' : (oiMeasuredChange ? compactNumber(absOi)
        : (callOi !== null && putOi !== null ? compactNumber(callOi + putOi) : '--'));
    setText('intelOI', oiVal);
    const oiLabel = oiAvailable
        ? (oiMeasuredChange ? 'OI CHANGE' : 'TOTAL OPEN INTEREST')
        : 'OPTIONS DATA UNAVAILABLE';
    setHtml(document.getElementById('intelOILabel'), getSentimentBadge(oiLabel));
    setElementTitle('intelOI', oiAvailable ? `Call OI ${compactNumber(callOi)}; put OI ${compactNumber(putOi)}; support ${oi.support_level || '--'}; resistance ${oi.resistance_level || '--'}; basis ${normalizeToken(oi.analysis_basis || 'unknown')}.` : `OI unavailable: ${dataQuality.reason || oi.reason || 'option chain unavailable'}.`);
    
    // PCR also requires an explicit authoritative-data acknowledgement.  Empty
    // objects and one-strike contract-lookup fallbacks are not PCR observations.
    const pcrAvailable = pcr.available === true && String(pcr.status || '').toUpperCase() !== 'UNAVAILABLE';
    const primaryPcr = intelNumber(pcr.primary_pcr ?? pcr.pcr_value ?? pcr.pcr_oi);
    setText('intelPCR', pcrAvailable && Number.isFinite(primaryPcr) ? primaryPcr.toFixed(2) : '--');
    const pcrSignal = pcrAvailable ? normalizeToken(pcr.contrarian_signal || pcr.signal || pcr.sentiment || 'NEUTRAL') : 'OPTIONS DATA UNAVAILABLE';
    setHtml(document.getElementById('intelPCRLabel'), getSentimentBadge(pcrSignal));
    setElementTitle('intelPCR', pcrAvailable ? `${pcr.interpretation || 'PCR contrarian sentiment.'} Basis: ${normalizeToken(pcr.pcr_basis || 'unknown')}.` : `PCR unavailable: ${dataQuality.reason || pcr.reason || 'option chain unavailable'}.`);
    
    // Regime tile: 3 live horizons only (1–5m / hour / day). No ADX/ER/RSI/VWAP here.
    try {
        updateRegimeMoodPanel(intel, data && data.config);
    } catch (e) { /* non-fatal */ }

    const flowAvailable = flow.available === true && String(flow.source || '').toLowerCase() !== 'unavailable';
    const flowRatio = intelNumber(flow.buy_sell_ratio ?? flow.ratio);
    setText('intelFlow', flowAvailable && Number.isFinite(flowRatio) ? `${flowRatio.toFixed(2)}x` : '--');
    const flowConfirm = flowAvailable
        ? (flow.multi_source_flow?.confirmation || flow.signal || 'NEUTRAL')
        : 'FLOW UNAVAILABLE';
    setHtml(document.getElementById('intelFlowLabel'), getSentimentBadge(flowConfirm));
    setElementTitle('intelFlow', flowAvailable
        ? `Order-flow proxy ${flowRatio.toFixed(2)}x from ${normalizeToken(flow.source || 'unknown')}; aggressor ${normalizeToken(flow.aggressor || 'none')}. CE/PE activity ${Number(flow.options_flow?.ce_pe_activity_ratio ?? 1).toFixed(2)}x.`
        : `Order-flow unavailable: ${flow.reason || flow.limitation || 'insufficient measured volume or depth'}.`);

    const maxPain = intel.max_pain || {};
    setText('intelMaxPain', maxPain.strike ? Math.round(maxPain.strike).toLocaleString('en-IN') : '--');
    if (maxPain.distance_pct !== undefined && maxPain.distance_pct !== null) {
        setText('intelMaxPainLabel', `${formatSignedPct(maxPain.distance_pct)} from spot`);
        setElementTitle('intelMaxPain', `Expiry magnet based on lowest option-buyer payout. Spot is ${formatSignedPct(maxPain.distance_pct)} from max pain.`);
    } else {
        setText('intelMaxPainLabel', 'Expiry magnet unavailable');
        setElementTitle('intelMaxPain', dataQuality.reason || 'Option-chain OI unavailable');
    }

    // VWAP lives inside Regime metrics (standalone VWAP tile removed).
    const vwapDev = intelNumber(vol.vwap_deviation_pct);
    const scVwap = intelNumber((intel.session_context || {}).vwap_deviation_pct);
    const vwapShow = Number.isFinite(vwapDev)
        ? formatSignedPct(vwapDev)
        : (Number.isFinite(scVwap) ? formatSignedPct(scVwap) : (tradeState.vwap_deviation || '--'));
    setText('intelVwap', vwapShow);
    if (document.getElementById('intelVwap')) {
        const profileShort = profile.state === 'ABOVE_VALUE' ? 'above VA'
            : (profile.state === 'BELOW_VALUE' ? 'below VA'
                : (profile.nearest_level ? `near ${profile.nearest_level}` : ''));
        setElementTitle(
            'intelVwap',
            `VWAP ${vwapShow}${profileShort ? ` · ${profileShort}` : ''}. `
            + `Engine VWAP ${vol.vwap || '--'}.${profileSummary}`
        );
    }
    const rsi = intelNumber(regime.rsi);
    setText('intelBreadth', tradeState.overbought_oversold ||
        (rsi !== null ? `${rsi.toFixed(1)} RSI` : '--'));
    const wtSummary = wavetrend.available
        ? ` WaveTrend ${normalizeToken(wavetrend.bias || 'neutral')} / ${normalizeToken(wavetrend.exhaustion || 'neutral')} (dashboard-only exhaustion).`
        : '';
    setElementTitle('intelBreadth', `RSI(14) on 5m closes. This replaces broad-market breadth because this system tracks four indices, not all stocks.${wtSummary}`);
    const atrValue = Number(tradeState.atr_value);
    const atrFrame = String(tradeState.atr_timeframe || '5min').toLowerCase();
    const atrAvailable = Number.isFinite(atrValue) && atrValue > 0;
    setText('intelATR', atrAvailable ? `ATR ${atrValue.toFixed(2)} pts` : '--');
    setText('intelATRLabel', atrAvailable ? `${atrFrame} volatility` : 'ATR unavailable');
    setElementTitle('intelATR', atrAvailable
        ? `ATR ${atrValue.toFixed(2)} points on ${atrFrame}. Source: ${normalizeToken(tradeState.atr_reason || 'scanner state')}.`
        : `ATR unavailable: ${tradeState.atr_reason || 'missing 5m candle state'}.`);

    const greeks = intel.greeks || {};
    const vix = intelNumber(greeks.volatility_value);
    const ivp = intelNumber(greeks.volatility_percentile);
    const callDelta = greeks.is_fallback ? null : intelNumber(greeks.call?.delta);
    const putDelta = greeks.is_fallback ? null : intelNumber(greeks.put?.delta);
    const theta = greeks.is_fallback ? null : intelNumber(greeks.call?.theta);
    setText('greekIV', vix > 0 ? vix.toFixed(1) : '--');
    setText('greekCallDelta', callDelta !== null ? `+₹${Math.abs(callDelta * 100).toFixed(0)}` : '--');
    setText('greekPutDelta', putDelta !== null ? `+₹${Math.abs(putDelta * 100).toFixed(0)}` : '--');
    setText('greekTheta', theta !== null ? `₹${Math.abs(theta).toFixed(1)}/day` : '--');
    const vixLabel = greeks.volatility_percentile_available === true && ivp !== null
        ? `VIX P${ivp.toFixed(0)}` : '';
    setText('greekSummary', [vixLabel, callDelta === null && putDelta === null
        ? (greeks.msg || 'Option Greeks unavailable') : 'Estimated move per 100 index points'].filter(Boolean).join(' · '));

}

function setupEventListeners() {

    // ═══ 3-WAY MODE CONTROL ═══
    document.querySelectorAll('.mode-opt').forEach(btn => {
        btn.addEventListener('click', async () => {
            const mode = btn.dataset.mode;
            const previousMode = String(state.latestData?.mode || document.querySelector('.mode-opt.active')?.dataset.mode || 'HISTORICAL').toUpperCase();
            let confirmRealMode = false;
            let realModeVerification = '';
            if (mode === 'REAL' && state.latestData?.mode !== 'REAL') {
                const confirmed = confirm('REAL mode can place live orders. Continue?');
                if (!confirmed) return;
                confirmRealMode = true;
                realModeVerification = 'YES';
            }
            document.querySelectorAll('.mode-opt').forEach(modeBtn => {
                modeBtn.disabled = true;
            });
            const response = await window.sendCommand({
                cmd: 'set_mode',
                mode,
                confirm_real_mode: confirmRealMode,
                real_mode_verification: realModeVerification,
                reset: mode === 'HISTORICAL'
            });
            const payload = response?.data || {};
            const actualMode = String(payload.mode || previousMode).toUpperCase();
            syncModeControls(actualMode);
            if (payload.status === 'ok') {
                showToast(`Mode changed to ${actualMode}`, actualMode === 'REAL' ? 'warning' : 'success');
            } else if (payload.status === 'blocked') {
                const reason = payload.reason ? `: ${payload.reason}` : '';
                showToast(`Mode switch blocked${reason}`, 'warning');
            } else {
                showToast(`Mode switch not confirmed: ${payload.reason || 'no backend response'}`, 'error');
            }
            refreshDashboardSnapshot('mode-click', { force: true });
            return;
        });
    });


    // ═══ UPSTOX AUTH MODAL LISTENERS ═══
    const closeUpstoxAuth = document.getElementById('closeUpstoxAuth');
    if (closeUpstoxAuth) {
        closeUpstoxAuth.addEventListener('click', () => {
            const modal = document.getElementById('upstoxAuthModal');
            if (modal) modal.style.display = 'none';
        });
    }

    const upstoxModal = document.getElementById('upstoxAuthModal');
    if (upstoxModal) {
        upstoxModal.addEventListener('click', (e) => {
            if (e.target === upstoxModal) upstoxModal.style.display = 'none';
        });
    }

    const submitUpstoxAuth = document.getElementById('submitUpstoxAuth');
    if (submitUpstoxAuth) {
        submitUpstoxAuth.addEventListener('click', () => {
            const input = document.getElementById('upstoxAuthInput');
            if (input && input.value) {
                submitUpstoxCode(input.value);
                const modal = document.getElementById('upstoxAuthModal');
                if (modal) modal.style.display = 'none';
                input.value = '';
            } else {
                showToast('Please paste URL, Auth Code, or Token', 'warning');
            }
        });
    }

    const retryUpstoxLogin = document.getElementById('retryUpstoxLogin');
    if (retryUpstoxLogin) {
        retryUpstoxLogin.addEventListener('click', async () => {
            const statusMsg = document.getElementById('upstoxAuthStatusMsg');
            if (statusMsg) statusMsg.innerHTML = '<span style="color:var(--text-secondary)">Checking...</span>';
            try {
                const res = await authFetch('/api/upstox_status', { cache: 'no-store' });
                const st = await res.json();
                if (statusMsg) {
                    if (st.connected || st.authenticated) {
                        statusMsg.innerHTML = `<span style="color:var(--green)">✅ Session ACTIVE (${st.user_id || '7JATYL'}). Expires: ${st.expires_at || '24h'} (${st.hours_remaining || 0}h left)</span>`;
                        showToast('Upstox session is live and connected!', 'success');
                        if (window.sendCommand) window.sendCommand({ cmd: 'get_state' });
                        refreshDashboardSnapshot('upstox_refresh');
                    } else {
                        statusMsg.innerHTML = `<span style="color:var(--red)">⚠️ Status: ${st.status || 'OFFLINE'} — ${st.reason || 'Auth needed'}</span>`;
                        showToast('Upstox: ' + (st.reason || 'Authentication required'), 'warning');
                    }
                }
            } catch (e) {
                if (statusMsg) statusMsg.innerHTML = `<span style="color:var(--red)">Probe error: ${e.message}</span>`;
            }
        });
    }

    const upstoxBadge = document.getElementById('gw_upstox');
    if (upstoxBadge) {
        upstoxBadge.style.cursor = 'pointer';
        upstoxBadge.onclick = startUpstoxAuthFlow;
        upstoxBadge.title = 'Upstox market data. Click to log in or refresh.';
    }

    window.addEventListener('message', (event) => {
        if (event && event.data === 'upstox_auth_complete') {
            showToast('Upstox login completed. Updating session...', 'success');
            const modal = document.getElementById('upstoxAuthModal');
            if (modal) modal.style.display = 'none';
            if (window.sendCommand) window.sendCommand({ cmd: 'get_state' });
            refreshDashboardSnapshot('upstox_auth_complete');
        }
    });

    // Hidden legacy stubs
    const closeFyersAuth = document.getElementById('closeFyersAuth');
    const submitFyersAuth = document.getElementById('submitFyersAuth');
    const retryFyersLogin = document.getElementById('retryFyersLogin');
    const fyersBadge = document.getElementById('gw_fyers');
    const kiteBadge = document.getElementById('gw_kite');

    const autoSwitch = document.getElementById('autoModeSwitch');
    if (autoSwitch) {
        autoSwitch.addEventListener('change', () => {
            const isChecked = autoSwitch.checked;
            const previouslyConfirmed = Boolean(state.latestData?.config?.auto_mode);
            let realModeVerification = '';
            let confirmRealMode = false;
            if (isChecked && state.latestData?.mode === 'REAL') {
                const confirmed = confirm('REAL AUTO MODE will allow live broker orders. Enable auto-execution?');
                if (!confirmed) {
                    autoSwitch.checked = false;
                    showToast('REAL Auto Mode not enabled.', 'warning');
                    return;
                }
                realModeVerification = 'YES';
                confirmRealMode = true;
            }
            // Do not claim AUTO is on before the scanner acknowledges all readiness
            // checks.  The previous optimistic update showed ENABLED even when the
            // backend refused the request for missing chains/warmup.
            autoSwitch.disabled = true;
            window.sendCommand({
                cmd: 'configure', 
                auto_mode: isChecked,
                confirm_real_mode: confirmRealMode,
                real_mode_verification: realModeVerification
            }).then((msg) => {
                const accepted = msg?.type === 'config_ack' && msg?.data?.status === 'ok';
                if (accepted) {
                    if (state.latestData && state.latestData.config) {
                        state.latestData.config.auto_mode = isChecked;
                    }
                } else {
                    autoSwitch.checked = previouslyConfirmed;
                }
                autoSwitch.disabled = false;
            }).catch(() => {
                autoSwitch.checked = previouslyConfirmed;
                autoSwitch.disabled = false;
            });
        });
    }

    // ═══ LOG FILTERS ═══
    const tfPolicy = document.getElementById('topTimeframePolicy');
    if (tfPolicy) {
        tfPolicy.addEventListener('change', () => {
            const policy = tfPolicy.value || 'INCLUDE_5MIN';
            window.sendCommand({
                cmd: 'configure',
                timeframe_entry_policy: policy
            });
            if (state.latestData && state.latestData.config) {
                state.latestData.config.ut_timeframe_entry_policy = policy;
            }
            const labels = {
                PRIMARY_1: '1M',
                PRIMARY_5: '5M',
                PRIMARY_15: '15M',
                INCLUDE_5MIN: '5M+15M',
                INCLUDE_1MIN_5MIN: '1M+5M',
                INCLUDE_1MIN_15MIN: '1M+15M',
                INCLUDE_1MIN_5MIN_15MIN: '1M+5M+15M'
            };
            showToast(`TF Policy: ${labels[policy] || policy}`, 'info');
        });
    }

    document.querySelectorAll('.log-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.logFilter = btn.dataset.type;
            state.lastRenderedLogData = '';
            state.lastRenderedEngineLabData = '';
            renderLogs();
        });
    });

    // ═══ KILL ALL BUTTON ═══
    const killAllBtn = document.getElementById('killAllBtn');
    if (killAllBtn) {
        killAllBtn.addEventListener('click', () => {
            if (confirm("⚠️ Are you sure you want to kill ALL open trades?")) {
                window.sendCommand({ cmd: 'kill_all' });
                showToast("Kill All signal sent", "warning");
            }
        });
    }
}


function updateMarketTime() {
    const now = new Date();
    const options = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    const timeStr = now.toLocaleTimeString('en-IN', options);
    const dateStr = now.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' });
    setText('marketTime', timeStr);
    setText('marketDate', dateStr);

    if (state.latestData) {
        const heartbeat = state.latestData.dashboard_heartbeat || {};
        const heartbeatTime = heartbeat.timestamp ? new Date(heartbeat.timestamp).getTime() : NaN;
        const heartbeatAgeMs = Number.isFinite(heartbeatTime) ? Math.max(0, Date.now() - heartbeatTime) : null;
        const scanAgeMs = Number.isFinite(Number(heartbeat.scan_age_ms)) ? Number(heartbeat.scan_age_ms) : null;
        const wsAgeMs = state.lastWsUpdateAt ? Math.max(0, Date.now() - state.lastWsUpdateAt) : null;
        const liveAgeMs = heartbeatAgeMs !== null ? heartbeatAgeMs : wsAgeMs;
        const liveSuffix = heartbeatAgeMs !== null
            ? (scanAgeMs !== null ? ` / scan ${(scanAgeMs / 1000).toFixed(1)}s` : '')
            : ' / ws';
        const liveText = liveAgeMs === null ? '--' : `${(liveAgeMs / 1000).toFixed(1)}s${liveSuffix}`;
        setText('liveRefreshAge', liveText);
    }
}

function setText(id, val) { 
    const el = document.getElementById(id); 
    if (el) el.textContent = val; 
}

function formatSharpeDailyYearly(source = {}) {
    if (source && source.sharpe_ratio_display) return source.sharpe_ratio_display;
    const dailyRaw = source && source.sharpe_ratio_daily !== undefined
        ? source.sharpe_ratio_daily
        : source && source.sharpe_ratio;
    const daily = Number(dailyRaw || 0);
    const dailySafe = Number.isFinite(daily) ? daily : 0;
    const yearlyRaw = source && source.sharpe_ratio_yearly !== undefined
        ? source.sharpe_ratio_yearly
        : dailySafe * Math.sqrt(252);
    const yearly = Number(yearlyRaw || 0);
    const yearlySafe = Number.isFinite(yearly) ? yearly : 0;
    return `${dailySafe.toFixed(2)} / ${yearlySafe.toFixed(2)}`;
}

function formatPrice(p) { return (!p || p === 0 || isNaN(p)) ? '--' : p.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
function formatMoney(value, compact = false) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return '₹0';
    const sign = num < 0 ? '-' : '';
    const abs = Math.abs(num);
    if (compact && abs >= 100000) return `${sign}₹${(abs / 100000).toFixed(1)}L`;
    if (compact && abs >= 1000) return `${sign}₹${(abs / 1000).toFixed(1)}K`;
    return `${sign}₹${abs.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}
function formatPercent(value) {
    const num = Number(value || 0);
    return `${Number.isFinite(num) ? num.toFixed(1) : '0.0'}%`;
}
function baseInstrumentLabel(value = '') {
    const text = String(value || '').toUpperCase();
    if (text.startsWith('BANKNIFTY')) return 'BANKNIFTY';
    if (text.startsWith('SENSEX')) return 'SENSEX';
    if (text.startsWith('MIDCPNIFTY') || text.startsWith('MIDCAPNIFTY')) return 'MIDCPNIFTY';
    if (text.startsWith('NIFTY')) return 'NIFTY';
    return String(value || '--').split(' ')[0] || '--';
}


function updatePnlPanel(trades = {}) {
    const monetary = trades.monetary || {};
    const headline = monetary.headline || trades.summary || {};
    const capital = monetary.capital || {};
    const period = monetary.period_pnl || {};
    const summary = trades.summary || {};
    const basis = monetary.accounting_basis || {};
    const basisLabel = document.getElementById('pnlBasisLabel');
    if (basisLabel) {
        const lane = String(basis.lane || summary.lane || 'UNKNOWN').toUpperCase();
        const description = String(basis.description || 'Accounting basis unavailable');
        basisLabel.textContent = `${lane}: ${description}`;
        basisLabel.className = 'pnl-basis-label ' + (basis.is_live_money === true ? 'live' : 'nonmoney');
    }

    setText('pnlWinRate', formatPercent(headline.win_rate || 0));
    setText('pnlWinMeta', `${summary.wins || 0}W / ${summary.losses || 0}L`);
    const totalTrades = (summary.wins || 0) + (summary.losses || 0);
    setText('pnlProfitFactor', totalTrades > 0 ? Number(headline.profit_factor || 0).toFixed(2) : "N/A");
    setText('pnlSharpe', `Sharpe ${formatSharpeDailyYearly(headline)}`);
    setText('pnlDrawdown', `DD ${formatMoney(headline.max_drawdown || 0, true)} (${formatPercent(headline.max_drawdown_pct || 0)})`);
    setText('pnlCurrentDd', `Current ${formatMoney(headline.current_drawdown || 0, true)} / ${formatPercent(headline.current_drawdown_pct || 0)}`);
    setText('pnlCapital', formatMoney(capital.configured_demat_capital || 0, true));
    setText('pnlAllocation', `F ${formatMoney(capital.futures_allocation || 0, true)} / O ${formatMoney(capital.options_allocation || 0, true)} / Live ${formatMoney(capital.active_trade_allocation || 0, true)}`);
    setText('pnlDaily', formatMoney(period.daily || summary.daily_pnl || 0, true));
    setText('pnlWeekly', formatMoney(period.weekly || 0, true));
    setText('pnlMonthly', formatMoney(period.monthly || 0, true));
    setText('pnlAvgHold', `${Number(headline.avg_hold_minutes || 0).toFixed(1)}m`);

    const breakdowns = monetary.breakdowns || {};
    const makeRows = (title, obj) => {
        const rows = Object.entries(obj || {}).slice(0, 4);
        if (!rows.length) return `<div class="pnl-list-row"><span>${title}</span><small>--</small></div>`;
        return rows.map(([name, value]) => `
            <div class="pnl-list-row">
                <span>${title}: ${name}</span>
                <small>${formatMoney(value.pnl || 0, true)} / ${formatPercent(value.win_rate || 0)}</small>
            </div>
        `).join('');
    };
    const breakdownEl = document.getElementById('pnlBreakdowns');
    if (breakdownEl) {
        breakdownEl.innerHTML = [
            makeRows('IDX', breakdowns.index),
            makeRows('TYPE', breakdowns.instrument_type),
            makeRows('TF', breakdowns.timeframe),
        ].join('');
    }

    const tradeLine = (label, row) => row && row.instrument
        ? `<div class="pnl-list-row"><span>${label}: ${row.instrument} ${row.timeframe || ''}</span><small>${formatMoney(row.pnl || 0, true)}</small></div>`
        : `<div class="pnl-list-row"><span>${label}</span><small>--</small></div>`;
    const extremesEl = document.getElementById('pnlExtremes');
    if (extremesEl) {
        extremesEl.innerHTML = [
            tradeLine('Largest Winner', monetary.largest_winner),
            tradeLine('Largest Loser', monetary.largest_loser),
        ].join('');
    }

    const liveEl = document.getElementById('pnlLivePositions');
    if (liveEl) {
        const live = monetary.live_positions || [];
        liveEl.innerHTML = live.length ? live.slice(0, 4).map(row => `
            <div class="pnl-live-row">
                <b>${row.instrument} ${row.direction || ''} ${row.timeframe || ''}</b>
                <small>P&L ${formatMoney(row.pnl || 0, true)} / Alloc ${formatMoney(row.allocation || 0, true)} / SL ${row.stop || '--'} / TGT ${row.target || '--'}</small>
            </div>
        `).join('') : '<div class="pnl-live-row"><small>No current live/open position.</small></div>';
    }
}

function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    
    let icon = '🔔';
    const typeLower = type.toLowerCase();
    if (typeLower.includes('buy') || typeLower.includes('long')) icon = '📈';
    else if (typeLower.includes('sell') || typeLower.includes('short')) icon = '📉';
    else if (typeLower.includes('error') || typeLower.includes('alert')) icon = '⚠️';
    
    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">${icon}</div>
            <div class="toast-body">${message}</div>
        </div>
        <div class="toast-progress"><div class="toast-bar"></div></div>
    `;
    
    container.appendChild(toast);
    
    const dismiss = () => {
        toast.style.animation = 'fadeOutToast 0.3s forwards';
        setTimeout(() => toast.remove(), 300);
    };
    
    toast.addEventListener('click', dismiss);
    setTimeout(() => { if(toast.parentElement) dismiss(); }, 5000);
}

async function completeUpstoxRedirectIfPresent() {
    const params = new URLSearchParams(window.location.search);
    const authCode = params.get('code') || params.get('auth_code');
    if (!authCode) return;

    try {
        showToast('Completing Upstox login...', 'info');
        await submitUpstoxCode(authCode);
    } finally {
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

async function startUpstoxAuthFlow() {
    // 1. Immediately open modal so badge click is responsive instantly
    const modal = document.getElementById('upstoxAuthModal');
    if (modal) {
        modal.style.display = 'flex';
        const input = document.getElementById('upstoxAuthInput');
        if (input) input.focus();
    }
    const statusMsg = document.getElementById('upstoxAuthStatusMsg');
    if (statusMsg) {
        statusMsg.innerHTML = '<span style="color:var(--text-secondary)">Checking Upstox session...</span>';
    }

    try {
        const [statusRes, urlRes] = await Promise.allSettled([
            authFetch('/api/upstox_status', { cache: 'no-store' }),
            authFetch('/api/upstox_auth_url', { cache: 'no-store' })
        ]);

        let authUrl = '/api/upstox_login';
        if (urlRes.status === 'fulfilled') {
            try {
                const uData = await urlRes.value.json();
                if (uData.status === 'ok' && uData.auth_url) {
                    authUrl = uData.auth_url;
                }
            } catch (e) {}
        }
        const loginLink = document.getElementById('openUpstoxLogin');
        if (loginLink) {
            loginLink.href = authUrl;
        }

        if (statusRes.status === 'fulfilled') {
            try {
                const sData = await statusRes.value.json();
                if (statusMsg) {
                    if (sData.authenticated || sData.connected) {
                        statusMsg.innerHTML = `<span style="color:var(--green)">✅ Session ACTIVE (${sData.user_id || '7JATYL'}). Expires: ${sData.expires_at || '24h'} (${sData.hours_remaining || 0}h left)</span>`;
                    } else {
                        statusMsg.innerHTML = `<span style="color:var(--red)">⚠️ Status: ${sData.status || 'OFFLINE'} — ${sData.reason || 'Login required'}. Click "Open Login" or paste token below.</span>`;
                    }
                }
            } catch (e) {}
        }
    } catch (err) {
        if (statusMsg) {
            statusMsg.innerHTML = `<span style="color:var(--red)">⚠️ Error checking status: ${err.message}</span>`;
        }
    }
}
window.startUpstoxAuthFlow = startUpstoxAuthFlow;

async function submitUpstoxCode(inputVal) {
    const statusMsg = document.getElementById('upstoxAuthStatusMsg');
    try {
        let code = String(inputVal || '').trim();
        if (code.includes('code=')) {
            try {
                const urlObj = new URL(code.startsWith('http') ? code : 'http://dummy/' + code);
                code = urlObj.searchParams.get('code') || code;
            } catch (e) {}
        }
        const isJwt = code.startsWith('eyJ');
        if (statusMsg) statusMsg.innerHTML = '<span style="color:var(--text-secondary)">Validating with Upstox...</span>';
        const submit = await authFetch('/api/upstox_auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(isJwt ? { token: code } : { code: code })
        });
        const result = await submit.json();
        if (result.status === 'ok') {
            showToast('Upstox token updated. Primary broker connected.', 'success');
            const modal = document.getElementById('upstoxAuthModal');
            if (modal) modal.style.display = 'none';
            if (window.sendCommand) window.sendCommand({ cmd: 'get_state' });
            refreshDashboardSnapshot('upstox_auth');
        } else {
            const errText = result.message || 'Upstox auth update failed';
            showToast(errText, 'error');
            if (statusMsg) {
                statusMsg.innerHTML = `<span style="color:var(--red)">❌ ${errText}</span>`;
            }
        }
    } catch (err) {
        showToast('Submission failed: ' + err.message, 'error');
        if (statusMsg) {
            statusMsg.innerHTML = `<span style="color:var(--red)">❌ Network error: ${err.message}</span>`;
        }
    }
}
window.submitUpstoxCode = submitUpstoxCode;

// Backward compatibility aliases
async function startKiteAuthFlow() { return startUpstoxAuthFlow(); }
async function startFyersAuthFlow() { return startUpstoxAuthFlow(); }
async function submitFyersCode(code) { return submitUpstoxCode(code); }
async function completeFyersRedirectIfPresent() { return completeUpstoxRedirectIfPresent(); }

function updateGatewayStatus(status, mode) {
    status = status || {};
    const upstox = status.upstox || {};
    const upstoxOnline = status.upstox_healthy === true || status.upstox_session_live === true || upstox.healthy === true;
    const gateways = {
        'gw_upstox': upstoxOnline,
        'gw_kite': false,
        'gw_fyers': false,
    };

    for (const [id, isOnline] of Object.entries(gateways)) {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('standby');
            if (isOnline) {
                el.classList.add('online');
                el.classList.remove('offline');
            } else {
                el.classList.add('offline');
                el.classList.remove('online');
            }
        }
    }

    const upstoxEl = document.getElementById('gw_upstox');
    if (upstoxEl) {
        upstoxEl.style.cursor = 'pointer';
        upstoxEl.onclick = startUpstoxAuthFlow;
        upstoxEl.title = upstoxOnline
            ? 'Upstox market data connected. Click to view or refresh.'
            : 'Upstox market data disconnected. Click to log in or refresh.';
    }

    const angelEl = document.getElementById('gw_angelone');
    if (angelEl) {
        const broker = status.broker_status || {};
        const modeName = String(mode || '').trim().toUpperCase();
        let appearance = '';
        let label = 'CHECKING';
        let detail = 'Waiting for trading mode';
        if (['HISTORICAL', 'REPLAY', 'BACKTEST', 'PAPER', 'LIVE_PAPER', 'MANUAL'].includes(modeName)) {
            appearance = 'standby';
            label = 'STANDBY';
            detail = 'Angel One connects for order execution in REAL mode';
        } else if (['REAL', 'LIVE_REAL', 'LIVE'].includes(modeName)) {
            appearance = 'offline';
            if (broker.provider !== 'angelone') {
                label = 'UNAVAILABLE';
                detail = 'Angel One execution status unavailable';
            } else if (broker.configured === false) {
                label = 'NOT CONFIGURED';
                detail = 'Angel One execution credentials are incomplete';
            } else if (broker.connected !== true) {
                label = 'DISCONNECTED';
                detail = 'Angel One execution session unavailable in REAL mode';
            } else if (broker.lease_held !== true) {
                label = 'BLOCKED';
                detail = 'This RP94 instance does not hold the Angel One order execution lease';
            } else {
                appearance = 'online';
                label = 'CONNECTED';
                detail = 'Angel One execution session connected';
            }
        }
        angelEl.classList.remove('online', 'offline', 'standby');
        if (appearance) angelEl.classList.add(appearance);
        angelEl.title = detail;
        const labelEl = document.getElementById('gw_angelone_state');
        if (labelEl && labelEl.textContent !== label) labelEl.textContent = label;
    }
}

function updateDiagnosticsPanel(diagnostics) {
    diagnostics = diagnostics || {};
    const rejects = diagnostics.rejects || {};
    const stabilization = diagnostics.stabilization || {};
    const timeouts = diagnostics.timeouts || {};
    const sources = diagnostics.sources || {};
    const latency = diagnostics.latency || {};
    const repaintGuard = diagnostics.repaint_guard || {};
    const exitReasons = diagnostics.exit_reasons || {};
    const dataFreshness = diagnostics.data_freshness || {};
    const sessionStats = diagnostics.session_stats || {};
    const activePeak = diagnostics.active_peak || {};
    const simulation = diagnostics.simulation || {};
    const instrumentSelection = diagnostics.instrument_selection || {};
    updateHardwarePanel(diagnostics.system_metrics || {});
    const isHistorical = (diagnostics.mode || state.latestData?.mode) === 'HISTORICAL';

    const isWaitReason = reason => /signal waiting|partial candle deferred|deferred until candle close/i.test(String(reason || ''));
    const legacyWaitingEntries = Object.entries(rejects).filter(([reason]) => isWaitReason(reason));
    const rejectEntries = Object.entries(rejects)
        .map(([reason, count]) => [String(reason || 'Unknown'), Number(count || 0)])
        .filter(([reason, count]) => Number.isFinite(count) && count > 0 && !isWaitReason(reason))
        .sort((a, b) => b[1] - a[1]);
    const topReject = rejectEntries[0];
    const rejectSummary = diagnostics.reject_summary || {};
    const hasSummaryCount = value => (typeof value === 'number' || (typeof value === 'string' && value.trim() !== ''))
        && Number.isFinite(Number(value)) && Number(value) >= 0;
    const hasFullRejectTotal = hasSummaryCount(rejectSummary.total);
    const totalRejects = hasFullRejectTotal ? Math.max(0, Number(rejectSummary.total))
        : rejectEntries.reduce((sum, [, count]) => sum + count, 0);
    const rejectReasonCount = hasSummaryCount(rejectSummary.reason_count)
        ? Math.max(0, Number(rejectSummary.reason_count)) : rejectEntries.length;
    const waitingSummary = diagnostics.waiting_summary || {};
    const hasFullWaitingTotal = hasSummaryCount(waitingSummary.total);
    const waitingEntries = [...Object.entries(diagnostics.waiting || {}), ...legacyWaitingEntries];
    const totalWaiting = hasFullWaitingTotal ? Math.max(0, Number(waitingSummary.total))
        : waitingEntries.reduce((sum, [, count]) => sum + (Number.isFinite(Number(count)) ? Math.max(0, Number(count)) : 0), 0);

    // Sources: A:ON F:ON Y:ON
    const yahooState = sources.yahoo_mode === 'emergency_used' ? 'EMERG' : 'STBY';
    const sourceText = [
        `K:${sources.kite ? 'ON' : 'OFF'}`,
        `F:${sources.fyers ? 'ON' : 'OFF'}`,
        `Y:${sources.yahoo ? yahooState : 'OFF'}`,
    ].join(' ');
    setText('diagSources', sourceText || '--');

    // Data Age: live mode shows broker tick age; historical mode uses cached candles.
    const ages = Object.entries(dataFreshness);
    if (isHistorical) {
        setText('diagDataAge', 'Cached history');
    } else if (ages.length > 0) {
        const ageStrs = ages.map(([inst, age]) => {
            const label = inst.replace('MIDCPNIFTY', 'MC').replace('BANKNIFTY', 'BN').replace('SENSEX', 'SX').replace('NIFTY', 'NF');
            if (age < 0) return `${label}:--`;
            if (age < 60) return `${label}:${Math.round(age)}s`;
            return `${label}:${Math.round(age / 60)}m`;
        });
        setText('diagDataAge', ageStrs.join(' '));
    } else {
        setText('diagDataAge', '--');
    }
    // Color the data age card if any instrument is stale (>60s)
    const dataAgeEl = document.getElementById('diagDataAge');
    if (dataAgeEl) {
        const maxAge = isHistorical ? 0 : Math.max(...Object.values(dataFreshness).filter(v => v >= 0), 0);
        dataAgeEl.style.color = maxAge > 60 ? 'var(--red)' : maxAge > 30 ? 'var(--orange)' : '';
    }

    // Scan Speed: Sync exactly with top-right panel
    const topLatency = document.getElementById('scanLatency') ? document.getElementById('scanLatency').innerText : '--ms';
    const topCount = document.getElementById('scanCount') ? document.getElementById('scanCount').innerText : '0';
    setText('diagScanSpeed', `#${topCount} / ${topLatency}`);

    // Session Stats: entries/exits
    const entries = sessionStats.entries || 0;
    const exits = sessionStats.exits || 0;
    setText('diagSessionLabel', isHistorical ? 'Pre-Gate Select' : 'Session');
    setText(
        'diagSessionStats',
        isHistorical
            ? `${instrumentSelection.OPT || 0} OPT / ${instrumentSelection.FUT || 0} FUT`
            : `${entries} sig / ${exits} exit`
    );

    // Repaint Guard: checked/aborted/passed
    const rgChecked = repaintGuard.checked || 0;
    const rgAborted = repaintGuard.aborted || 0;
    const rgPassed = repaintGuard.passed || 0;
    const rgEl = document.getElementById('diagRepaintGuard');
    if (rgEl) {
        rgEl.textContent = rgChecked === 0 ? 'No checks' : `${rgAborted} abort / ${rgPassed} pass`;
        rgEl.style.color = rgAborted > 0 ? 'var(--orange)' : '';
    }

    // Profit Guard: count of profit-saving vs threat/raw exits (incl. post-restart reasons)
    const profitSaves = (exitReasons.SMART_PROFIT_LOCK || 0) +
                        (exitReasons.MAJOR_WIN_GUARD || 0) +
                        (exitReasons.LOW_GAIN_PROTECT || 0);
    const peakThreat = (exitReasons.PEAK_THREAT_EXIT || 0) + (exitReasons.LOSS_THREAT_EXIT || 0);
    const lossBands = (exitReasons.RUPEE_LOSS_BAND_EXIT || 0) + (exitReasons.EARLY_LOSS_CUT || 0);
    const rawStops = (exitReasons.STOP_HIT || 0) + (exitReasons.TRAILING_STOP || 0);
    const targets = exitReasons.TARGET_HIT || 0;
    const repaintAborts = exitReasons.REPAINT_ABORT || 0;
    const otherExits = Object.entries(exitReasons).reduce((sum, [reason, count]) => {
        const key = String(reason || '').toUpperCase();
        if ([
            'SMART_PROFIT_LOCK', 'MAJOR_WIN_GUARD', 'LOW_GAIN_PROTECT',
            'PEAK_THREAT_EXIT', 'LOSS_THREAT_EXIT', 'RUPEE_LOSS_BAND_EXIT',
            'EARLY_LOSS_CUT', 'STOP_HIT', 'TRAILING_STOP',
            'TARGET_HIT', 'REPAINT_ABORT',
        ].includes(key)) return sum;
        return sum + Number(count || 0);
    }, 0);
    const pgEl = document.getElementById('diagProfitGuard');
    if (pgEl) {
        const parts = [];
        if (profitSaves > 0) parts.push(`${profitSaves} lock`);
        if (peakThreat > 0) parts.push(`${peakThreat} threat`);
        if (lossBands > 0) parts.push(`${lossBands} band`);
        if (targets > 0) parts.push(`${targets} tgt`);
        if (rawStops > 0) parts.push(`${rawStops} sl`);
        if (repaintAborts > 0) parts.push(`${repaintAborts} rp`);
        if (otherExits > 0) parts.push(`${otherExits} other`);
        pgEl.textContent = parts.length > 0 ? parts.join(' / ') : 'No exits';
        pgEl.style.color = profitSaves > (rawStops + peakThreat + lossBands) ? 'var(--green)'
            : (rawStops + peakThreat + lossBands) > 0 ? 'var(--orange)' : '';
    }

    // Active Peak: show per-instrument peak vs current PnL
    const peakEntries = Object.entries(activePeak);
    const apEl = document.getElementById('diagActivePeak');
    if (apEl) {
        if (peakEntries.length === 0) {
            apEl.textContent = 'No active';
            apEl.style.color = '';
        } else {
            const strs = peakEntries.map(([inst, data]) => {
                const label = inst.replace('MIDCPNIFTY', 'MC').replace('BANKNIFTY', 'BN').replace('SENSEX', 'SX').replace('NIFTY', 'NF');
                return `${label}:${data.current}/${data.peak}`;
            });
            apEl.textContent = strs.join(' ');
            // Red if current is way below peak
            const worstRatio = Math.min(...peakEntries.map(([, d]) => d.peak > 0 ? d.current / d.peak : 1));
            apEl.style.color = worstRatio < 0.5 ? 'var(--red)' : worstRatio < 0.75 ? 'var(--orange)' : 'var(--green)';
        }
    }

    // Top Reject
    const windowLabel = isHistorical && simulation.backtest_days ? `, ${simulation.backtest_days}d` : '';
    const classifyReject = reason => {
        const text = String(reason || '').toLowerCase();
        if (text.includes('regime')) return 'Regime';
        if (text.includes('price-action') || text.includes('support') || text.includes('resistance') || text.includes('aroon')) return 'Price action';
        if (text.includes('confidence') || text.includes('grade')) return 'Quality';
        if (text.includes('source') || text.includes('data') || text.includes('quote')) return 'Data';
        if (text.includes('cooldown') || text.includes('time') || text.includes('session')) return 'Timing';
        if (text.includes('exposure') || text.includes('concurrency') || text.includes('risk')) return 'Risk';
        return 'Other';
    };
    const categoryTotals = rejectEntries.reduce((acc, [reason, count]) => {
        const category = classifyReject(reason);
        acc[category] = (acc[category] || 0) + count;
        return acc;
    }, {});
    const categoryRows = Object.entries(categoryTotals).sort((a, b) => b[1] - a[1]);
    const topShare = topReject && totalRejects > 0 ? (topReject[1] / totalRejects) * 100 : 0;
    const categoryText = categoryRows.slice(0, 3)
        .map(([category, count]) => `${category} ${Math.round((count / totalRejects) * 100)}%`)
        .join(' / ');

    setText('diagTopReject', topReject ? topReject[0] : '--');
    setText('diagRejectTotalLabel', hasFullRejectTotal ? 'Total Blocks' : 'Shown Blocks');
    setText('diagRejectTotal', totalRejects.toLocaleString('en-IN'));
    setText('diagRejectDiversity', `${rejectReasonCount} reasons${windowLabel}`);
    setText('diagWaitingTotal', `${totalWaiting.toLocaleString('en-IN')}${hasFullWaitingTotal ? '' : ' shown'} waits; not rejected`);
    setText(
        'diagRejectContext',
        topReject ? `${topReject[1].toLocaleString('en-IN')} blocks / ${topShare.toFixed(0)}% of rejects${windowLabel}` : 'No rejects in current window'
    );
    const insightEl = document.getElementById('diagRejectInsights');
    if (insightEl) {
        const concentration = topShare >= 60 ? 'Concentrated' : topShare >= 35 ? 'Clustered' : 'Distributed';
        const second = rejectEntries[1];
        const secondText = second ? `${second[0]} (${second[1]})` : 'None';
        insightEl.innerHTML = rejectEntries.length ? [
            `<span><b>${escapeHtml(concentration)}</b>Top reason is ${topShare.toFixed(0)}% of all blocks</span>`,
            `<span><b>${escapeHtml(categoryRows[0]?.[0] || 'Category')}</b>${escapeHtml(categoryText || 'No category mix')}</span>`,
            `<span><b>Next reason</b>${escapeHtml(secondText)}</span>`,
        ].join('') : 'No rejected-signal counters yet.';
    }

    setText('diagMode', diagnostics.mode || state.latestData?.mode || '--');

    // Reject List
    const list = document.getElementById('diagRejectList');
    if (!list) return;
    const rows = rejectEntries
        .slice(0, 8)
        .map(([reason, count]) => {
            const pct = totalRejects > 0 ? (count / totalRejects) * 100 : 0;
            return `<div><span>${escapeHtml(reason)}<em>${escapeHtml(classifyReject(reason))} / ${pct.toFixed(0)}% of rejects</em></span><b>${count.toLocaleString('en-IN')}</b></div>`;
        });
    list.innerHTML = rows.length ? rows.join('') : 'No rejected-signal counters yet.';
}

function updateAnalyticsStrip(trades) {
    const strip = document.getElementById('analyticsStrip');
    if (!strip || !trades || !trades.analytics) return;
    const a = trades.analytics;
    const bestTf = a.by_timeframe ? Object.entries(a.by_timeframe).sort((x,y) => y[1].pnl - x[1].pnl)[0] : null;
    const bestGrade = a.by_grade ? Object.entries(a.by_grade).sort((x,y) => y[1].pnl - x[1].pnl)[0] : null;
    const bestIdx = a.by_instrument ? Object.entries(a.by_instrument).sort((x,y) => y[1].pnl - x[1].pnl)[0] : null;
    const bestType = a.by_type ? Object.entries(a.by_type).sort((x,y) => y[1].pnl - x[1].pnl)[0] : null;
    const topExit = a.by_exit_reason ? Object.entries(a.by_exit_reason).sort((x,y) => y[1].count - x[1].count)[0] : null;
    
    const spans = strip.querySelectorAll('span');
    if (spans[0] && bestTf) spans[0].textContent = `TF: ${bestTf[0]} (₹${bestTf[1].pnl.toLocaleString()})`;
    if (spans[1] && bestGrade) spans[1].textContent = `Grade: ${bestGrade[0]} (₹${bestGrade[1].pnl.toLocaleString()})`;
    if (spans[2] && bestIdx) spans[2].textContent = `Index: ${bestIdx[0]} (₹${bestIdx[1].pnl.toLocaleString()})`;
    if (spans[3] && bestType) spans[3].textContent = `Type: ${bestType[0]} (₹${bestType[1].pnl.toLocaleString()})`;
    if (spans[4] && topExit) spans[4].textContent = `Exit: ${topExit[0].replace(/_/g,' ')} (${topExit[1].count})`;

    // Price-action advisory: surface the ADVERSE bucket when the PA module is on.
    // pnl of ADVERSE = what blocking those trades would change the total PnL by.
    const adv = a.by_price_action && a.by_price_action.ADVERSE;
    let paSpan = document.getElementById('paAdvisorySpan');
    if (adv && adv.count) {
        if (!paSpan) { paSpan = document.createElement('span'); paSpan.id = 'paAdvisorySpan'; strip.appendChild(paSpan); }
        paSpan.textContent = `PA-Adverse: ${adv.count} (₹${Number(adv.pnl).toLocaleString()}, WR ${adv.win_rate}%)`;
        paSpan.title = 'Trades the price-action module flags ADVERSE. Blocking them would change total PnL by ' + (-Number(adv.pnl)).toLocaleString();
    } else if (paSpan) {
        paSpan.remove();
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('analysingSignal');
    if (btn) {
        btn.addEventListener('click', () => {
            btn.classList.remove('state-idle', 'state-offline', 'state-analysing', 'state-live');
            btn.classList.add('state-analysing');
            const txt = document.getElementById('analysisText');
            if (txt) txt.textContent = 'FORCING...';
            // Trigger animation restart
            const dot = document.getElementById('analysisDot');
            if (dot) {
                dot.style.animation = 'none';
                dot.offsetHeight; /* trigger reflow */
                dot.style.animation = null; 
            }
            authFetch('/api/recalibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            }).catch(console.error);
        });
    }
});
