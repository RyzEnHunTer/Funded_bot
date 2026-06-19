/**
 * Prop Firm ML Trading System — Dashboard App
 * 
 * Loads results.json and renders interactive charts and metrics.
 */

(async function () {
    "use strict";

    // ─── Load Data ──────────────────────────────────────────────────────
    let data;
    try {
        const resp = await fetch("results.json");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        data = await resp.json();
    } catch (err) {
        document.body.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:center;height:100vh;
                        font-family:Inter,sans-serif;color:#94a3b8;background:#0a0e1a;flex-direction:column;gap:16px;">
                <h1 style="font-size:24px;color:#f1f5f9;">No Results Found</h1>
                <p>Run the pipeline first: <code style="background:#1e293b;padding:4px 10px;border-radius:6px;">python scripts/run_pipeline.py</code></p>
                <p style="font-size:13px;">Then export results to generate <code>results.json</code></p>
            </div>`;
        return;
    }

    const m = data.metrics;
    const eq = data.equity_curve;
    const trades = data.trades;

    // ─── Header ─────────────────────────────────────────────────────────
    const statusEl = document.getElementById("challengeStatus");
    const statusText = statusEl.querySelector(".status-text");
    statusText.textContent = m.status;
    if (m.status === "PASSED") statusEl.classList.add("passed");
    else if (m.status === "FAILED") statusEl.classList.add("failed");
    else statusEl.classList.add("progress");

    document.getElementById("pairBadge").textContent = `${data.pair} ${data.timeframe}`;
    document.getElementById("generatedAt").textContent = `Generated: ${data.generated_at.split(".")[0]}`;

    // ─── Metric Cards ───────────────────────────────────────────────────
    const returnPct = m.return_pct;
    document.getElementById("metricReturn").textContent = `${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(2)}%`;
    document.getElementById("metricPnl").textContent = `$${m.total_pnl.toLocaleString("en-US", { minimumFractionDigits: 2 })}`;

    document.getElementById("metricWinRate").textContent = `${m.win_rate.toFixed(1)}%`;
    document.getElementById("metricTrades").textContent = `${m.winners}W / ${m.losers}L (${m.trades} total)`;

    document.getElementById("metricDrawdown").textContent = `${m.max_drawdown_pct.toFixed(2)}%`;
    document.getElementById("metricDDLimit").textContent = `/ 10.00% limit`;

    document.getElementById("metricSharpe").textContent = m.sharpe_ratio.toFixed(2);
    document.getElementById("metricPF").textContent = `PF: ${m.profit_factor.toFixed(2)}`;

    // ─── Equity Chart ───────────────────────────────────────────────────
    if (eq.length > 0) {
        const labels = eq.map(p => {
            const d = new Date(p.time);
            return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
        });
        const values = eq.map(p => p.equity);

        // Downsample if too many points
        const maxPoints = 500;
        let dsLabels = labels, dsValues = values;
        if (labels.length > maxPoints) {
            const step = Math.ceil(labels.length / maxPoints);
            dsLabels = labels.filter((_, i) => i % step === 0);
            dsValues = values.filter((_, i) => i % step === 0);
        }

        new Chart(document.getElementById("equityChart"), {
            type: "line",
            data: {
                labels: dsLabels,
                datasets: [
                    {
                        label: "Equity",
                        data: dsValues,
                        borderColor: "#6366f1",
                        backgroundColor: "rgba(99, 102, 241, 0.08)",
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                    },
                    {
                        label: "Baseline",
                        data: dsValues.map(() => m.initial_balance),
                        borderColor: "rgba(148, 163, 184, 0.3)",
                        borderDash: [6, 4],
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: "index" },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1e293b",
                        titleColor: "#f1f5f9",
                        bodyColor: "#94a3b8",
                        borderColor: "rgba(99,102,241,0.2)",
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: ctx => `$${ctx.raw.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
                        },
                    },
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: "rgba(99,102,241,0.06)" },
                        ticks: { color: "#64748b", maxTicksLimit: 8, font: { size: 11 } },
                    },
                    y: {
                        display: true,
                        grid: { color: "rgba(99,102,241,0.06)" },
                        ticks: {
                            color: "#64748b",
                            font: { size: 11 },
                            callback: v => `$${(v / 1000).toFixed(0)}k`,
                        },
                    },
                },
            },
        });

        // ─── Drawdown Chart ─────────────────────────────────────────────
        const peak = [];
        let maxEq = values[0];
        const ddPct = values.map(v => {
            if (v > maxEq) maxEq = v;
            peak.push(maxEq);
            return -((maxEq - v) / maxEq) * 100;
        });

        let dsDd = ddPct;
        if (ddPct.length > maxPoints) {
            const step = Math.ceil(ddPct.length / maxPoints);
            dsDd = ddPct.filter((_, i) => i % step === 0);
        }

        new Chart(document.getElementById("drawdownChart"), {
            type: "line",
            data: {
                labels: dsLabels,
                datasets: [{
                    label: "Drawdown",
                    data: dsDd,
                    borderColor: "#ef4444",
                    backgroundColor: "rgba(239, 68, 68, 0.1)",
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1e293b",
                        titleColor: "#f1f5f9",
                        bodyColor: "#ef4444",
                        borderColor: "rgba(239,68,68,0.2)",
                        borderWidth: 1,
                        padding: 12,
                        callbacks: { label: ctx => `${ctx.raw.toFixed(2)}%` },
                    },
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: "rgba(99,102,241,0.06)" },
                        ticks: { color: "#64748b", maxTicksLimit: 5, font: { size: 11 } },
                    },
                    y: {
                        display: true,
                        grid: { color: "rgba(99,102,241,0.06)" },
                        ticks: {
                            color: "#64748b",
                            font: { size: 11 },
                            callback: v => `${v.toFixed(1)}%`,
                        },
                        max: 0,
                    },
                },
            },
        });
    }

    // ─── Trade P&L Distribution ─────────────────────────────────────────
    if (trades.length > 0) {
        const pnls = trades.map(t => t.pnl);
        const colors = pnls.map(p => p >= 0 ? "rgba(16, 185, 129, 0.8)" : "rgba(239, 68, 68, 0.8)");
        const borders = pnls.map(p => p >= 0 ? "#10b981" : "#ef4444");

        new Chart(document.getElementById("tradeDistChart"), {
            type: "bar",
            data: {
                labels: trades.map((_, i) => `#${i + 1}`),
                datasets: [{
                    label: "P&L",
                    data: pnls,
                    backgroundColor: colors,
                    borderColor: borders,
                    borderWidth: 1,
                    borderRadius: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1e293b",
                        titleColor: "#f1f5f9",
                        bodyColor: "#94a3b8",
                        borderColor: "rgba(99,102,241,0.2)",
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: ctx => `$${ctx.raw.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#64748b", font: { size: 10 } },
                    },
                    y: {
                        grid: { color: "rgba(99,102,241,0.06)" },
                        ticks: {
                            color: "#64748b",
                            font: { size: 11 },
                            callback: v => `$${v.toLocaleString()}`,
                        },
                    },
                },
            },
        });

        // ─── Trade Table ────────────────────────────────────────────────
        const tbody = document.getElementById("tradeTableBody");
        trades.forEach((t, i) => {
            const tr = document.createElement("tr");
            const pnlClass = t.pnl >= 0 ? "pnl-positive" : "pnl-negative";
            const dirClass = t.direction === "LONG" ? "dir-long" : "dir-short";
            const exitClass = t.exit_reason === "take_profit" ? "tp" : "sl";
            const exitLabel = t.exit_reason === "take_profit" ? "TP" : "SL";
            const entryTime = new Date(t.entry_time).toLocaleDateString("en-US", { month: "short", day: "numeric" });

            tr.innerHTML = `
                <td>${i + 1}</td>
                <td class="${dirClass}">${t.direction}</td>
                <td>${t.entry_price.toFixed(5)}</td>
                <td>${t.exit_price.toFixed(5)}</td>
                <td>${t.lots.toFixed(2)}</td>
                <td class="${pnlClass}">$${t.pnl.toFixed(2)}</td>
                <td class="${pnlClass}">${t.pnl_pips > 0 ? "+" : ""}${t.pnl_pips.toFixed(1)}</td>
                <td><span class="exit-badge ${exitClass}">${exitLabel}</span></td>
                <td>${t.bars_held}</td>
            `;
            tbody.appendChild(tr);
        });

        document.getElementById("tradeSummary").textContent =
            `Avg Win: $${m.avg_win.toFixed(0)} | Avg Loss: $${Math.abs(m.avg_loss).toFixed(0)} | Avg Hold: ${m.avg_bars_held}h`;
    }

    // ─── Compliance Section ─────────────────────────────────────────────
    const compGrid = document.getElementById("complianceGrid");
    const checks = [
        { label: "Max Drawdown", current: m.max_drawdown_pct, limit: 10.0, unit: "%" },
        { label: "Profit Target", current: m.return_pct, limit: 10.0, unit: "%", inverse: true },
        { label: "Trading Days", current: m.trading_days, limit: 4, unit: " days", inverse: true },
    ];

    checks.forEach(c => {
        const passed = c.inverse ? c.current >= c.limit : c.current < c.limit;
        const pct = c.inverse
            ? Math.min(100, (c.current / c.limit) * 100)
            : Math.min(100, (c.current / c.limit) * 100);
        const barColor = passed ? "var(--accent-profit)" : "var(--accent-loss)";

        const item = document.createElement("div");
        item.className = "compliance-item";
        item.innerHTML = `
            <div class="compliance-item-header">
                <span class="compliance-label">${c.label}</span>
                <span class="compliance-status ${passed ? "pass" : "fail"}">${passed ? "PASS" : "FAIL"}</span>
            </div>
            <div class="compliance-bar-track">
                <div class="compliance-bar-fill" style="width:${pct}%;background:${barColor}"></div>
            </div>
            <div class="compliance-values">
                <span class="compliance-current">${typeof c.current === "number" ? c.current.toFixed(2) : c.current}${c.unit}</span>
                <span class="compliance-limit">${c.limit}${c.unit} limit</span>
            </div>
        `;
        compGrid.appendChild(item);
    });

})();
