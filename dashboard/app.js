/**
 * Antigravity Live Dashboard JS
 */

let equityChart;
let equityData = [];
let timeData = [];

async function fetchLiveState() {
    try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error("Failed to fetch state");
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        console.log("Waiting for bot to start...", e);
        document.getElementById('phaseBadge').textContent = "OFFLINE";
        document.getElementById('phaseBadge').style.background = "rgba(239, 68, 68, 0.2)";
        document.getElementById('phaseBadge').style.color = "#EF4444";
    }
}

function updateDashboard(state) {
    const config = state.config || {};
    
    // Status Badge
    const badge = document.getElementById('phaseBadge');
    badge.textContent = config.phase || "LIVE";
    badge.style.background = "rgba(16, 185, 129, 0.2)";
    badge.style.color = "var(--profit)";

    // Top Metrics
    const balance = config.starting_balance || 0;
    const targetPct = config.profit_target_pct || 14.0;
    
    document.getElementById('valBalance').textContent = `$${balance.toLocaleString()}`;
    document.getElementById('valDailyTrades').textContent = state.daily_trades_count || 0;

    // Calculate P&L from journal
    const journal = state.trade_journal || [];
    let totalPnl = 0;
    let wins = 0;
    
    journal.forEach(t => {
        totalPnl += (t.pnl || 0);
        if (t.pnl > 0) wins++;
    });

    const valPnl = document.getElementById('valPnl');
    valPnl.textContent = `${totalPnl >= 0 ? '+' : '-'}$${Math.abs(totalPnl).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    valPnl.className = `metric-value ${totalPnl >= 0 ? 'profit' : 'loss'}`;

    const winRate = journal.length > 0 ? (wins / journal.length * 100).toFixed(1) : 0;
    document.getElementById('valWinRate').textContent = `Win Rate: ${winRate}%`;

    // Progress Bar & Target Tracking
    let progress = 0;
    const isFunded = config.phase === "FUNDED";
    let targetAmount = 0;
    
    if (isFunded) {
        document.getElementById('valProfitLeft').textContent = `Status: Funded & Payout Ready`;
        document.getElementById('valProgress').textContent = `Infinity`;
        document.getElementById('progressFill').style.width = `100%`;
    } else {
        if (balance > 0 && targetPct > 0) {
            targetAmount = balance * (targetPct / 100);
            progress = Math.min((totalPnl / targetAmount) * 100, 100);
            if (progress < 0) progress = 0;
            
            let amountLeft = targetAmount - totalPnl;
            if (amountLeft < 0) amountLeft = 0;
            
            document.getElementById('valProfitLeft').textContent = `Profit Left: $${amountLeft.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        }
        document.getElementById('valProgress').textContent = `${progress.toFixed(1)}%`;
        document.getElementById('progressFill').style.width = `${progress}%`;
    }

    // Active Trades
    const active = state.managed_positions || {};
    const activeIds = Object.keys(active);
    document.getElementById('activeCount').textContent = activeIds.length;
    
    const activeBody = document.getElementById('activeTradesBody');
    if (activeIds.length === 0) {
        activeBody.innerHTML = `<tr><td colspan="5" class="empty-state">No active trades. Scanning markets...</td></tr>`;
    } else {
        activeBody.innerHTML = '';
        activeIds.forEach(id => {
            const t = active[id];
            const dir = t.direction === 1 ? 'LONG' : 'SHORT';
            const dirCls = t.direction === 1 ? 'dir-long' : 'dir-short';
            
            let statusHtml = '<span class="status-tag">Open</span>';
            if (t.breakeven_locked) {
                statusHtml = '<span class="status-tag locked">BE Locked</span>';
            }
            if (t.pending_reason) {
                statusHtml = '<span class="status-tag brain">Brain Exiting</span>';
            }

            activeBody.innerHTML += `
                <tr>
                    <td><strong>${t.pair}</strong></td>
                    <td class="${dirCls}">${dir}</td>
                    <td>${t.entry_price.toFixed(5)}</td>
                    <td>${t.original_lots}</td>
                    <td>${statusHtml}</td>
                </tr>
            `;
        });
    }

    // Trade History (last 5)
    const historyBody = document.getElementById('historyBody');
    document.getElementById('historyCount').textContent = journal.length;
    historyBody.innerHTML = '';
    
    const recent = journal.slice(-5).reverse();
    if (recent.length === 0) {
        historyBody.innerHTML = `<tr><td colspan="4" class="empty-state">No history yet.</td></tr>`;
    } else {
        recent.forEach(t => {
            const pnlCls = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
            const sign = t.pnl >= 0 ? '+' : '';
            historyBody.innerHTML += `
                <tr>
                    <td><strong>${t.pair || '--'}</strong></td>
                    <td>${new Date(t.exit_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
                    <td class="${pnlCls}">${sign}$${(t.pnl || 0).toFixed(2)}</td>
                    <td>${t.reason.replace('_', ' ')}</td>
                </tr>
            `;
        });
    }

    // Update Chart
    updateChart(journal);
}

function initChart() {
    const ctx = document.getElementById('equityChart').getContext('2d');
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Realized P&L',
                data: [],
                borderColor: '#8B5CF6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94A3B8' }
                }
            }
        }
    });
}

function updateChart(journal) {
    if (!equityChart || journal.length === 0) return;
    
    let cumPnl = 0;
    const data = [0]; // start at 0
    const labels = ['Start'];

    journal.forEach((t, i) => {
        cumPnl += (t.pnl || 0);
        data.push(cumPnl);
        labels.push(i+1);
    });

    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    
    // Change color if negative
    if (cumPnl < 0) {
        equityChart.data.datasets[0].borderColor = '#EF4444';
        equityChart.data.datasets[0].backgroundColor = 'rgba(239, 68, 68, 0.1)';
    } else {
        equityChart.data.datasets[0].borderColor = '#8B5CF6';
        equityChart.data.datasets[0].backgroundColor = 'rgba(139, 92, 246, 0.1)';
    }
    
    equityChart.update();
}

// Initialize
initChart();
fetchLiveState();
// Poll every 3 seconds
setInterval(fetchLiveState, 3000);
