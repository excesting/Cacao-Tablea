// ============================================================
// TableaScan - Flask Application Script
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;

    if (path.includes('/detect')) {
        if (document.getElementById('webcamVideo') || document.getElementById('stat-total')) {
            pollProductionStats();
        }
    } else if (path.includes('/analytics')) {
        initAnalyticsPage();
    }
});

// ============================================================
// UTILITY: Show / hide skeleton loaders
// ============================================================
function setLoading(id, isLoading) {
    const el = document.getElementById(id);
    if (!el) return;
    if (isLoading) {
        el.dataset.original = el.innerText;
        el.classList.add('animate-pulse', 'text-gray-300');
        el.innerText = '···';
    } else {
        el.classList.remove('animate-pulse', 'text-gray-300');
    }
}

function showError(id, message) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerText = message;
    el.classList.add('text-red-400');
}

// ============================================================
// 1. DATABASE STATS LOGIC (Detect / Inspect Page)
// ============================================================
function pollProductionStats() {
    fetch('/api/stats')
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        })
        .then(data => {
            const yieldEl  = document.getElementById('stat-yield');
            const totalEl  = document.getElementById('stat-total');
            const crackEl  = document.getElementById('stat-cracks');
            const bloomEl  = document.getElementById('stat-bloom');

            const total = data.total_scanned || 0;

            if (yieldEl) yieldEl.innerText = (data.current_yield || 0).toFixed(1) + '%';
            if (totalEl) totalEl.innerText = total.toLocaleString();

            const crackRate = total > 0 ? (data.crack  / total * 100) : 0;
            const bloomRate = total > 0 ? (data.fat_bloom / total * 100) : 0;

            if (crackEl) crackEl.innerText = crackRate.toFixed(1) + '%';
            if (bloomEl) bloomEl.innerText = bloomRate.toFixed(1) + '%';
        })
        .catch(err => console.error('[Stats] Fetch error:', err));
}

// ============================================================
// 2. ANALYTICS PAGE LOGIC
// ============================================================
let forecastChartInstance = null;
let defectTrendInstance   = null;

// Track metrics for Production Target calculation (Demand / Yield)
let currentDemand = 0;
let currentYield = 0;

function updateProductionTarget() {
    const targetEl = document.getElementById('summary-production-target');
    if (!targetEl || currentDemand === 0 || currentYield === 0) return;

    // Formula: Demand / Yield (as decimal)
    const yieldDecimal = currentYield / 100;
    const target = currentDemand / yieldDecimal;

    setLoading('summary-production-target', false);
    targetEl.innerText = Math.ceil(target).toLocaleString();
}

function initAnalyticsPage() {
    console.log('[Analytics] Initializing dashboard...');

    if (typeof Chart === 'undefined') {
        showChartError('demandForecastChart', 'Chart.js failed to load.');
        showChartError('defectTrendChart',    'Chart.js failed to load.');
        return;
    }

    // Set loading placeholders
    ['summary-demand', 'summary-projected-beans', 'stat-yield-top', 'stat-yield-sub', 'summary-production-target'].forEach(id => setLoading(id, true));

    // --- Fetch Forecast ---
    fetch('/api/forecast')
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        })
        .then(data => {
            renderForecastChart(data);
            renderDefectTrendChart(data);

            const expectedArr = data.expected_demand || [0, 0, 0, 0];
            currentDemand = expectedArr.reduce((a, b) => a + b, 0);

            setLoading('summary-demand', false);
            const demandEl = document.getElementById('summary-demand');
            if (demandEl) demandEl.innerText = currentDemand.toLocaleString();

            setLoading('summary-projected-beans', false);
            const beansEl = document.getElementById('summary-projected-beans');
            if (beansEl) {
                // Formula: 1.5kg beans per 215 pieces
                const projected_beans = (currentDemand / 215.0) * 1.5;
                beansEl.innerText = projected_beans.toLocaleString(undefined, {
                    minimumFractionDigits: 1,
                    maximumFractionDigits: 1
                }) + ' kg';
            }

            updateProductionTarget();
            
            if (data.status && data.status.startsWith('Fallback')) {
                const badge = document.getElementById('forecast-status-badge');
                if (badge) {
                    badge.innerText = 'Demo Data';
                    badge.classList.add('bg-yellow-50', 'text-yellow-700', 'border-yellow-100');
                    badge.classList.remove('hidden');
                }
            }
        })
        .catch(err => {
            ['summary-demand', 'summary-projected-beans'].forEach(id => showError(id, 'Error'));
        });

    // --- Fetch Weekly Stats ---
    fetch('/api/stats')
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        })
        .then(data => {
            currentYield = data.current_yield || 0;
            setLoading('stat-yield-top', false);
            const yieldTop = document.getElementById('stat-yield-top');
            if (yieldTop) yieldTop.innerText = currentYield.toFixed(1) + '%';

            setLoading('stat-yield-sub', false);
            const yieldSub = document.getElementById('stat-yield-sub');
            if (yieldSub) {
                // Now explicitly calculating on 4-week rolling basis
                yieldSub.innerText = "Yield (Last 4 Weeks Rolling)";
            }

            updateProductionTarget();
        })
        .catch(err => {
            ['stat-yield-top', 'stat-yield-sub'].forEach(id => showError(id, 'Error'));
        });
}

// ============================================================
// 3. CHART RENDERING
// ============================================================
function renderForecastChart(data) {
    const canvas = document.getElementById('demandForecastChart');
    if (!canvas) return;

    if (forecastChartInstance) forecastChartInstance.destroy();

    forecastChartInstance = new Chart(canvas, {
        type: 'line',
        data: {
            labels: data.labels || ['Wk 1', 'Wk 2', 'Wk 3', 'Wk 4'],
            datasets: [
                {
                    label: 'TFT Expected Demand',
                    data: data.expected_demand || [0, 0, 0, 0],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Supply Capacity',
                    data: data.projected_supply || [0, 0, 0, 0],
                    borderColor: '#A0522D',
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { ticks: { callback: val => val.toLocaleString() } }
            }
        }
    });
}

function renderDefectTrendChart(data) {
    const canvas = document.getElementById('defectTrendChart');
    if (!canvas) return;

    if (defectTrendInstance) defectTrendInstance.destroy();

    const percentages = (data.historical_defects || []).map(val => parseFloat((val * 100).toFixed(2)));

    defectTrendInstance = new Chart(canvas, {
        type: 'line',
        data: {
            labels: data.historical_time || [],
            datasets: [{
                label: 'Defect Rate (%)',
                data: percentages,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, ticks: { callback: val => val + '%' } }
            }
        }
    });
}

// ============================================================
// 4. SCANNER UTILITIES (Refresh fix)
// ============================================================
function confirmScan(scanData) {
    fetch('/api/confirm_scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scanData)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            // Fixes the manual refresh issue
            window.location.reload(); 
        } else {
            alert("Error saving: " + data.error);
        }
    })
    .catch(err => console.error("Save failed:", err));
}

function showChartError(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const wrapper = canvas.parentElement;
    canvas.style.display = 'none';
    const msg = document.createElement('div');
    msg.className = 'flex items-center justify-center h-full text-sm text-red-400 font-medium';
    msg.innerText = `⚠ ${message}`;
    wrapper.appendChild(msg);
}