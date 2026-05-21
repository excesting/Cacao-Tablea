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

function initAnalyticsPage() {
    console.log('[Analytics] Initializing dashboard...');

    if (typeof Chart === 'undefined') {
        console.error('Chart.js failed to load.');
        return;
    }

    // Set loading placeholders
    const cardIds = [
        'card-demand', 'card-defect-rate', 'card-required', 
        'card-recommended', 'card-batches', 'card-expected-defect', 
        'card-expected-usable', 'card-gap'
    ];
    cardIds.forEach(id => setLoading(id, true));

    // Fetch BOTH APIs simultaneously
    Promise.all([
        fetch('/api/forecast').then(res => res.json()),
        fetch('/api/stats').then(res => res.json())
    ])
    .then(([forecastData, statsData]) => {
        
        // 1. Render the charts
        renderForecastChart(forecastData);
        renderDefectTrendChart(forecastData);

        // Check if fallback data is being used
        if (forecastData.status && forecastData.status.startsWith('Fallback')) {
            const badge = document.getElementById('forecast-status-badge');
            if (badge) {
                badge.innerText = 'Demo Data';
                badge.classList.add('bg-yellow-50', 'text-yellow-700', 'border-yellow-100');
                badge.classList.remove('hidden');
            }
        } else {
            const badge = document.getElementById('forecast-status-badge');
            if (badge) badge.classList.remove('hidden');
        }

        // 2. Extract Base Variables
        // We only look at Week 1 for current manufacturing goals
        const week1Demand = forecastData.expected_demand[0] || 0;
        
        // Defect rate is calculated from the 4-week rolling yield
        const currentYield = statsData.current_yield || 100;
        const defectRate = 1 - (currentYield / 100);

        // 3. Mathematical Calculations
        const BATCH_SIZE = 1166;
        
        // Prevent dividing by zero if defect rate happens to be 100%
        const requiredProduction = defectRate < 1 ? (week1Demand / (1 - defectRate)) : week1Demand;
        
        // Round up batches to prevent shortfall
        const batches = Math.ceil(requiredProduction / BATCH_SIZE);
        
        const recommendedProduction = batches * BATCH_SIZE;
        const expectedDefect = Math.round(recommendedProduction * defectRate);
        const expectedUsable = recommendedProduction - expectedDefect;
        const gap = expectedUsable - week1Demand;
        const kilosNeeded = (recommendedProduction * (1.5 / 117)).toFixed(1);

        // 4. Update the UI Cards
        cardIds.forEach(id => setLoading(id, false));

        document.getElementById('card-demand').innerText = week1Demand.toLocaleString();
        document.getElementById('card-defect-rate').innerText = (defectRate * 100).toFixed(2) + "%";
        document.getElementById('card-required').innerText = Math.ceil(requiredProduction).toLocaleString();
        
        document.getElementById('card-recommended').innerText = recommendedProduction.toLocaleString();
        document.getElementById('card-batches').innerText = batches + " Full Batches Needed";

        document.getElementById('card-expected-defect').innerText = expectedDefect.toLocaleString() + " pcs";
        document.getElementById('card-expected-usable').innerText = expectedUsable.toLocaleString() + " pcs";

        // Style the Gap dynamically
        const gapElement = document.getElementById('card-gap');
        if (gap >= 0) {
            gapElement.innerText = "+" + gap.toLocaleString() + " pcs (Safety Stock)";
            gapElement.className = "text-xl font-bold text-leaf-600 mt-1";
        } else {
            gapElement.innerText = gap.toLocaleString() + " pcs (Shortfall!)";
            gapElement.className = "text-xl font-bold text-red-600 mt-1";
        }

        if (document.getElementById('card-kilos')) {
            document.getElementById('card-kilos').innerText = kilosNeeded;
        }

    })
    .catch(err => {
        console.error("Dashboard calculation error:", err);
        cardIds.forEach(id => showError(id, 'Error'));
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
// 4. SCANNER UTILITIES
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
            window.location.reload(); 
        } else {
            alert("Error saving: " + data.error);
        }
    })
    .catch(err => console.error("Save failed:", err));
}
