import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# --- Load the AdaBoost Pickle Model ---
MODEL_PATH = "Adaboost_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("AdaBoost model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found in root directory.")

# Fallback feature list based on the uploaded pkl metadata
FEATURE_NAMES = [
    "Age", "Gender", "Tenure", "Usage Frequency", 
    "Support Calls", "Payment Delay", "Subscription Type", 
    "Contract Length", "Total Spend", "Last Interaction"
]

if model and hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)

# Categorical mapping helpers
CATEGORICAL_MAPS = {
    "Gender": {"Male": 0, "Female": 1},
    "Subscription Type": {"Basic": 0, "Standard": 1, "Premium": 2},
    "Contract Length": {"Monthly": 0, "Quarterly": 1, "Annual": 2}
}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost AI Predictive Analytics Dashboard</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            /* Cyber Dark (Default) */
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --bg-input: #374151;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --border: #374151;
            --accent-primary: #6366f1;
            --accent-secondary: #ec4899;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --gradient-1: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --gradient-card: linear-gradient(145deg, rgba(31, 41, 55, 0.7), rgba(17, 24, 39, 0.8));
            --success: #10b981;
            --danger: #ef4444;
            --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --font-family: 'Plus Jakarta Sans', sans-serif;
        }

        [data-theme="neon"] {
            --bg-primary: #05050a;
            --bg-secondary: #0d0e1b;
            --bg-card: #15182e;
            --bg-input: #1e2342;
            --text-main: #ffffff;
            --text-muted: #8a94c0;
            --border: #2a315c;
            --accent-primary: #00f2fe;
            --accent-secondary: #4facfe;
            --accent-glow: rgba(0, 242, 254, 0.5);
            --gradient-1: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #000851 100%);
            --gradient-card: linear-gradient(145deg, rgba(21, 24, 46, 0.8), rgba(13, 14, 27, 0.9));
            --success: #00ff88;
            --danger: #ff0055;
        }

        [data-theme="sunset"] {
            --bg-primary: #1a0b1e;
            --bg-secondary: #26122c;
            --bg-card: #361a3e;
            --bg-input: #4a2454;
            --text-main: #fff0f5;
            --text-muted: #c39ebc;
            --border: #5c2d67;
            --accent-primary: #ff512f;
            --accent-secondary: #f09819;
            --accent-glow: rgba(255, 81, 47, 0.4);
            --gradient-1: linear-gradient(135deg, #ff512f 0%, #dd2476 50%, #f09819 100%);
            --gradient-card: linear-gradient(145deg, rgba(54, 26, 62, 0.8), rgba(38, 18, 44, 0.9));
            --success: #20e2d7;
            --danger: #ff3366;
        }

        [data-theme="emerald"] {
            --bg-primary: #061e14;
            --bg-secondary: #0b2e20;
            --bg-card: #12402e;
            --bg-input: #1b543e;
            --text-main: #ecfdf5;
            --text-muted: #a7f3d0;
            --border: #276a50;
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-glow: rgba(16, 185, 129, 0.4);
            --gradient-1: linear-gradient(135deg, #10b981 0%, #059669 50%, #06b6d4 100%);
            --gradient-card: linear-gradient(145deg, rgba(18, 64, 46, 0.8), rgba(11, 46, 32, 0.9));
            --success: #34d399;
            --danger: #f87171;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            font-family: var(--font-family);
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* --- Animations --- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 15px var(--accent-glow); }
            50% { box-shadow: 0 0 30px var(--accent-glow); }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }

        .animate-fade-in { animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        .animate-float { animation: float 4s ease-in-out infinite; }

        /* --- Header Section --- */
        header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1.2rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(10px);
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 42px;
            height: 42px;
            background: var(--gradient-1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            color: #fff;
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .logo-text h1 {
            font-size: 1.3rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .theme-selector-wrapper {
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--bg-card);
            padding: 6px 14px;
            border-radius: 30px;
            border: 1px solid var(--border);
        }

        .theme-selector-wrapper label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .theme-select {
            background: transparent;
            border: none;
            color: var(--text-main);
            font-weight: 700;
            font-size: 0.85rem;
            cursor: pointer;
            outline: none;
        }

        /* --- Layout Grid --- */
        .dashboard-container {
            max-width: 1440px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-container { grid-template-columns: 1fr; }
        }

        /* --- Cards Styling --- */
        .card {
            background: var(--gradient-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.8rem;
            box-shadow: var(--card-shadow);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.8rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-title i {
            color: var(--accent-primary);
        }

        /* --- Form Elements --- */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-control {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px var(--accent-glow);
        }

        .btn-predict {
            width: 100%;
            margin-top: 1.5rem;
            padding: 14px;
            background: var(--gradient-1);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 20px var(--accent-glow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px var(--accent-glow);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        /* --- Right Analytics Area --- */
        .analytics-grid {
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 1.5rem;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.2rem;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .metric-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: rgba(99, 102, 241, 0.1);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
        }

        .metric-data h4 {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .metric-data div {
            font-size: 1.4rem;
            font-weight: 800;
            margin-top: 2px;
        }

        /* --- Output Dashboard Banner --- */
        .prediction-result-banner {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            border-left: 6px solid var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }

        .result-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .badge-success { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }

        /* --- Charts Grid --- */
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .charts-container { grid-template-columns: 1fr; }
        }

        .chart-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.2rem;
            position: relative;
            height: 300px;
        }

        /* Loading Overlay */
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(10, 14, 23, 0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
            border-radius: 20px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--border);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <!-- Top Navigation Header -->
    <header>
        <div class="logo-container">
            <div class="logo-icon animate-float">
                <i class="fa-solid fa-brain"></i>
            </div>
            <div class="logo-text">
                <h1>AdaBoost Intelligence Dashboard</h1>
                <p>Real-time Ensemble ML Analytics Framework</p>
            </div>
        </div>

        <!-- Color Theme Switcher -->
        <div class="theme-selector-wrapper">
            <i class="fa-solid fa-palette" style="color: var(--accent-primary);"></i>
            <label for="themeSelect">Theme:</label>
            <select id="themeSelect" class="theme-select" onchange="changeTheme(this.value)">
                <option value="dark">Cyber Dark</option>
                <option value="neon">Electric Neon</option>
                <option value="sunset">Sunset Glow</option>
                <option value="emerald">Emerald Forest</option>
            </select>
        </div>
    </header>

    <!-- Main Dashboard Container -->
    <div class="dashboard-container">
        
        <!-- Input Form Side -->
        <div class="card animate-fade-in">
            <div class="loading-overlay" id="loadingOverlay">
                <div class="spinner"></div>
                <p style="margin-top: 10px; font-weight: 600;">Processing Ensemble Model...</p>
            </div>

            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-sliders"></i> Input Parameters
                </div>
                <span style="font-size: 0.75rem; color: var(--text-muted);">10 Active Features</span>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="Age" class="input-control" value="35" required min="18" max="100">
                    </div>

                    <div class="form-group">
                        <label>Gender</label>
                        <select name="Gender" class="input-control">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Tenure (Months)</label>
                        <input type="number" name="Tenure" class="input-control" value="12" required min="0">
                    </div>

                    <div class="form-group">
                        <label>Usage Freq (Days/Mo)</label>
                        <input type="number" name="Usage Frequency" class="input-control" value="18" required min="0" max="30">
                    </div>

                    <div class="form-group">
                        <label>Support Calls</label>
                        <input type="number" name="Support Calls" class="input-control" value="2" required min="0">
                    </div>

                    <div class="form-group">
                        <label>Payment Delay (Days)</label>
                        <input type="number" name="Payment Delay" class="input-control" value="1" required min="0">
                    </div>

                    <div class="form-group">
                        <label>Subscription</label>
                        <select name="Subscription Type" class="input-control">
                            <option value="Basic">Basic</option>
                            <option value="Standard" selected>Standard</option>
                            <option value="Premium">Premium</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Contract</label>
                        <select name="Contract Length" class="input-control">
                            <option value="Monthly">Monthly</option>
                            <option value="Quarterly" selected>Quarterly</option>
                            <option value="Annual">Annual</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Total Spend ($)</label>
                        <input type="number" step="0.01" name="Total Spend" class="input-control" value="450.00" required min="0">
                    </div>

                    <div class="form-group">
                        <label>Last Interaction</label>
                        <input type="number" name="Last Interaction" class="input-control" value="5" required min="0">
                    </div>

                </div>

                <button type="submit" class="btn-predict">
                    <i class="fa-solid fa-bolt"></i> Run Predictive Inference
                </button>
            </form>
        </div>

        <!-- Right Analytics Output Side -->
        <div class="analytics-grid animate-fade-in" style="animation-delay: 0.15s;">
            
            <!-- Result Overview Banner -->
            <div class="card" style="padding: 1.2rem 1.8rem;">
                <div class="prediction-result-banner">
                    <div>
                        <div class="result-title">Model Classification Output</div>
                        <div class="result-value" id="resultLabel">
                            -- <span class="badge badge-success" id="resultBadge">Awaiting Data</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div class="result-title">Confidence Probability</div>
                        <div class="result-value" id="confidenceValue" style="color: var(--accent-primary);">--%</div>
                    </div>
                </div>

                <!-- KPI Metric Row -->
                <div class="metrics-row">
                    <div class="metric-card">
                        <div class="metric-icon"><i class="fa-solid fa-microchip"></i></div>
                        <div class="metric-data">
                            <h4>Model Type</h4>
                            <div>AdaBoost</div>
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-icon"><i class="fa-solid fa-layer-group"></i></div>
                        <div class="metric-data">
                            <h4>Estimators</h4>
                            <div id="estimatorsCount">50 Trees</div>
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-icon"><i class="fa-solid fa-gauge-high"></i></div>
                        <div class="metric-data">
                            <h4>Status</h4>
                            <div style="color: var(--success); font-size: 1.1rem; font-weight:700;">Active</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Visual Charts -->
            <div class="charts-container">
                <div class="chart-box">
                    <h4 style="margin-bottom: 10px; font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase;">
                        <i class="fa-solid fa-chart-pie"></i> Probability Breakdown
                    </h4>
                    <canvas id="probabilityChart"></canvas>
                </div>

                <div class="chart-box">
                    <h4 style="margin-bottom: 10px; font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase;">
                        <i class="fa-solid fa-chart-bar"></i> Feature Profile
                    </h4>
                    <canvas id="featureChart"></canvas>
                </div>
            </div>

        </div>

    </div>

    <script>
        let probChart = null;
        let featChart = null;

        // Dynamic Theme Switcher
        function changeTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            updateChartColors();
        }

        // Color helper for Charts
        function getComputedThemeColor(variable) {
            return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
        }

        // Initialize Empty Charts
        function initCharts() {
            const ctx1 = document.getElementById('probabilityChart').getContext('2d');
            const ctx2 = document.getElementById('featureChart').getContext('2d');

            probChart = new Chart(ctx1, {
                type: 'doughnut',
                data: {
                    labels: ['Class 0 (Active)', 'Class 1 (Churn)'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: [getComputedThemeColor('--success'), getComputedThemeColor('--accent-secondary')],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { color: getComputedThemeColor('--text-main') } } }
                }
            });

            featChart = new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: ['Tenure', 'Usage', 'Calls', 'Delay', 'Spend'],
                    datasets: [{
                        label: 'Input Value Overview',
                        data: [12, 18, 2, 1, 450],
                        backgroundColor: getComputedThemeColor('--accent-primary'),
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: getComputedThemeColor('--text-muted') }, grid: { display: false } },
                        y: { ticks: { color: getComputedThemeColor('--text-muted') }, grid: { color: getComputedThemeColor('--border') } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        function updateChartColors() {
            if(!probChart || !featChart) return;
            
            probChart.data.datasets[0].backgroundColor = [
                getComputedThemeColor('--success'), 
                getComputedThemeColor('--accent-secondary')
            ];
            probChart.options.plugins.legend.labels.color = getComputedThemeColor('--text-main');
            probChart.update();

            featChart.data.datasets[0].backgroundColor = getComputedThemeColor('--accent-primary');
            featChart.options.scales.x.ticks.color = getComputedThemeColor('--text-muted');
            featChart.options.scales.y.ticks.color = getComputedThemeColor('--text-muted');
            featChart.options.scales.y.grid.color = getComputedThemeColor('--border');
            featChart.update();
        }

        // Handle AJAX Prediction Request
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const overlay = document.getElementById('loadingOverlay');
            overlay.classList.add('active');

            const formData = new FormData(this);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if(data.status === 'success') {
                    // Update Text Values
                    document.getElementById('resultLabel').innerHTML = `
                        Class ${data.prediction}
                        <span class="badge ${data.prediction === 1 ? 'badge-danger' : 'badge-success'}">
                            ${data.prediction === 1 ? 'High Risk' : 'Low Risk / Active'}
                        </span>
                    `;
                    document.getElementById('confidenceValue').innerText = (data.probabilities[data.prediction] * 100).toFixed(1) + '%';
                    
                    if (data.n_estimators) {
                        document.getElementById('estimatorsCount').innerText = `${data.n_estimators} Trees`;
                    }

                    // Update Charts
                    probChart.data.datasets[0].data = data.probabilities;
                    probChart.update();

                    featChart.data.labels = ['Tenure', 'Usage Freq', 'Calls', 'Delay', 'Last Inter.'];
                    featChart.data.datasets[0].data = [
                        formData.get('Tenure'),
                        formData.get('Usage Frequency'),
                        formData.get('Support Calls'),
                        formData.get('Payment Delay'),
                        formData.get('Last Interaction')
                    ];
                    featChart.update();
                } else {
                    alert('Prediction Error: ' + data.message);
                }
            } catch (err) {
                alert('Request Failed: ' + err.message);
            } finally {
                setTimeout(() => overlay.classList.remove('active'), 400);
            }
        });

        window.onload = initCharts;
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model file missing or failed to load."})

    try:
        # Extract features according to expected ordering
        input_data = []
        for feature in FEATURE_NAMES:
            val = request.form.get(feature)
            
            # Categorical Encoding lookup
            if feature in CATEGORICAL_MAPS:
                val = CATEGORICAL_MAPS[feature].get(val, 0)
            else:
                val = float(val) if val else 0.0
                
            input_data.append(val)

        # Prepare 2D array for prediction
        features_array = np.array([input_data])
        
        # Inference execution
        prediction = int(model.predict(features_array)[0])
        
        probabilities = [0.5, 0.5]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_array)[0].tolist()

        n_estimators = getattr(model, "n_estimators", 50)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probabilities": probabilities,
            "n_estimators": n_estimators
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
