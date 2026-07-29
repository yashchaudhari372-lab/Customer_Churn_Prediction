import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the AdaBoost pickle model
MODEL_FILE = 'Adaboost_model.pkl'

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_model()

# Feature Metadata for UI representation
FEATURE_KEYS = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type', 'Contract Length', 
    'Total Spend', 'Last Interaction'
]

# HTML/CSS/JS Combined Responsive Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Predictive Analytics Dashboard</title>
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- FontAwesome icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root[data-theme="dark"] {
            --bg-primary: #0a0d14;
            --bg-secondary: #121824;
            --bg-card: rgba(22, 30, 46, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --card-glow: rgba(99, 102, 241, 0.12);
            --chart-grid: rgba(255, 255, 255, 0.05);
        }

        :root[data-theme="ocean"] {
            --bg-primary: #07131e;
            --bg-secondary: #0d2135;
            --bg-card: rgba(15, 38, 61, 0.7);
            --border-color: rgba(56, 189, 248, 0.15);
            --text-main: #f0f9ff;
            --text-muted: #7dd3fc;
            --accent-primary: #0284c7;
            --accent-secondary: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
            --card-glow: rgba(56, 189, 248, 0.15);
            --chart-grid: rgba(56, 189, 248, 0.08);
        }

        :root[data-theme="sunset"] {
            --bg-primary: #180c1e;
            --bg-secondary: #271432;
            --bg-card: rgba(48, 23, 62, 0.7);
            --border-color: rgba(244, 63, 94, 0.18);
            --text-main: #fff1f2;
            --text-muted: #fca5a5;
            --accent-primary: #f43f5e;
            --accent-secondary: #fb923c;
            --accent-gradient: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
            --card-glow: rgba(244, 63, 94, 0.15);
            --chart-grid: rgba(244, 63, 94, 0.08);
        }

        :root[data-theme="cyber"] {
            --bg-primary: #05050a;
            --bg-secondary: #0c0c16;
            --bg-card: rgba(20, 20, 35, 0.8);
            --border-color: rgba(236, 72, 153, 0.25);
            --text-main: #ffffff;
            --text-muted: #f472b6;
            --accent-primary: #ec4899;
            --accent-secondary: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #ec4899 0%, #22d3ee 100%);
            --card-glow: rgba(236, 72, 153, 0.2);
            --chart-grid: rgba(236, 72, 153, 0.1);
        }

        :root[data-theme="emerald"] {
            --bg-primary: #061512;
            --bg-secondary: #0c241f;
            --bg-card: rgba(16, 44, 38, 0.7);
            --border-color: rgba(16, 185, 129, 0.18);
            --text-main: #ecfdf5;
            --text-muted: #6ee7b7;
            --accent-primary: #10b981;
            --accent-secondary: #14b8a6;
            --accent-gradient: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            --card-glow: rgba(16, 185, 129, 0.15);
            --chart-grid: rgba(16, 185, 129, 0.08);
        }

        :root[data-theme="gold"] {
            --bg-primary: #14110c;
            --bg-secondary: #221c13;
            --bg-card: rgba(41, 34, 23, 0.7);
            --border-color: rgba(234, 179, 8, 0.18);
            --text-main: #fefce8;
            --text-muted: #fde047;
            --accent-primary: #eab308;
            --accent-secondary: #f97316;
            --accent-gradient: linear-gradient(135deg, #eab308 0%, #fb923c 100%);
            --card-glow: rgba(234, 179, 8, 0.15);
            --chart-grid: rgba(234, 179, 8, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, var(--card-glow) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, var(--card-glow) 0%, transparent 40%);
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.35rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand i {
            color: var(--accent-primary);
            -webkit-text-fill-color: initial;
            animation: pulse 2s infinite;
        }

        /* Theme Selector */
        .theme-selector {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--bg-primary);
            padding: 0.4rem 0.6rem;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .theme-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: transform 0.2s ease;
        }

        .theme-btn:hover {
            transform: scale(1.2);
        }

        .theme-btn.active {
            border-color: #fff;
            box-shadow: 0 0 10px var(--accent-primary);
        }

        /* Main Container */
        .dashboard-container {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1700px;
            margin: 0 auto;
        }

        @media (max-width: 1200px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }

        /* Card Container Styling */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-gradient);
            opacity: 0.7;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-main);
        }

        /* Form Control Layout */
        .input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .form-control {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.65rem 0.85rem;
            border-radius: 10px;
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
        }

        .form-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--card-glow);
        }

        /* Button Styling */
        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-gradient);
            color: #fff;
            border: none;
            padding: 0.85rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px var(--card-glow);
            transition: all 0.3s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--card-glow);
            filter: brightness(1.1);
        }

        /* Right Panel Graphics */
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1.5rem;
        }

        .metric-card {
            grid-column: span 12;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.25rem 1.75rem;
            border: 1px solid var(--border-color);
        }

        .metric-info h4 {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-prediction {
            padding: 0.5rem 1.25rem;
            border-radius: 30px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-positive {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .chart-card-lg {
            grid-column: span 7;
            height: 380px;
        }

        .chart-card-sm {
            grid-column: span 5;
            height: 380px;
        }

        @media (max-width: 992px) {
            .chart-card-lg, .chart-card-sm {
                grid-column: span 12;
            }
        }

        /* Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }

        .fade-in {
            animation: fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <i class="fa-solid fa-chart-line"></i>
            <span>AdaBoost AI Analytics</span>
        </div>

        <div class="theme-selector">
            <div class="theme-btn active" style="background: #6366f1;" onclick="setTheme('dark', this)" title="Dark Mode"></div>
            <div class="theme-btn" style="background: #0284c7;" onclick="setTheme('ocean', this)" title="Ocean Cyan"></div>
            <div class="theme-btn" style="background: #f43f5e;" onclick="setTheme('sunset', this)" title="Sunset Glow"></div>
            <div class="theme-btn" style="background: #ec4899;" onclick="setTheme('cyber', this)" title="Cyberpunk"></div>
            <div class="theme-btn" style="background: #10b981;" onclick="setTheme('emerald', this)" title="Emerald Forest"></div>
            <div class="theme-btn" style="background: #eab308;" onclick="setTheme('gold', this)" title="Luxury Gold"></div>
        </div>
    </header>

    <div class="dashboard-container">
        <!-- Input Form Section -->
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-sliders" style="color: var(--accent-primary);"></i>
                    Model Parameters
                </div>
            </div>

            <form id="prediction-form" onsubmit="handlePredict(event)">
                <div class="input-grid">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" class="form-control" name="Age" value="34" required>
                    </div>

                    <div class="form-group">
                        <label>Gender</label>
                        <select class="form-control" name="Gender">
                            <option value="0">Female (0)</option>
                            <option value="1">Male (1)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Tenure (Months)</label>
                        <input type="number" class="form-control" name="Tenure" value="12" required>
                    </div>

                    <div class="form-group">
                        <label>Usage Frequency</label>
                        <input type="number" class="form-control" name="Usage Frequency" value="18" required>
                    </div>

                    <div class="form-group">
                        <label>Support Calls</label>
                        <input type="number" class="form-control" name="Support Calls" value="2" required>
                    </div>

                    <div class="form-group">
                        <label>Payment Delay</label>
                        <input type="number" class="form-control" name="Payment Delay" value="1" required>
                    </div>

                    <div class="form-group">
                        <label>Subscription Type</label>
                        <select class="form-control" name="Subscription Type">
                            <option value="0">Basic (0)</option>
                            <option value="1">Standard (1)</option>
                            <option value="2">Premium (2)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Contract Length</label>
                        <select class="form-control" name="Contract Length">
                            <option value="0">Monthly (0)</option>
                            <option value="1">Quarterly (1)</option>
                            <option value="2">Annual (2)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Total Spend ($)</label>
                        <input type="number" step="0.01" class="form-control" name="Total Spend" value="450.50" required>
                    </div>

                    <div class="form-group">
                        <label>Last Interaction</label>
                        <input type="number" class="form-control" name="Last Interaction" value="5" required>
                    </div>

                    <button type="submit" class="btn-submit">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Run Prediction Analysis
                    </button>
                </div>
            </form>
        </div>

        <!-- Dashboard Outputs -->
        <div class="analytics-grid fade-in">
            <!-- Output Metrics -->
            <div class="metric-card span-12">
                <div class="metric-info">
                    <h4>CLASSIFICATION RESULT</h4>
                    <div id="prediction-text" class="metric-value">Waiting...</div>
                </div>
                <div id="status-badge" class="badge-prediction badge-positive">Ready</div>
                <div class="metric-info" style="text-align: right;">
                    <h4>CONFIDENCE SCORE</h4>
                    <div id="confidence-score" class="metric-value" style="color: var(--accent-primary);">0.0%</div>
                </div>
            </div>

            <!-- Radar / Multi Feature View Chart -->
            <div class="glass-card chart-card-lg">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-chart-radar" style="color: var(--accent-primary);"></i>
                        Feature Input Distribution
                    </div>
                </div>
                <div style="height: 300px; position: relative;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <!-- Probability Doughnut Chart -->
            <div class="glass-card chart-card-sm">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-chart-pie" style="color: var(--accent-primary);"></i>
                        Probability Breakdown
                    </div>
                </div>
                <div style="height: 300px; position: relative;">
                    <canvas id="pieChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let radarChartInst = null;
        let pieChartInst = null;

        function setTheme(themeName, element) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            element.classList.add('active');
            updateChartTheme();
        }

        function getAccentColor() {
            return getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim();
        }

        function getSecondaryColor() {
            return getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim();
        }

        function getGridColor() {
            return getComputedStyle(document.documentElement).getPropertyValue('--chart-grid').trim();
        }

        function initCharts() {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const ctxPie = document.getElementById('pieChart').getContext('2d');

            radarChartInst = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Age', 'Tenure', 'Usage Freq', 'Support Calls', 'Payment Delay', 'Last Interaction'],
                    datasets: [{
                        label: 'Normalized Values',
                        data: [34, 12, 18, 2, 1, 5],
                        backgroundColor: 'rgba(99, 102, 241, 0.25)',
                        borderColor: getAccentColor(),
                        borderWidth: 2,
                        pointBackgroundColor: getAccentColor()
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            grid: { color: getGridColor() },
                            angleLines: { color: getGridColor() },
                            pointLabels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } },
                            ticks: { display: false }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            pieChartInst = new Chart(ctxPie, {
                type: 'doughnut',
                data: {
                    labels: ['Class 0', 'Class 1'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: [getAccentColor(), getSecondaryColor()],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } }
                        }
                    }
                }
            });
        }

        function updateChartTheme() {
            if (!radarChartInst || !pieChartInst) return;

            radarChartInst.data.datasets[0].borderColor = getAccentColor();
            radarChartInst.data.datasets[0].pointBackgroundColor = getAccentColor();
            radarChartInst.options.scales.r.grid.color = getGridColor();
            radarChartInst.options.scales.r.angleLines.color = getGridColor();
            radarChartInst.update();

            pieChartInst.data.datasets[0].backgroundColor = [getAccentColor(), getSecondaryColor()];
            pieChartInst.update();
        }

        async function handlePredict(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((val, key) => data[key] = parseFloat(val));

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('prediction-text').innerText = `Class ${result.prediction}`;
                    document.getElementById('confidence-score').innerText = `${(result.max_probability * 100).toFixed(1)}%`;
                    
                    const badge = document.getElementById('status-badge');
                    badge.innerText = `Prediction: Class ${result.prediction}`;
                    badge.className = result.prediction === 1 ? 'badge-prediction badge-negative' : 'badge-prediction badge-positive';

                    // Update Charts
                    pieChartInst.data.datasets[0].data = result.probabilities;
                    pieChartInst.update();

                    radarChartInst.data.datasets[0].data = [
                        data['Age'], 
                        data['Tenure'], 
                        data['Usage Frequency'], 
                        data['Support Calls'], 
                        data['Payment Delay'], 
                        data['Last Interaction']
                    ];
                    radarChartInst.update();
                } else {
                    alert('Prediction Error: ' + result.error);
                }
            } catch (err) {
                console.error(err);
                alert('Error submitting model request.');
            }
        }

        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'AdaBoost pickle model is not loaded correctly.'})

    try:
        data = request.get_json()
        
        # Build array based on model requirements
        input_data = np.array([[
            data.get('Age', 0),
            data.get('Gender', 0),
            data.get('Tenure', 0),
            data.get('Usage Frequency', 0),
            data.get('Support Calls', 0),
            data.get('Payment Delay', 0),
            data.get('Subscription Type', 0),
            data.get('Contract Length', 0),
            data.get('Total Spend', 0.0),
            data.get('Last Interaction', 0)
        ]])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'max_probability': max(probabilities)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
