import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

MODEL_FILE = "Adaboost_model.pkl"

# Load the trained AdaBoost model
model = None
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_FILE} not found. Please place it in the same directory.")

# Feature names aligned with the loaded model
FEATURE_NAMES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="indigo">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Analytics & Prediction Dashboard</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <!-- Bootstrap 5 & FontAwesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root[data-theme="indigo"] {
            --bg-color: #f4f7fe;
            --card-bg: #ffffff;
            --text-color: #2b3674;
            --text-muted: #a3edba;
            --primary: #4318ff;
            --secondary: #6ad2ff;
            --accent: #ffb547;
            --border-color: #e9ecef;
            --card-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        }

        :root[data-theme="dark"] {
            --bg-color: #0b1437;
            --card-bg: #111c44;
            --text-color: #ffffff;
            --text-muted: #8f9bba;
            --primary: #7551ff;
            --secondary: #00f2fe;
            --accent: #01b574;
            --border-color: #2b3674;
            --card-shadow: 0px 18px 40px rgba(0, 0, 0, 0.4);
        }

        :root[data-theme="sunset"] {
            --bg-color: #fff5f2;
            --card-bg: #ffffff;
            --text-color: #4a154b;
            --text-muted: #8d6d88;
            --primary: #ff5252;
            --secondary: #ff793f;
            --accent: #ffb142;
            --border-color: #ffe0d8;
            --card-shadow: 0px 18px 40px rgba(255, 82, 82, 0.1);
        }

        :root[data-theme="cyber"] {
            --bg-color: #0d0f18;
            --card-bg: #181b2a;
            --text-color: #00f0ff;
            --text-muted: #708090;
            --primary: #ff007f;
            --secondary: #00f0ff;
            --accent: #ffe600;
            --border-color: #2a2f4c;
            --card-shadow: 0px 0px 25px rgba(0, 240, 255, 0.15);
        }

        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: all 0.3s ease;
            min-height: 100vh;
        }

        .navbar {
            background-color: var(--card-bg);
            box-shadow: var(--card-shadow);
            padding: 1rem 2rem;
            margin-bottom: 2rem;
        }

        .glass-card {
            background-color: var(--card-bg);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            padding: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1.5rem;
        }

        .glass-card:hover {
            transform: translateY(-5px);
        }

        .form-control, .form-select {
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 500;
        }

        .form-control:focus, .form-select:focus {
            background-color: var(--bg-color);
            color: var(--text-color);
            border-color: var(--primary);
            box-shadow: 0 0 0 0.25rem rgba(67, 24, 255, 0.15);
        }

        .btn-predict {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            width: 100%;
            transition: all 0.3s ease;
        }

        .btn-predict:hover {
            opacity: 0.9;
            transform: scale(1.02);
            color: #fff;
        }

        .theme-badge {
            cursor: pointer;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: inline-block;
            margin-left: 8px;
            border: 2px solid #fff;
            transition: transform 0.2s ease;
        }

        .theme-badge:hover {
            transform: scale(1.2);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .chart-container {
            position: relative;
            height: 320px;
            width: 100%;
        }

        /* Loading Overlay Animation */
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: auto;
        }

        .spinner-border {
            width: 3rem;
            height: 3rem;
            color: var(--primary);
        }
    </style>
</head>
<body>

    <!-- Top Navbar -->
    <nav class="navbar navbar-expand-lg">
        <div class="container-fluid">
            <a class="navbar-brand fw-bold" href="#" style="color: var(--text-color);">
                <i class="fa-solid fa-brain me-2" style="color: var(--primary);"></i>AdaBoost ML Dashboard
            </a>
            <div class="d-flex align-items-center">
                <span class="me-2 small fw-semibold">Theme:</span>
                <span class="theme-badge" style="background: #4318ff;" onclick="setTheme('indigo')" title="Indigo Modern"></span>
                <span class="theme-badge" style="background: #00f2fe;" onclick="setTheme('dark')" title="Dark Emerald"></span>
                <span class="theme-badge" style="background: #ff5252;" onclick="setTheme('sunset')" title="Sunset Glow"></span>
                <span class="theme-badge" style="background: #ff007f;" onclick="setTheme('cyber')" title="Cyber Neon"></span>
            </div>
        </div>
    </nav>

    <div class="container-fluid px-4">
        <div class="row">
            <!-- Left Input Column -->
            <div class="col-lg-5">
                <div class="glass-card position-relative">
                    <div id="loading" class="loading-overlay">
                        <div class="spinner-border" role="status"></div>
                    </div>
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-sliders me-2" style="color: var(--primary);"></i>Model Input Parameters</h5>
                    <form id="predictionForm">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Age</label>
                                <input type="number" class="form-control" name="Age" value="35" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Gender</label>
                                <select class="form-select" name="Gender">
                                    <option value="1">Male</option>
                                    <option value="0">Female</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Tenure (months)</label>
                                <input type="number" class="form-control" name="Tenure" value="12" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Usage Frequency</label>
                                <input type="number" class="form-control" name="Usage Frequency" value="18" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Support Calls</label>
                                <input type="number" class="form-control" name="Support Calls" value="2" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Payment Delay</label>
                                <input type="number" class="form-control" name="Payment Delay" value="1" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Subscription Type</label>
                                <select class="form-select" name="Subscription Type">
                                    <option value="0">Basic</option>
                                    <option value="1">Standard</option>
                                    <option value="2">Premium</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Contract Length</label>
                                <select class="form-select" name="Contract Length">
                                    <option value="0">Monthly</option>
                                    <option value="1">Quarterly</option>
                                    <option value="2">Annual</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Total Spend ($)</label>
                                <input type="number" step="0.01" class="form-control" name="Total Spend" value="550.00" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Last Interaction (days)</label>
                                <input type="number" class="form-control" name="Last Interaction" value="5" required>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-predict mt-4">
                            <i class="fa-solid fa-bolt me-2"></i>Generate Analytics & Prediction
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Visualizations & Analytics Column -->
            <div class="col-lg-7">
                <!-- Result Overview Cards -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="glass-card text-center">
                            <span class="text-muted small fw-bold">PREDICTION OUTPUT</span>
                            <div id="predictionResult" class="metric-value my-1" style="color: var(--primary);">--</div>
                            <span id="predictionBadge" class="badge bg-secondary">Awaiting Input</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card text-center">
                            <span class="text-muted small fw-bold">CONFIDENCE PROBABILITY</span>
                            <div id="confidenceResult" class="metric-value my-1" style="color: var(--secondary);">0.0%</div>
                            <div class="progress mt-2" style="height: 6px;">
                                <div id="confidenceBar" class="progress-bar bg-info" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Interactive Charts -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="glass-card">
                            <h6 class="fw-bold mb-3"><i class="fa-solid fa-chart-pie me-2" style="color: var(--primary);"></i>Class Probability</h6>
                            <div class="chart-container">
                                <canvas id="probabilityChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card">
                            <h6 class="fw-bold mb-3"><i class="fa-solid fa-chart-bar me-2" style="color: var(--secondary);"></i>Feature Importance Analysis</h6>
                            <div class="chart-container">
                                <canvas id="importanceChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Dashboard JavaScript Logic -->
    <script>
        let probChart, importChart;

        function setTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            updateChartColors();
        }

        function initCharts() {
            const ctxProb = document.getElementById('probabilityChart').getContext('2d');
            probChart = new Chart(ctxProb, {
                type: 'doughnut',
                data: {
                    labels: ['Class 0', 'Class 1'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: ['#6ad2ff', '#4318ff'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { animateScale: true, animateRotate: true }
                }
            });

            const ctxImport = document.getElementById('importanceChart').getContext('2d');
            importChart = new Chart(ctxImport, {
                type: 'bar',
                data: {
                    labels: {{ feature_names | tojson }},
                    datasets: [{
                        label: 'Importance',
                        data: Array(10).fill(0.1),
                        backgroundColor: '#7551ff',
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { beginAtZero: true } }
                }
            });
        }

        function updateChartColors() {
            if (!probChart || !importChart) return;
            const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim();
            const secondary = getComputedStyle(document.documentElement).getPropertyValue('--secondary').trim();

            probChart.data.datasets[0].backgroundColor = [secondary, primary];
            importChart.data.datasets[0].backgroundColor = primary;
            
            probChart.update();
            importChart.update();
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => data[key] = parseFloat(value));

            document.getElementById('loading').classList.add('active');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if(result.error) {
                    alert(result.error);
                    return;
                }

                // Update Overview Metrics
                document.getElementById('predictionResult').innerText = "Class " + result.prediction;
                document.getElementById('predictionBadge').innerText = "Status: Optimal";
                document.getElementById('predictionBadge').className = "badge bg-success";

                const maxProb = (Math.max(...result.probabilities) * 100).toFixed(1);
                document.getElementById('confidenceResult').innerText = maxProb + "%";
                document.getElementById('confidenceBar').style.width = maxProb + "%";

                // Update Probability Chart
                probChart.data.datasets[0].data = result.probabilities;
                probChart.update();

                // Update Feature Importance Chart
                importChart.data.datasets[0].data = result.feature_importances;
                importChart.update();

            } catch (err) {
                console.error(err);
                alert("Failed to fetch prediction.");
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        });

        window.onload = () => {
            initCharts();
            // Automatically run initial prediction on startup
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not loaded."}), 500

    try:
        data = request.get_json()
        
        # Prepare input array in feature sequence order
        input_data = [data.get(feat, 0.0) for feat in FEATURE_NAMES]
        features_array = np.array([input_data])

        # Model evaluation
        prediction = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0].tolist()

        # Compute or extract feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
        else:
            importances = [1.0 / len(FEATURE_NAMES)] * len(FEATURE_NAMES)

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities,
            "feature_importances": importances
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
