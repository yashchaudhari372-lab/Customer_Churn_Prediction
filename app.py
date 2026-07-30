import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Path to the AdaBoost pickle file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Adaboost_model.pkl')

# Load AdaBoost model safely
model = None
feature_names = []
num_features = 4  # Default fallback feature count

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # Attempt to extract feature names or counts if available in model object
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        num_features = len(feature_names)
    elif hasattr(model, 'n_features_in_'):
        num_features = model.n_features_in_
        feature_names = [f"Feature_{i+1}" for i in range(num_features)]
    else:
        feature_names = [f"Feature_{i+1}" for i in range(num_features)]
        
    print(f" Successfully loaded AdaBoost Model with {num_features} features.")
except Exception as e:
    print(f" Warning: Could not load 'Adaboost_model.pkl'. Error: {e}")
    print("Using a dummy fallback for layout preview. Ensure 'Adaboost_model.pkl' is in the root directory.")


@app.route('/')
def home():
    """Render the dashboard UI."""
    return render_template('index.html', features=feature_names, num_features=num_features)


@app.route('/api/info', methods=['GET'])
def get_model_info():
    """API endpoint to get model metadata."""
    if model is None:
        return jsonify({
            'status': 'demo_mode',
            'message': 'No Adaboost_model.pkl found in directory.',
            'features': feature_names
        })
        
    classes = list(model.classes_) if hasattr(model, 'classes_') else ['Class 0', 'Class 1']
    return jsonify({
        'status': 'active',
        'algorithm': 'AdaBoostClassifier',
        'features': feature_names,
        'num_features': len(feature_names),
        'classes': [str(c) for c in classes]
    })


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to handle feature inputs and make prediction with AdaBoost."""
    try:
        data = request.get_json(force=True)
        raw_inputs = data.get('inputs', [])
        
        # Convert input string array to floats
        features = [float(x) for x in raw_inputs]
        input_array = np.array(features).reshape(1, -1)
        
        if model is not None:
            # Real prediction using AdaBoost model
            prediction = model.predict(input_array)[0]
            
            # Predict probabilities if model supports predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_array)[0].tolist()
                classes = [str(c) for c in model.classes_]
            else:
                probabilities = [0.5, 0.5]
                classes = ['Class 0', 'Class 1']
                
            # Retrieve estimator feature importances if available
            importances = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_.tolist()
            else:
                importances = [1.0 / len(features)] * len(features)
        else:
            # Fallback mock prediction for UI testing without model file
            prediction = "Positive Result" if sum(features) > 10 else "Negative Result"
            probabilities = [0.25, 0.75] if sum(features) > 10 else [0.80, 0.20]
            classes = ['Class 0 (Low)', 'Class 1 (High)']
            importances = [round(1.0 / len(features), 2)] * len(features)

        # Format feature contribution analysis
        feature_breakdown = []
        for i, val in enumerate(features):
            f_name = feature_names[i] if i < len(feature_names) else f"Feature_{i+1}"
            imp = importances[i] if i < len(importances) else 0.1
            feature_breakdown.append({
                'name': f_name,
                'value': val,
                'importance': round(imp * 100, 2)
            })

        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'probabilities': probabilities,
            'classes': classes,
            'feature_breakdown': feature_breakdown,
            'confidence': f"{max(probabilities) * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    # Host on 0.0.0.0 to enable accessibility from local network devices
    app.run(host='0.0.0.0', port=5000, debug=True)
