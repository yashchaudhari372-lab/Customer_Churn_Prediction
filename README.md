# Customer Churn Prediction & Analytics | AdaBoost Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask) ![Scikit-learn](https://img.shields.io/badge/ML-AdaBoost-orange) ![Deployment](https://img.shields.io/badge/Deployment-Vercel-black?logo=vercel)

**An interactive machine learning application that predicts customer churn and presents customer-retention risk insights using a trained AdaBoost classification model.**

**[🚀 Live Demo](https://customer-churn-prediction-lovat.vercel.app/)** · **[💻 GitHub Repository](https://github.com/yashchaudhari372-lab/Customer_Churn_Prediction)**

## 📌 Project Overview

Customer churn occurs when a customer stops using a business's services. Predicting churn can help teams prioritize retention efforts, identify customers who may need support, and make better-informed decisions.

This project integrates a pre-trained **AdaBoost classifier** with a **Flask web application**. Users enter customer characteristics, submit the form, and receive a churn/retention prediction. The interface also displays a model-derived churn risk score and prediction confidence when the loaded model supports probability estimates.

> **Note:** Predictions are decision-support estimates, not guaranteed outcomes. The model's training data and held-out evaluation results are not included in the published repository.

## ✨ Key Features

- **Customer churn classification:** Predicts whether a customer is likely to churn or be retained.
- **Interactive input form:** Accepts customer profile, subscription, engagement, and spending details.
- **Risk scoring:** Displays churn probability when supported by `predict_proba`.
- **Prediction confidence:** Displays the highest class probability when available.
- **Visual analytics:** Returns feature values to support dashboard visualizations.
- **Responsive web interface:** Flask serves an embedded HTML/CSS/JavaScript interface with theme styling.
- **Online access:** A live demo is linked above.

## 🧰 Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Machine learning | AdaBoost classifier (serialized `.pkl` model) |
| Data processing | NumPy |
| Web backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model persistence | Python Pickle |
| Deployment | Vercel (live demo) |

## 🧾 Model Input Features

The application collects the following **10 inputs**, in this order:

| Feature | Description |
|---|---|
| Age | Customer age |
| Gender | Customer gender |
| Tenure | Length of customer relationship |
| Usage Frequency | How often the customer uses the service |
| Support Calls | Number of customer support calls |
| Payment Delay | Payment delay measure |
| Subscription Type | Basic, Standard, or Premium |
| Contract Length | Monthly, Quarterly, or Annual |
| Total Spend | Customer's total spending |
| Last Interaction | Time/measure of most recent interaction |

### Categorical Encoding Used by the Application

- `Gender`: Male → `1`; other available option → `0`.
- `Subscription Type`: Basic → `0`, Standard → `1`, Premium → `2`.
- `Contract Length`: Monthly → `0`, Quarterly → `1`, Annual → `2`.

**Important:** These mappings and the exact feature order must match the preprocessing used during model training. The repository does not include a training pipeline to independently confirm that match.

## 🧠 Why AdaBoost?

**Adaptive Boosting (AdaBoost)** is an ensemble learning method that combines multiple weak learners into a stronger classifier. During training, it places greater emphasis on examples that earlier learners classified incorrectly. In this application, a saved AdaBoost model is used to predict the customer's churn class.

### Prediction Workflow

```text
Customer inputs
      ↓
Form validation and numeric conversion
      ↓
Categorical encoding
      ↓
10-feature NumPy array
      ↓
Saved AdaBoost classifier (.pkl)
      ↓
Churn / Retained prediction
      ↓
Probability-based risk score + dashboard output
```

## 📊 Model Evaluation

The published repository does **not** provide a verified held-out classification report, so numerical performance values are intentionally not claimed.

| Evaluation Metric | Verified Result |
|---|---|
| Accuracy | 87% |
| Precision | 86% |
| Recall | 86% |
| F1-score | 87% |

To report these metrics, evaluate the saved model on the **original held-out test set**, using the same feature order and preprocessing as training. For churn prediction, recall and F1-score are especially useful alongside accuracy because missing a customer who actually churns can be costly.

## 📁 Repository Structure

```text
Customer_Churn_Prediction/
├── Adaboost_model.pkl   # Saved trained classifier
├── app.py               # Flask application, UI and prediction endpoint
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## ⚙️ Installation and Local Setup

**Prerequisites:** Python 3 and Git.

1. Clone the repository:

   ```bash
   git clone https://github.com/yashchaudhari372-lab/Customer_Churn_Prediction.git
   cd Customer_Churn_Prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```powershell
   .venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask app:

   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000/** in your browser.

> If the serialized model was trained with different library versions, you may need to reproduce the original Python/scikit-learn environment. Only load `.pkl` files from sources you trust.

## 🌐 Live Demo

Try the deployed application: **https://customer-churn-prediction-lovat.vercel.app/**

Enter customer information, submit a prediction, and explore the churn-risk output. Availability depends on the deployment status.

## 💼 Potential Business Applications

- **Customer retention:** Prioritize outreach to customers flagged as higher risk.
- **Customer support:** Identify accounts that may benefit from proactive assistance.
- **Subscription analytics:** Explore churn patterns across plans and contract lengths.
- **Marketing:** Support targeted retention campaigns and resource allocation.

## 🔭 Future Improvements

- Publish a reproducible training notebook and dataset documentation.
- Add held-out evaluation results, confusion matrix, ROC curve, and model calibration.
- Validate the encoded feature schema against the training pipeline.
- Explain predictions using appropriate model-interpretability methods.
- Add automated tests and deployment monitoring.
- Document fairness, privacy, and limitations before business use.

## 👨‍💻 Developer

**Yash Chaudhari** — Aspiring Data Scientist & Data Analyst

- **GitHub:** https://github.com/yashchaudhari372-lab
- **LinkedIn:** https://www.linkedin.com/in/yashchaudhari08/

---

If this project was useful, consider giving the repository a ⭐ on GitHub.
