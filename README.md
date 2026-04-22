# 🛡️ CyberShield
### AI-Powered Fraud Detection & Cybercrime Complaint Intelligence Portal

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![ML](https://img.shields.io/badge/Accuracy-94.92%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Detect. Classify. Report. Protect.

A 6th semester mini-project that detects SMS/WhatsApp fraud using 
Machine Learning, classifies it into NCRP legal categories, and 
auto-generates a cybercrime complaint PDF aligned with India's 
National Cybercrime Reporting Portal (cybercrime.gov.in).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 ML Ensemble | LinearSVC + LR + MultinomialNB — 94.92% accuracy |
| 🏷️ NCRP Classification | 8 fraud categories mapped to CAT-01 to CAT-08 |
| 🌡️ Word Risk Heatmap | Token-level explainability using LR coefficients |
| 🔁 Velocity Tracker | Cross-complaint repeat offender detection |
| 🧠 Human-in-the-Loop | Admin reviews uncertain cases → model retrains |
| 📄 NCRP PDF Generator | Auto-filled legal complaint document |
| 📋 NCRP Form Helper | Copy-per-field assistant for cybercrime.gov.in |
| 🔐 Google OAuth | Secure login with complaint history |

---

## 🚀 Setup & Run

```bash
# Clone
git clone https://github.com/ABHISHEK27Y/fraudReporter
cd fraudReporter

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.template .env
# Edit .env with your credentials

# Run
python app.py
```

Open http://127.0.0.1:5000

---

## 🧠 ML Pipeline

- **Dataset:** 26,531 rows from 7 Indian SMS sources
- **Models:** Calibrated ensemble (SVM 60% + LR 20% + NB 20%)
- **Accuracy:** 94.92% | **ROC-AUC:** 98.94% | **F1:** 95%
- **Preprocessing:** URL/Phone/OTP/Amount token replacement

---

## 📁 Project Structure

```
├── app.py              # Main Flask application
├── models.py           # SQLAlchemy database models
├── pipefinal.py        # ML training pipeline
├── templates/          # HTML templates (Jinja2)
├── requirements.txt    # Dependencies
└── .env.template       # Environment variable template
```

---

## 🔑 Environment Variables

```
SECRET_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
ADMIN_EMAIL=
DATABASE_URL=sqlite:///cybershield.db
```

---

## 👨‍💻 Tech Stack

Python · Flask · scikit-learn · SQLAlchemy · 
ReportLab · Flask-Dance · Google OAuth 2.0

---

## 📜 License
MIT — For educational purposes
