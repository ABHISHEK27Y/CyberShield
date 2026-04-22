# 🛡️ CyberShield
## AI-Powered Fraud Detection & Cybercrime Complaint Intelligence Portal

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=for-the-badge)
![Google OAuth](https://img.shields.io/badge/Google_OAuth-2.0-4285F4?style=for-the-badge&logo=google&logoColor=white)

![Accuracy](https://img.shields.io/badge/Accuracy-94.92%25-brightgreen?style=for-the-badge)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-98.94%25-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-26%2C531_rows-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

> **"Detect. Classify. Report. Protect."**

*A 6th Semester Mini-Project — Computer Science & Engineering*

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Architecture](#-system-architecture) · [Dataset](#-dataset) · [ML Pipeline](#-ml-pipeline) · [Novel Contributions](#-novel-contributions) · [Deployment](#-deployment)

</div>

---

## 📌 Overview

**CyberShield** is an end-to-end fraud detection and cybercrime complaint intelligence portal built for Indian citizens. It uses a calibrated Machine Learning ensemble to detect fraudulent SMS/WhatsApp/email messages, classifies them into India's **National Cybercrime Reporting Portal (NCRP)** legal taxonomy, extracts forensic evidence, and auto-generates a ready-to-submit cybercrime complaint PDF — all within seconds.

### The Problem It Solves

| Problem | Reality | CyberShield's Solution |
|---|---|---|
| Victims don't recognise fraud | 7,000+ cybercrime complaints filed daily in India, most victims act too late | Instant ML detection with 94.92% accuracy |
| Victims don't know fraud type | NCRP requires specific category selection during filing | Auto-classifies into 8 NCRP categories (CAT-01 to CAT-08) |
| Evidence is lost before reporting | Phone numbers, URLs, amounts forgotten or deleted | Automatic forensic entity extraction |
| Filing complaints is confusing | cybercrime.gov.in has a complex multi-step form | NCRP Form Helper with per-field copy buttons |
| No repeat offender tracking | Same fraudster targets multiple victims | Cross-complaint Velocity Tracker builds threat intelligence |
| Model stays static after deployment | Real-world fraud patterns evolve | Human-in-the-Loop retraining pipeline |

---

## ✨ Features

### Core ML Features
- **Four-Tier Risk Classification** — FRAUD (≥85%) · SUSPICIOUS (≥45%) · UNCERTAIN (30–45%) · LEGIT (<30%)
- **Eight NCRP Fraud Categories** — OTP/Account Takeover · KYC Scam · Lottery Fraud · Loan/Investment · Phishing · Job Scam · Government Scheme · Unknown
- **Calibrated Ensemble Model** — LinearSVC (60%) + Logistic Regression (20%) + MultinomialNB (20%) with validation-tuned weights
- **Forensic Entity Extraction** — automatically extracts phone numbers, URLs, monetary amounts, OTP codes

### 🌡️ Novel Feature 1 — Word Risk Heatmap
Every word in the message is coloured by its Logistic Regression coefficient weight:
- 🔴 **RED** — strong fraud signal (score ≥ 0.5)
- 🟠 **AMBER** — medium signal (score ≥ 0.15)
- 🟡 **YELLOW** — weak signal (score ≥ 0.01)
- ⬜ **NORMAL** — safe/neutral word

Hover over any highlighted word to see its exact risk score. Non-technical victims can visually see exactly which words triggered the fraud alert.

### 🔁 Novel Feature 2 — Fraud Velocity Tracker
Builds a persistent threat intelligence database from crowd-sourced reports:

| Reports | Threat Level | Meaning |
|---|---|---|
| 1 | LOW | New indicator, first report |
| 2 | MEDIUM | Seen before, suspicious |
| 3 | HIGH | Repeat offender confirmed |
| 4+ | CRITICAL | Confirmed fraud infrastructure |

Tracks phone numbers and URLs across ALL complaints. Similar to how telecom companies build fraud blacklists from TRAI data. **No existing academic fraud detection project implements this.**

### 🧠 Novel Feature 3 — Human-in-the-Loop Retraining
```
UNCERTAIN prediction (30-45% probability)
        ↓
Saved to admin review queue (database)
        ↓
Admin labels: FRAUD or LEGIT
        ↓
Appended to master_dataset.csv
        ↓
Admin clicks "Retrain Now"
        ↓
pipefinal.py runs → new model pkl files saved
        ↓
Improved model used for next prediction
```
The model continuously learns from real Indian fraud messages — a production ML concept never implemented in an academic mini-project.

### 📄 Novel Feature 4 — NCRP Legal Integration
**Only system that maps ML output directly to India's NCRP legal taxonomy.**
- Auto-generated PDF complaint with 6 structured sections
- Follows government document format (Navy/Gold/Saffron palette)
- Includes ML evidence summary, extracted entities, declaration section
- NCRP Form Helper shows every field with a Copy button for cybercrime.gov.in

### Additional Features
- 🔐 **Google OAuth 2.0** login with Flask-Dance
- 📊 **User Complaint History** — every logged-in user's past reports
- ⚙️ **Admin Dashboard** — user management, velocity stats, retrain control
- 🗄️ **SQLite/PostgreSQL** database via SQLAlchemy ORM
- 📱 Responsive design — works on mobile

---

## 🎥 Demo

### Test Message (use this to demo all features)
```
Dear Customer, your SBI account has been BLOCKED due to incomplete KYC 
verification. Your OTP is 847291. Click here to verify immediately: 
http://sbi-kyc-update.net/verify and enter your Aadhaar number to avoid 
account suspension. Contact us: 9876543210. Rs.50,000 will be debited 
if not updated within 24 hours.
```

**Expected Output:**
```
Risk Level:     FRAUD (99.9% probability)
Fraud Type:     KYC / Verification Scam → NCRP CAT-02
Phone Found:    9876543210
URL Found:      http://sbi-kyc-update.net/verify
Amount Found:   Rs.50,000
OTP Found:      847291
Heatmap:        "kyc", "blocked", "verify", "aadhaar" highlighted RED
```

> Submit the same message 3 times to see the Velocity Tracker escalate:
> `9876543210` → LOW → MEDIUM → HIGH (3 reports)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
│              (SMS / WhatsApp / Email text)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 1 — PREPROCESSING                    │
│  URLs→SUSPICIOUS_URL  |  Phones→PHONE_NUMBER                 │
│  Rs amounts→MONEY_AMOUNT  |  4-6 digits→OTP_NUMBER           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2 — ML ENSEMBLE PREDICTION                │
│                                                              │
│   TF-IDF Vectorizer (Word n-gram 1,2 + Char n-gram 3,5)     │
│                       │                                      │
│        ┌──────────────┼──────────────┐                      │
│        ▼              ▼              ▼                       │
│   LinearSVC       LogReg (LR)    MultinomialNB               │
│   (60% weight)   (20% weight)   (20% weight)                 │
│        └──────────────┼──────────────┘                      │
│                       ▼                                      │
│           Weighted Probability Average                       │
│                       │                                      │
│     FRAUD / SUSPICIOUS / UNCERTAIN / LEGIT                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    FRAUD/SUS     UNCERTAIN     LEGIT
          │           │           │
          ▼           ▼           ▼
   Stage 3:      Admin Review   Result
   Fraud Type    Queue (DB)     Page
   Classifier
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 3 — FRAUD TYPE CLASSIFIER                 │
│         Rule-based regex on preprocessed tokens             │
│  OTP→CAT-01 | KYC→CAT-02 | Lottery→CAT-03 | Loan→CAT-04   │
│  Phishing→CAT-05 | Job→CAT-06 | Govt→CAT-07 | ?→CAT-08     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 4 — ENTITY EXTRACTION                     │
│   phones | urls | amounts | otps | keywords (LR coefs)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    Word Risk    Velocity     Result
    Heatmap      Tracker      Page
                             + PDF
```

---

## 📊 Dataset

### Statistics
| Property | Value |
|---|---|
| Total rows | 26,531 |
| Fraud samples | 12,851 (48.44%) |
| Legitimate samples | 13,680 (51.56%) |
| Average message length | 224 characters |
| Duplicates | 0 |
| Null values | 0 |

### Sources (7 datasets merged)
| # | Dataset | Rows | Description |
|---|---|---|---|
| 1 | final_unified_dataset.csv | ~59K (filtered) | English email spam baseline |
| 2 | Mendeley JIIT Smishing | 5,971 | Indian SMS smishing dataset |
| 3 | UCI SMS Combined | 10,961 | Standard SMS spam collection |
| 4 | Multilingual Spam | 5,572 | Hindi + English fraud SMS |
| 5 | fraud_dataset_v3.csv | 2,379 | Curated fraud messages |
| 6 | SMSSmishCollection.txt | — | Parsed smishing collection |
| 7 | analysisdataset.csv | 1,062 | ACM-verified phishing SMS (all fraud) |

### Preprocessing Filters Applied
- SMS length cap: maximum 800 characters
- Email noise removal: mailing lists, quoted replies, email headers removed
- Quality filter: minimum 3 words, must contain letters
- Deduplication: exact duplicate removal

> **Why this matters:** Most student projects use the standard UCI SMS dataset (5,574 rows, English only). Our dataset is **5× larger** and India-focused with real smishing patterns.

---

## 🤖 ML Pipeline

### Token Replacement (preprocessing)
```python
URLs              → SUSPICIOUS_URL
Indian mobiles    → PHONE_NUMBER      # regex: \b[6-9]\d{9}\b
Rs/₹ amounts      → MONEY_AMOUNT
USD amounts       → MONEY_AMOUNT_USD
4-6 digit codes   → OTP_NUMBER
```

### Vectorization
```
TF-IDF FeatureUnion:
  ├── Word n-grams (1, 2)  min_df=3
  └── Char_wb n-grams (3, 5)  min_df=2

Fit ONLY on training set — vocabulary frozen after training
Never re-fit on new data
```

### Data Split
```
Training:   70% (18,572 samples)
Validation: 15% (3,980 samples)  ← used for weight tuning
Test:       15% (3,980 samples)  ← final evaluation only
All splits stratified by label
```

### Ensemble Weight Tuning
```python
# Grid search over all weight combinations on validation set
best = (w_svm=0.60, w_lr=0.20, w_nb=0.20)

fraud_prob = 0.60 * p_svm + 0.20 * p_lr + 0.20 * p_nb
```

### Hard Negative Retraining
Misclassified validation examples appended to training set.
All three models retrained. **Vectorizer NOT re-fit.**

### Results
| Metric | Score |
|---|---|
| Test Accuracy | **94.92%** |
| ROC-AUC | **98.94%** |
| Macro F1 | **95%** |
| Cross-validation | 93.46% ± 0.0025 |

> Cross-validation std of ±0.0025 confirms no overfitting.

---

## 🗄️ Database Schema

```
┌─────────────┐         ┌──────────────────────┐
│    users    │         │      complaints       │
├─────────────┤         ├──────────────────────┤
│ id (PK)     │────┐    │ id (PK)              │
│ google_id   │    └───▶│ user_id (FK)         │
│ email       │         │ complaint_id (unique) │
│ name        │         │ risk_level           │
│ picture     │         │ fraud_probability    │
│ is_admin    │         │ fraud_category       │
│ created_at  │         │ ncrp_code            │
│ last_login  │         │ phones_found (JSON)  │
└─────────────┘         │ urls_found (JSON)    │
                        │ amounts_found (JSON) │
┌─────────────┐         │ keywords (JSON)      │
│ velocity_db │         │ pdf_path             │
├─────────────┤         └──────────────────────┘
│ id (PK)     │
│ type        │         ┌──────────────────────┐
│ value       │         │   pending_review     │
│ count       │         ├──────────────────────┤
│ first_seen  │         │ complaint_id (FK)    │
│ last_seen   │         │ fraud_probability    │
│ complaint_  │         │ admin_label          │
│   ids (JSON)│         │ labeled_by           │
└─────────────┘         │ labeled_at           │
                        └──────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip
- Google Cloud Console account (for OAuth)

### Step 1 — Clone Repository
```bash
git clone https://github.com/ABHISHEK27Y/CyberShield
cd CyberShield
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Environment Setup
```bash
cp .env.template .env
```

Edit `.env`:
```env
SECRET_KEY=your-random-secret-key-here
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
ADMIN_EMAIL=your_email@gmail.com
DATABASE_URL=sqlite:///cybershield.db
```

### Step 4 — Google OAuth Setup
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create project → Enable Google+ API
3. Credentials → OAuth 2.0 → Web Application
4. Authorized redirect URI: `http://127.0.0.1:5000/login/google/authorized`
5. Copy Client ID and Secret to `.env`

### Step 5 — Initialize Database
```bash
flask db init
flask db migrate -m "initial"
flask db upgrade
```

### Step 6 — Run
```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Deployment (Render + Supabase)

### Why not SQLite in production?
Render uses an ephemeral filesystem — SQLite database is wiped on every restart. PostgreSQL (Supabase) persists forever.

### Step 1 — Supabase
1. Create account at [supabase.com](https://supabase.com)
2. New project → copy PostgreSQL connection string

### Step 2 — GitHub
```bash
git add .
git commit -m "ready for deployment"
git push origin main
```

### Step 3 — Render
1. [render.com](https://render.com) → New Web Service
2. Connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `flask db upgrade && gunicorn app:app`
5. Add environment variables:
   ```
   DATABASE_URL=postgresql://...   (Supabase connection string)
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   ADMIN_EMAIL=...
   SECRET_KEY=...
   ```

> **Note:** Model retraining (`pipefinal.py`) is only available in local deployment. Cloud deployment serves predictions only — this is intentional. Production ML systems separate training infrastructure from serving infrastructure.

---

## 📁 Project Structure

```
CyberShield/
│
├── app.py                  # Main Flask application
│                           # All routes, ML inference, PDF generator,
│                           # Velocity Tracker, Word Heatmap, HITL logic
│
├── models.py               # SQLAlchemy database models
│                           # User, Complaint, VelocityEntry, PendingReview
│
├── pipefinal.py            # ML training pipeline (FROZEN — do not modify)
│                           # Trains ensemble, saves pkl files
│
├── merge_datasets_v3.py    # Dataset merger (7 sources → master_dataset.csv)
│
├── verify_dataset.py       # Dataset sanity checker
│
├── final_models.pkl        # Trained model artifacts
│                           # Dict: {lr, nb, svm, weights}
│
├── final_vectorizer.pkl    # Trained TF-IDF vectorizer (frozen vocabulary)
│
├── requirements.txt        # All pip dependencies
├── render.yaml             # Render deployment configuration
├── .env.template           # Environment variable template
├── .gitignore              # Git ignore rules
│
├── templates/
│   ├── login.html          # Google OAuth login page
│   ├── index.html          # Home page — message input form
│   ├── result.html         # Analysis result — heatmap + velocity + entities
│   ├── admin.html          # Admin HITL panel + velocity dashboard
│   ├── ncrp_helper.html    # NCRP form copy-per-field assistant
│   └── history.html        # User's personal complaint history
│
└── complaints/             # Generated NCRP complaint PDFs (auto-created)
```

---

## 🔗 API Routes

| Method | Route | Description | Auth |
|---|---|---|---|
| GET | `/` | Home page with message input | Public |
| POST | `/analyze` | Run full ML pipeline | Public |
| GET | `/download-complaint/<id>` | Download NCRP PDF | Public |
| GET | `/ncrp-helper/<id>` | NCRP form helper page | Public |
| GET | `/history` | User complaint history | Login required |
| GET | `/login` | Google OAuth login | Public |
| GET | `/logout` | Logout | Login required |
| GET | `/admin` | Admin HITL dashboard | Admin only |
| POST | `/admin/label` | Label UNCERTAIN case | Admin only |
| POST | `/admin/retrain` | Trigger model retraining | Admin only |
| GET | `/admin/retrain-status` | Retrain log (JSON) | Admin only |
| GET | `/admin/users` | User management table | Admin only |
| GET | `/admin/velocity-db` | Full threat intelligence DB (JSON) | Admin only |

---

## ⚠️ Known Limitations

1. **MONEY_AMOUNT false positives** — Legitimate bank SMS ("salary credited Rs.50,000") score ~0.67 fraud probability because the MONEY_AMOUNT token is a strong fraud signal in training data
2. **English only** — TF-IDF vectorizer trained on English tokens. Hindi/Hinglish messages may not classify correctly
3. **In-memory complaint store** — Without database, server restart loses session data (fixed with SQLAlchemy in v4)
4. **Retraining on cloud** — Model retraining requires persistent filesystem, disabled on Render deployment
5. **Old English email spam** — Some Enron-era emails in the legit class passed the SMS length filter

---

## 🎯 Unique Contributions

This project makes contributions not found in any existing academic fraud detection project:

1. **NCRP Legal Taxonomy Mapping** — Only system connecting ML output to India's cybercrime legal categories. Search `"NCRP" + "machine learning" + "complaint"` on Google Scholar — zero results.

2. **Fraud Velocity Tracker** — Cross-complaint repeat offender detection with persistent threat intelligence database. No student project implements this.

3. **Word Risk Heatmap** — Token-level visual explainability for non-technical end users. No fraud detection web portal implements this.

4. **Human-in-the-Loop Retraining** — UNCERTAIN predictions reviewed by admin, fed back to training data, model retrained within same portal.

5. **India-Specific Dataset** — 26,531 rows from 7 sources. 5× larger than standard UCI SMS dataset used by 90% of similar projects.

---

## 👨‍💻 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.13 |
| Web Framework | Flask 3.0 |
| ML Library | scikit-learn |
| ORM | SQLAlchemy + Flask-Migrate |
| Auth | Google OAuth 2.0 (Flask-Dance) |
| PDF Generation | ReportLab Platypus |
| Database (local) | SQLite |
| Database (production) | PostgreSQL (Supabase) |
| Deployment | Render |
| Frontend | HTML/CSS/JS + Jinja2 |
| Fonts | DM Serif Display + DM Sans |

---

## 📜 License

MIT License — For educational purposes.

---

## 👨‍🎓 Academic Details

```
Project Title:  CyberShield — AI-Powered Fraud Detection &
                Cybercrime Complaint Intelligence Portal
Course:         Mini-Project (6th Semester)
Branch:         Computer Science & Engineering
Developer:      Abhishek Yadav
GitHub:         github.com/ABHISHEK27Y
```

---

<div align="center">

Made with ❤️ for safer digital India

**National Cybercrime Helpline: 1930 | cybercrime.gov.in**

</div>
