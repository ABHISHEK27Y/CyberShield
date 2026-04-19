"""
Fraud Detection & Cybercrime Complaint Portal  v3.0
====================================================
NEW in v3:
  1. Fraud Velocity Tracker  — cross-complaint repeat offender detection
     Tracks phone numbers & URLs across ALL submissions. Flags repeat offenders
     with a velocity score and threat level (LOW / MEDIUM / HIGH / CRITICAL).

  2. Word Risk Heatmap       — token-level visual explainability
     Highlights every word in the original message with a colour based on its
     LR fraud coefficient weight: RED (high fraud signal) → YELLOW (medium) →
     GREEN (safe). Non-technical victims see exactly what triggered the alert.

Previous features:
  - Google OAuth login
  - Admin Human-in-the-Loop review + retrain
  - NCRP Form Helper
  - Redesigned PDF

Install:
    pip install flask flask-dance flask-login reportlab
"""

import os, re, uuid, pickle, datetime, csv, subprocess, json
from pathlib import Path
from functools import wraps
from collections import defaultdict

import numpy as np
from flask import (Flask, request, render_template, send_file,
                   abort, redirect, url_for, flash, jsonify)
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, current_user)
from flask_dance.contrib.google import make_google_blueprint, google

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "final_models.pkl"
VEC_PATH    = BASE_DIR / "final_vectorizer.pkl"
COMPLAINTS  = BASE_DIR / "complaints"
DATASET     = BASE_DIR / "master_dataset.csv"
REVIEW_CSV  = BASE_DIR / "pending_review.csv"
VELOCITY_DB = BASE_DIR / "velocity_db.json"
COMPLAINTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# FLASK + AUTH
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
google_bp = make_google_blueprint(
    client_id     = os.environ.get("GOOGLE_CLIENT_ID",     "YOUR_GOOGLE_CLIENT_ID"),
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET"),
    scope         = ["openid",
                     "https://www.googleapis.com/auth/userinfo.email",
                     "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to   = "google_login_callback",
)
app.register_blueprint(google_bp, url_prefix="/login")

login_manager = LoginManager(app)
login_manager.login_view = "login_page"

ADMIN_EMAILS = { os.environ.get("ADMIN_EMAIL", "admin@example.com") }
_users: dict = {}

class User(UserMixin):
    def __init__(self, uid, name, email, picture=""):
        self.id = uid; self.name = name
        self.email = email; self.picture = picture
        self.is_admin = email in ADMIN_EMAILS

@login_manager.user_loader
def load_user(uid): return _users.get(uid)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with open(VEC_PATH,   "rb") as f: vectorizer = pickle.load(f)
with open(MODEL_PATH, "rb") as f: model_data = pickle.load(f)
lr      = model_data["lr"]
nb      = model_data["nb"]
svm     = model_data["svm"]
weights = model_data["weights"]

_complaint_store: dict = {}
_pending_review:  list = []

# ─────────────────────────────────────────────
# PREPROCESSING & PREDICTION
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+",  " SUSPICIOUS_URL ",   text)
    text = re.sub(r"\b[6-9]\d{9}\b",    " PHONE_NUMBER ",     text)
    text = re.sub(r"rs\.?\s?\d[\d,]*",  " MONEY_AMOUNT ",     text, flags=re.IGNORECASE)
    text = re.sub(r"\$\s?\d[\d,]*",     " MONEY_AMOUNT_USD ", text)
    text = re.sub(r"\b\d{4,6}\b",       " OTP_NUMBER ",       text)
    return re.sub(r"\s+", " ", text).strip()

def is_conversational(text: str) -> bool:
    words = text.split()
    if len(words) > 20: return False
    return bool(set(words) & {"hi","hello","hey","thanks","thank","ok","okay",
                               "yes","no","bye","good","morning","evening","night"})

def predict(raw_message: str) -> dict:
    processed  = preprocess(raw_message)
    vec        = vectorizer.transform([processed])
    p_lr       = lr.predict_proba(vec)[0][1]
    p_nb       = nb.predict_proba(vec)[0][1]
    p_svm      = svm.predict_proba(vec)[0][1]
    w_svm, w_lr, w_nb = weights
    fraud_prob = w_svm*p_svm + w_lr*p_lr + w_nb*p_nb
    if is_conversational(raw_message) and fraud_prob < 0.35:
        fraud_prob = 0.05
    legit_prob = 1.0 - fraud_prob
    if   fraud_prob >= 0.85: risk, pred, thresh = "FRAUD",      1, 0.85
    elif fraud_prob >= 0.45: risk, pred, thresh = "SUSPICIOUS",  1, 0.45
    elif fraud_prob >= 0.30: risk, pred, thresh = "UNCERTAIN",   0, 0.30
    else:                    risk, pred, thresh = "LEGIT",       0, 0.30
    return {"prediction": pred,
            "fraud_probability": round(float(fraud_prob), 4),
            "legit_probability": round(float(legit_prob), 4),
            "risk_level": risk, "threshold_used": thresh,
            "processed_text": processed}

# ─────────────────────────────────────────────
# FRAUD TYPE CLASSIFIER
# ─────────────────────────────────────────────
FRAUD_CATEGORIES = {
    "OTP / Account Takeover":     "CAT-01",
    "KYC / Verification Scam":    "CAT-02",
    "Lottery / Prize Fraud":      "CAT-03",
    "Loan / Investment Fraud":    "CAT-04",
    "Phishing / URL-based Fraud": "CAT-05",
    "Job / Work-from-Home Scam":  "CAT-06",
    "Government Scheme Fraud":    "CAT-07",
    "Unknown Fraud":              "CAT-08",
}
_FRAUD_RULES = [
    ("OTP / Account Takeover",
     r"\botp_number\b|\botp\b|\bone.?time.?pass|\baccount.{0,20}(block|suspend|verif|lock)"
     r"|\b(debit|credit).card\b|\bpin\b|\bpassword\b"),
    ("KYC / Verification Scam",
     r"\bkyc\b|\baadhaar\b|\bpan.card\b|\bverif(y|ication)\b"
     r"|\bupdate.{0,20}(document|detail|account)|\bbank.{0,20}(detail|account)|re.?kyc"),
    ("Lottery / Prize Fraud",
     r"\blotter(y|ies)\b|\bprize\b|\bwon\b|\bwinner\b|\bcongratulation|\bcash.?prize"),
    ("Loan / Investment Fraud",
     r"\bloan\b|\bemi\b|\binvest(ment)?\b|\bprofit\b|\bstock\b|\btrading\b|\bcrypto|\bbitcoin"),
    ("Phishing / URL-based Fraud",
     r"suspicious_url|\bclick.{0,20}(link|here|below)|\bdownload\b|\binstall\b"),
    ("Job / Work-from-Home Scam",
     r"\bjob\b|\bwork.from.home\b|\bpart.time\b|\bearning\b|\brecruit|\bhiring\b"),
    ("Government Scheme Fraud",
     r"\byojana\b|\bpm.awas\b|\bgovernment.scheme|\bsubsid(y|ies)\b|\bnrega\b"),
]

def classify_fraud_type(processed_text: str) -> dict:
    for category, pattern in _FRAUD_RULES:
        if re.search(pattern, processed_text, re.IGNORECASE):
            return {"category": category, "ncrp_code": FRAUD_CATEGORIES[category]}
    return {"category": "Unknown Fraud", "ncrp_code": "CAT-08"}

# ─────────────────────────────────────────────
# ENTITY EXTRACTOR
# ─────────────────────────────────────────────
def extract_entities(raw_message: str, sender: str = "") -> dict:
    phones  = list(set(re.findall(r"\b[6-9]\d{9}\b", raw_message)))
    urls    = list(set(re.findall(
        r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9\-]+\.[a-z]{2,4}/[^\s]*", raw_message)))
    amounts = list(set(re.findall(
        r"(?:Rs\.?|₹|\$)\s?\d[\d,]*(?:\.\d{1,2})?", raw_message, re.IGNORECASE)))
    otps    = list(set(re.findall(r"\b\d{4,6}\b", raw_message)))
    return {"phones": phones, "urls": urls, "amounts": amounts,
            "otps": otps, "sender": sender.strip(),
            "keywords": _top_trigger_keywords(raw_message)}

def _top_trigger_keywords(raw_message: str, n: int = 5) -> list:
    try:
        vec           = vectorizer.transform([preprocess(raw_message)])
        feature_names = vectorizer.get_feature_names_out()
        coefs         = lr.coef_[0]
        _, cols       = vec.nonzero()
        SKIP = {"word__suspicious_url","word__phone_number",
                 "word__otp_number","word__money_amount","word__money_amount_usd"}
        scored = sorted([(feature_names[c], coefs[c]) for c in cols
                          if coefs[c] > 0 and not feature_names[c].startswith("char__")
                          and feature_names[c] not in SKIP],
                        key=lambda x: x[1], reverse=True)
        return [w.replace("word__","") for w,_ in scored[:n]]
    except Exception:
        return []

# ═══════════════════════════════════════════════════════
# NOVEL FEATURE 1 — FRAUD VELOCITY TRACKER
# ═══════════════════════════════════════════════════════
"""
Tracks every phone number and URL reported across all fraud complaints.
Builds a persistent JSON database (velocity_db.json) that counts:
  - How many unique complaints mentioned each indicator
  - First seen / last seen timestamps
  - Which complaint IDs reported it

Threat levels:
  1 report  → LOW      (new, unconfirmed)
  2 reports → MEDIUM   (seen before, suspicious)
  3 reports → HIGH     (repeat offender)
  4+ reports→ CRITICAL (confirmed fraud infrastructure)

This is how telecom companies build fraud blacklists from crowd-sourced data.
No existing student project implements cross-complaint pattern matching.
"""

def _load_velocity_db() -> dict:
    """Load persistent velocity database from disk."""
    if VELOCITY_DB.exists():
        try:
            return json.loads(VELOCITY_DB.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"phones": {}, "urls": {}}

def _save_velocity_db(db: dict):
    """Persist velocity database to disk."""
    VELOCITY_DB.write_text(json.dumps(db, indent=2), encoding="utf-8")

def update_velocity(complaint_id: str, phones: list, urls: list):
    """
    Called after every FRAUD/SUSPICIOUS detection.
    Updates counts for each phone and URL found in the message.
    """
    db  = _load_velocity_db()
    now = datetime.datetime.now().isoformat()

    for phone in phones:
        if phone not in db["phones"]:
            db["phones"][phone] = {
                "count": 0, "complaint_ids": [],
                "first_seen": now, "last_seen": now
            }
        entry = db["phones"][phone]
        if complaint_id not in entry["complaint_ids"]:
            entry["count"]          += 1
            entry["complaint_ids"].append(complaint_id)
            entry["last_seen"]       = now

    for url in urls:
        # Normalise URL key — strip trailing slashes
        key = url.rstrip("/").lower()
        if key not in db["urls"]:
            db["urls"][key] = {
                "count": 0, "complaint_ids": [],
                "first_seen": now, "last_seen": now,
                "original": url
            }
        entry = db["urls"][key]
        if complaint_id not in entry["complaint_ids"]:
            entry["count"]          += 1
            entry["complaint_ids"].append(complaint_id)
            entry["last_seen"]       = now

    _save_velocity_db(db)

def get_velocity_alerts(phones: list, urls: list) -> list:
    """
    Returns list of velocity alert dicts for any known repeat indicators.
    Each alert: {type, value, count, threat_level, first_seen, last_seen}
    """
    db     = _load_velocity_db()
    alerts = []

    def threat_level(count: int) -> str:
        if count >= 4: return "CRITICAL"
        if count >= 3: return "HIGH"
        if count >= 2: return "MEDIUM"
        return "LOW"

    for phone in phones:
        if phone in db["phones"]:
            e = db["phones"][phone]
            if e["count"] >= 1:
                alerts.append({
                    "type":        "Phone Number",
                    "value":       phone,
                    "count":       e["count"],
                    "threat_level": threat_level(e["count"]),
                    "first_seen":  e["first_seen"][:10],
                    "last_seen":   e["last_seen"][:10],
                })

    for url in urls:
        key = url.rstrip("/").lower()
        if key in db["urls"]:
            e = db["urls"][key]
            if e["count"] >= 1:
                alerts.append({
                    "type":        "URL / Link",
                    "value":       url,
                    "count":       e["count"],
                    "threat_level": threat_level(e["count"]),
                    "first_seen":  e["first_seen"][:10],
                    "last_seen":   e["last_seen"][:10],
                })

    # Sort: CRITICAL first
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda x: order.get(x["threat_level"], 9))
    return alerts

def get_velocity_stats() -> dict:
    """Summary stats for admin dashboard."""
    db = _load_velocity_db()
    all_indicators = list(db["phones"].values()) + list(db["urls"].values())
    return {
        "total_phones":    len(db["phones"]),
        "total_urls":      len(db["urls"]),
        "critical_count":  sum(1 for e in all_indicators if e["count"] >= 4),
        "high_count":      sum(1 for e in all_indicators if e["count"] == 3),
        "top_phones":      sorted(db["phones"].items(),
                                   key=lambda x: x[1]["count"], reverse=True)[:5],
        "top_urls":        sorted(db["urls"].items(),
                                   key=lambda x: x[1]["count"], reverse=True)[:5],
    }

# ═══════════════════════════════════════════════════════
# NOVEL FEATURE 2 — WORD RISK HEATMAP
# ═══════════════════════════════════════════════════════
"""
Generates token-level risk annotations for every word in the original message.
Uses Logistic Regression coefficient weights to score each token.

Output: list of {"word": str, "score": float, "level": str}
  level = "high"   (score > 0.5)  → rendered RED   in UI
  level = "medium" (score > 0.1)  → rendered AMBER
  level = "low"    (score > 0.0)  → rendered YELLOW
  level = "safe"   (score <= 0.0) → rendered normal

The frontend renders this as inline coloured spans — no images needed.
This is genuine token-level explainability accessible to non-technical users.
"""

def generate_word_heatmap(raw_message: str) -> list:
    """
    Returns list of word-level risk dicts for heatmap rendering.
    Preserves original word order and punctuation grouping.
    """
    try:
        processed     = preprocess(raw_message)
        feature_names = vectorizer.get_feature_names_out()
        coefs         = lr.coef_[0]

        # Build a lookup: feature_name → coefficient
        coef_lookup = {feature_names[i]: coefs[i] for i in range(len(coefs))}

        # Tokenise the ORIGINAL message (split on whitespace, keep punctuation)
        tokens = raw_message.split()
        result = []

        for token in tokens:
            # Clean version for lookup (lowercase, strip punctuation for matching)
            clean = re.sub(r"[^\w]", "", token.lower())

            # Try to find this word's coefficient in the vectorizer vocabulary
            word_key  = f"word__{clean}"
            score     = coef_lookup.get(word_key, 0.0)

            # Also check if the token IS a replacement token
            proc_token = preprocess(token)
            for rt in ["suspicious_url","phone_number","money_amount","money_amount_usd","otp_number"]:
                if rt in proc_token:
                    rt_key   = f"word__{rt}"
                    rt_score = coef_lookup.get(rt_key, 0.0)
                    if rt_score > score:
                        score = rt_score

            # Assign level
            if   score >= 0.5:  level = "high"
            elif score >= 0.15: level = "medium"
            elif score >= 0.01: level = "low"
            else:               level = "safe"

            result.append({
                "word":  token,
                "score": round(float(score), 4),
                "level": level,
            })

        return result

    except Exception:
        # Fallback: return all words as safe
        return [{"word": w, "score": 0.0, "level": "safe"}
                for w in raw_message.split()]

# ─────────────────────────────────────────────
# HUMAN-IN-THE-LOOP
# ─────────────────────────────────────────────
def save_pending_review(complaint_id, message, fraud_prob):
    row = {"id": complaint_id,
           "timestamp": datetime.datetime.now().isoformat(),
           "message": message, "fraud_probability": fraud_prob,
           "admin_label": "", "labeled_by": "", "labeled_at": ""}
    _pending_review.append(row)
    write_header = not REVIEW_CSV.exists()
    with open(REVIEW_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

def append_to_dataset(message, label):
    write_header = not DATASET.exists()
    with open(DATASET, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["text","label"])
        w.writerow([message, label])

# ─────────────────────────────────────────────
# PDF GENERATOR v3 — with velocity alerts
# ─────────────────────────────────────────────
def generate_complaint_pdf(data: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.8*cm)

    NAVY    = colors.HexColor("#0A1628")
    GOLD    = colors.HexColor("#C8960C")
    SAFFRON = colors.HexColor("#E8650A")
    LGRAY   = colors.HexColor("#F5F6FA")
    MGRAY   = colors.HexColor("#CBD0DC")
    DGRAY   = colors.HexColor("#3D4460")
    RED     = colors.HexColor("#C0392B")
    GREEN   = colors.HexColor("#1A7A35")
    WHITE   = colors.white
    PW      = 17.4*cm

    S = getSampleStyleSheet()
    def ST(name, **kw): return ParagraphStyle(name, parent=S["Normal"], **kw)
    LBL  = ST("lb", fontSize=8,   fontName="Helvetica-Bold",  textColor=DGRAY, leading=11)
    VAL  = ST("vl", fontSize=8.5, fontName="Helvetica",       textColor=NAVY,  leading=12)
    BODY = ST("bd", fontSize=8.5, fontName="Helvetica",       leading=14, alignment=TA_JUSTIFY)
    TINY = ST("ti", fontSize=7,   fontName="Helvetica",       textColor=DGRAY, leading=9, alignment=TA_CENTER)
    WARN = ST("wn", fontSize=7.5, fontName="Helvetica-Oblique", textColor=DGRAY, leading=10)

    story = []

    # Banner
    ban = Table([[
        Paragraph("⚖", ST("ic",fontSize=22,alignment=TA_CENTER,textColor=WHITE)),
        Paragraph("भारत सरकार · Government of India<br/>"
                  "<b>NATIONAL CYBERCRIME REPORTING PORTAL (NCRP)</b>",
                  ST("mt",fontSize=12,fontName="Helvetica-Bold",textColor=WHITE,
                     alignment=TA_CENTER,leading=16)),
        Paragraph("Ministry of<br/>Home Affairs",
                  ST("mh",fontSize=7.5,textColor=colors.HexColor("#AABCDC"),
                     alignment=TA_CENTER,leading=10)),
    ]], colWidths=[2*cm,12*cm,3.4*cm])
    ban.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(ban)
    gs = Table([[""]], colWidths=[PW], rowHeights=[3])
    gs.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GOLD)]))
    story.append(gs)
    sb = Table([[Paragraph(
        "CYBERCRIME COMPLAINT · AUTO-GENERATED BY ML FRAUD DETECTION SYSTEM · cybercrime.gov.in",
        ST("h2",fontSize=7.5,textColor=colors.HexColor("#FFD580"),
           fontName="Helvetica",alignment=TA_CENTER,leading=10)
    )]], colWidths=[PW])
    sb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),SAFFRON),
                             ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(sb)
    story.append(Spacer(1,0.25*cm))

    risk    = data.get("prediction_result",{})
    rl      = risk.get("risk_level","—")
    rl_color= {"FRAUD":RED,"SUSPICIOUS":SAFFRON,"UNCERTAIN":GOLD,"LEGIT":GREEN}.get(rl,NAVY)
    ref_no  = data.get("complaint_id","N/A")
    ts      = data.get("timestamp","")

    rt = Table([[
        Paragraph(f"Complaint Ref: <b>{ref_no}</b>",LBL),
        Paragraph(f"Date/Time: <b>{ts}</b>",LBL),
        Paragraph(f"Risk: <b>{rl}</b>",
                  ST("rls",fontSize=9,fontName="Helvetica-Bold",
                     textColor=rl_color,alignment=TA_RIGHT)),
    ]], colWidths=[5*cm,8*cm,4.4*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LGRAY),("BOX",(0,0),(-1,-1),0.5,MGRAY),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(rt)
    story.append(Spacer(1,0.2*cm))

    def sec(title):
        t = Table([[Paragraph(f"  {title}",
                    ST("sh",fontSize=8.5,fontName="Helvetica-Bold",
                       textColor=WHITE,leading=11))]], colWidths=[PW], rowHeights=[18])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
                                ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.append(t)

    def kv_table(rows):
        t = Table([[Paragraph(l,LBL),Paragraph(str(v) if v else "—",VAL)] for l,v in rows],
                  colWidths=[4.5*cm,12.9*cm])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LGRAY]),
            ("GRID",(0,0),(-1,-1),0.3,MGRAY),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1,0.15*cm))

    def bul(items): return "   |   ".join(items) if items else "None identified"

    entities = data.get("entities",{})
    ft       = data.get("fraud_type",{})
    cn       = data.get("complainant_name","Not provided")
    cc       = data.get("complainant_contact","Not provided")

    sec("SECTION 1 — COMPLAINANT DETAILS")
    kv_table([("Full Name:",cn),("Contact / Mobile:",cc),
              ("Portal:","cybercrime.gov.in"),
              ("Jurisdiction:","As per residential address of complainant")])

    sec("SECTION 2 — INCIDENT DETAILS")
    kv_table([("Incident Date:",ts.split(" ")[0]),
              ("Fraud Category:",ft.get("category","Unknown Fraud")),
              ("NCRP Category Code:",ft.get("ncrp_code","CAT-08")),
              ("Sender (if known):",entities.get("sender") or "Unknown"),
              ("Mode:","SMS / WhatsApp / Email")])

    sec("SECTION 3 — FRAUDULENT MESSAGE (verbatim)")
    safe_msg = data.get("message","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    mt = Table([[Paragraph(safe_msg,BODY)]], colWidths=[PW])
    mt.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),1.5,SAFFRON),
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#FFFBF5")),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story.append(mt)
    story.append(Spacer(1,0.15*cm))

    sec("SECTION 4 — EXTRACTED DIGITAL EVIDENCE")
    kv_table([("Phone Numbers:",bul(entities.get("phones",[]))),
              ("URLs / Links:",bul(entities.get("urls",[]))),
              ("Monetary Amounts:",bul(entities.get("amounts",[]))),
              ("OTP / PIN Codes:",bul(entities.get("otps",[])))])

    # ── VELOCITY ALERTS in PDF ──
    velocity_alerts = data.get("velocity_alerts", [])
    if velocity_alerts:
        sec("SECTION 4B — REPEAT OFFENDER INTELLIGENCE (Velocity Tracker)")
        va_rows = [("Repeat Indicators Found:", str(len(velocity_alerts)))]
        for a in velocity_alerts:
            va_rows.append((
                f"{a['type']} [{a['threat_level']}]:",
                f"{a['value']}  |  Reported {a['count']} time(s)  "
                f"|  First seen: {a['first_seen']}  |  Last seen: {a['last_seen']}"
            ))
        kv_table(va_rows)

    sec("SECTION 5 — ML MODEL EVIDENCE SUMMARY")
    kv_table([
        ("ML Risk Level:",rl),
        ("Fraud Probability:",f"{risk.get('fraud_probability',0)*100:.1f}%"),
        ("Legit Probability:",f"{risk.get('legit_probability',0)*100:.1f}%"),
        ("Decision Threshold:",str(risk.get("threshold_used","—"))),
        ("Top Trigger Keywords:",", ".join(entities.get("keywords",[])) or "—"),
        ("Model Architecture:","Ensemble — LinearSVC 60% + LR 20% + MultinomialNB 20%"),
        ("Vectorisation:","TF-IDF Word n-gram (1,2) + Char n-gram (3,5)"),
        ("Training Dataset:","26,531 SMS/Email | Accuracy 94.92% | ROC-AUC 98.94%"),
    ])
    story.append(Paragraph(
        "<b>Disclaimer:</b> Auto-generated by ML system. Law enforcement should independently verify.",
        WARN))

    story.append(PageBreak())
    sec("SECTION 6 — DECLARATION BY COMPLAINANT")
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph(
        f"I, <b>{cn}</b>, hereby declare that the information provided in this complaint "
        "is true and correct to the best of my knowledge and belief. I have received the above "
        "mentioned fraudulent/suspicious communication and wish to report the same to the "
        "appropriate cybercrime authority under the Information Technology Act, 2000 and the "
        "Indian Penal Code.", BODY))
    story.append(Spacer(1,2*cm))
    sig = Table([
        [Paragraph("_____________________________",VAL),Paragraph("_____________________________",VAL)],
        [Paragraph("Signature of Complainant",TINY),Paragraph("Date",TINY)],
    ], colWidths=[8.7*cm,8.7*cm])
    sig.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4)]))
    story.append(sig)
    story.append(Spacer(1,1*cm))
    fg = Table([[""]], colWidths=[PW], rowHeights=[2])
    fg.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GOLD)]))
    story.append(fg)
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph(
        "Submit at your nearest Cybercrime Police Station or cybercrime.gov.in  ·  "
        "National Helpline: <b>1930</b>  ·  To file online: Select category "
        f"'{ft.get('category','Unknown')}' ({ft.get('ncrp_code','CAT-08')})", TINY))

    doc.build(story)

# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/google-callback")
def google_login_callback():
    if not google.authorized:
        flash("Google login failed.", "error")
        return redirect(url_for("login_page"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Could not fetch Google profile.", "error")
        return redirect(url_for("login_page"))
    info = resp.json()
    uid  = info["id"]
    user = User(uid, info.get("name","User"), info.get("email",""), info.get("picture",""))
    _users[uid] = user
    login_user(user)
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login_page"))

# ─────────────────────────────────────────────
# MAIN ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", user=current_user)

@app.route("/analyze", methods=["POST"])
def analyze():
    raw_message         = request.form.get("message","").strip()
    sender              = request.form.get("sender","").strip()
    complainant_name    = request.form.get("complainant_name","").strip()
    complainant_contact = request.form.get("complainant_contact","").strip()

    if not raw_message:
        return render_template("index.html", error="Please enter a message.", user=current_user)

    # Stage 1 — predict
    result     = predict(raw_message)
    fraud_type = None
    if result["risk_level"] in ("FRAUD","SUSPICIOUS"):
        fraud_type = classify_fraud_type(result["processed_text"])
    elif result["risk_level"] == "UNCERTAIN":
        fraud_type = {"category": "Pending Admin Review", "ncrp_code": "TBD"}

    # Stage 2 — entities
    entities = extract_entities(raw_message, sender=sender)

    # ── NOVEL FEATURE 1: Velocity Tracker ──
    velocity_alerts = []
    if result["risk_level"] in ("FRAUD","SUSPICIOUS"):
        complaint_id_temp = str(uuid.uuid4())[:8].upper()
        update_velocity(complaint_id_temp, entities["phones"], entities["urls"])
        velocity_alerts = get_velocity_alerts(entities["phones"], entities["urls"])

    # ── NOVEL FEATURE 2: Word Heatmap ──
    word_heatmap = generate_word_heatmap(raw_message)

    complaint_id = str(uuid.uuid4())[:8].upper()
    timestamp    = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    _complaint_store[complaint_id] = {
        "complaint_id": complaint_id, "timestamp": timestamp,
        "message": raw_message, "complainant_name": complainant_name,
        "complainant_contact": complainant_contact,
        "prediction_result": result,
        "fraud_type": fraud_type or {"category":"N/A","ncrp_code":"N/A"},
        "entities": entities,
        "velocity_alerts": velocity_alerts,
    }

    if result["risk_level"] == "UNCERTAIN":
        save_pending_review(complaint_id, raw_message, result["fraud_probability"])

    return render_template("result.html",
        message=raw_message, result=result, fraud_type=fraud_type,
        entities=entities, complaint_id=complaint_id,
        show_pdf=result["risk_level"] in ("FRAUD","SUSPICIOUS"),
        timestamp=timestamp, user=current_user,
        velocity_alerts=velocity_alerts,
        word_heatmap=word_heatmap)

@app.route("/download-complaint/<complaint_id>")
def download_complaint(complaint_id):
    data = _complaint_store.get(complaint_id)
    if not data: abort(404)
    pdf_path = COMPLAINTS / f"complaint_{complaint_id}.pdf"
    if not pdf_path.exists():
        generate_complaint_pdf(data, pdf_path)
    return send_file(str(pdf_path), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"NCRP_Complaint_{complaint_id}.pdf")

@app.route("/ncrp-helper/<complaint_id>")
def ncrp_helper(complaint_id):
    data = _complaint_store.get(complaint_id)
    if not data: abort(404)
    return render_template("ncrp_helper.html", data=data, user=current_user)

# ─────────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────────
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated: return redirect(url_for("login_page"))
        if not current_user.is_admin: abort(403)
        return f(*args, **kwargs)
    return decorated

@app.route("/admin")
@require_admin
def admin_panel():
    vstats = get_velocity_stats()
    return render_template("admin.html",
                            pending=_pending_review,
                            user=current_user,
                            vstats=vstats)

@app.route("/admin/label", methods=["POST"])
@require_admin
def admin_label():
    cid       = request.form.get("complaint_id")
    label_str = request.form.get("label")
    if not cid or label_str not in ("fraud","legit"):
        flash("Invalid submission.", "error")
        return redirect(url_for("admin_panel"))
    label   = 1 if label_str == "fraud" else 0
    message = ""
    for item in _pending_review:
        if item["id"] == cid:
            item["admin_label"] = label_str
            item["labeled_by"]  = current_user.email
            item["labeled_at"]  = datetime.datetime.now().isoformat()
            message = item["message"]
            break
    if not message and cid in _complaint_store:
        message = _complaint_store[cid]["message"]
    if message:
        append_to_dataset(message, label)
        flash(f"✅ Labeled '{label_str}' and added to dataset. Hit Retrain to apply.", "success")
    else:
        flash("Message not found.", "error")
    return redirect(url_for("admin_panel"))

@app.route("/admin/retrain", methods=["POST"])
@require_admin
def admin_retrain():
    try:
        subprocess.Popen(
            ["python", str(BASE_DIR / "pipefinal.py")],
            cwd=str(BASE_DIR),
            stdout=open(BASE_DIR / "retrain.log","w"),
            stderr=subprocess.STDOUT,
        )
        flash("🔄 Retraining started. Check log for progress.", "success")
    except Exception as e:
        flash(f"❌ Retrain failed: {e}", "error")
    return redirect(url_for("admin_panel"))

@app.route("/admin/retrain-status")
@require_admin
def retrain_status():
    log_path = BASE_DIR / "retrain.log"
    log = log_path.read_text(encoding="utf-8",errors="replace")[-3000:] if log_path.exists() else "No log yet."
    return jsonify({"log": log})

@app.route("/admin/velocity-db")
@require_admin
def velocity_db_view():
    """JSON endpoint — full velocity database for admin inspection."""
    return jsonify(_load_velocity_db())

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Fraud Detection Portal v3.0")
    print("  Novel: Velocity Tracker + Word Heatmap")
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)