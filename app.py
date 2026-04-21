"""
CyberShield: AI-Powered Fraud Detection & Cybercrime Complaint Intelligence Portal  v4.0
====================================================
NEW in v4:
  - SQLAlchemy ORM (SQLite locally, PostgreSQL in production)
  - Flask-Migrate for schema management
  - Persistent Complaint, User, VelocityEntry, PendingReview tables
  - /history route — user complaint history
  - /admin/users route — user management table
  - Anonymous submissions still supported (user_id nullable)

Unchanged from v3:
  - preprocess() — FROZEN, never modify
  - vectorizer — never re-fit
  - pipefinal.py — never touched
  - All 5 existing templates
  - All feature logic (heatmap, velocity, HITL, PDF)
"""

import os, re, uuid, pickle, datetime, csv, subprocess, json
from pathlib import Path
from functools import wraps

import numpy as np
from flask import (Flask, request, render_template, send_file,
                   abort, redirect, url_for, flash, jsonify)
from flask_login import (LoginManager, login_user, logout_user, current_user)
from flask_dance.contrib.google import make_google_blueprint, google
from flask_migrate import Migrate
from sqlalchemy import func

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT

# ─────────────────────────────────────────────
# CONFIG & PATHS
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "final_models.pkl"
VEC_PATH   = BASE_DIR / "final_vectorizer.pkl"
COMPLAINTS = BASE_DIR / "complaints"
DATASET    = BASE_DIR / "master_dataset.csv"
COMPLAINTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# Database — always use absolute path so data persists across restarts
db_url = os.environ.get("DATABASE_URL", "")
if not db_url or db_url.strip() in ("", "sqlite:///cybershield.db"):
    # Force absolute path — prevents relative-path data loss on reload
    db_url = "sqlite:///" + str(BASE_DIR / "cybershield.db").replace("\\", "/")
# Render gives postgres:// — SQLAlchemy needs postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["REMEMBER_COOKIE_DURATION"] = datetime.timedelta(days=30)  # stay logged in 30 days
app.config["REMEMBER_COOKIE_SECURE"] = False   # set True in production (HTTPS)
app.config["REMEMBER_COOKIE_HTTPONLY"] = True

# Import models AFTER app config
from models import db, User, Complaint, VelocityEntry, PendingReview
db.init_app(app)
migrate = Migrate(app, db)

# ─────────────────────────────────────────────
# GOOGLE OAUTH
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# FLASK-LOGIN
# ─────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login_page"

ADMIN_EMAILS = {os.environ.get("ADMIN_EMAIL", "admin@example.com")}

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, uid)

# ─────────────────────────────────────────────
# LOAD ML MODELS (frozen — never re-fit)
# ─────────────────────────────────────────────
with open(VEC_PATH,   "rb") as f: vectorizer = pickle.load(f)
with open(MODEL_PATH, "rb") as f: model_data = pickle.load(f)
lr      = model_data["lr"]
nb      = model_data["nb"]
svm     = model_data["svm"]
weights = model_data["weights"]

# ─────────────────────────────────────────────
# PREPROCESSING — FROZEN: DO NOT MODIFY
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

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
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
# NOVEL FEATURE 1 — FRAUD VELOCITY TRACKER (DB-backed)
# ═══════════════════════════════════════════════════════
def _threat_level(count: int) -> str:
    if count >= 4: return "CRITICAL"
    if count >= 3: return "HIGH"
    if count >= 2: return "MEDIUM"
    return "LOW"

def update_velocity(complaint_id: str, phones: list, urls: list):
    """Upsert VelocityEntry rows for each phone/URL in this complaint."""
    now = datetime.datetime.utcnow()
    pairs = [("phone", p, p) for p in phones] + \
            [("url", u, u.rstrip("/").lower()) for u in urls]

    for itype, original, key in pairs:
        entry = VelocityEntry.query.filter_by(
            indicator_type=itype, indicator_value=key).first()
        if entry is None:
            entry = VelocityEntry(
                indicator_type  = itype,
                indicator_value = key,
                count           = 1,
                complaint_ids   = json.dumps([complaint_id]),
                first_seen      = now,
                last_seen       = now,
            )
            db.session.add(entry)
        else:
            ids = _safe_json(entry.complaint_ids, [])
            if complaint_id not in ids:
                ids.append(complaint_id)
                entry.count         = len(ids)
                entry.complaint_ids = json.dumps(ids)
                entry.last_seen     = now
    db.session.commit()

def get_velocity_alerts(phones: list, urls: list) -> list:
    """Return velocity alerts for any known repeat indicators."""
    alerts = []
    for phone in phones:
        entry = VelocityEntry.query.filter_by(
            indicator_type="phone", indicator_value=phone).first()
        if entry and entry.count >= 1:
            alerts.append({
                "type":         "Phone Number",
                "value":        phone,
                "count":        entry.count,
                "threat_level": _threat_level(entry.count),
                "first_seen":   entry.first_seen.strftime("%Y-%m-%d") if entry.first_seen else "",
                "last_seen":    entry.last_seen.strftime("%Y-%m-%d")  if entry.last_seen  else "",
            })
    for url in urls:
        key = url.rstrip("/").lower()
        entry = VelocityEntry.query.filter_by(
            indicator_type="url", indicator_value=key).first()
        if entry and entry.count >= 1:
            alerts.append({
                "type":         "URL / Link",
                "value":        url,
                "count":        entry.count,
                "threat_level": _threat_level(entry.count),
                "first_seen":   entry.first_seen.strftime("%Y-%m-%d") if entry.first_seen else "",
                "last_seen":    entry.last_seen.strftime("%Y-%m-%d")  if entry.last_seen  else "",
            })
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda x: order.get(x["threat_level"], 9))
    return alerts

def get_velocity_stats() -> dict:
    """Summary stats for admin dashboard."""
    phones = VelocityEntry.query.filter_by(indicator_type="phone").order_by(
        VelocityEntry.count.desc()).all()
    urls   = VelocityEntry.query.filter_by(indicator_type="url").order_by(
        VelocityEntry.count.desc()).all()
    all_e  = phones + urls
    return {
        "total_phones":   len(phones),
        "total_urls":     len(urls),
        "critical_count": sum(1 for e in all_e if e.count >= 4),
        "high_count":     sum(1 for e in all_e if e.count == 3),
        "top_phones": [(e.indicator_value, {
            "count": e.count,
            "last_seen": e.last_seen.strftime("%Y-%m-%d") if e.last_seen else "",
        }) for e in phones[:5]],
        "top_urls": [(e.indicator_value, {
            "count": e.count,
            "last_seen": e.last_seen.strftime("%Y-%m-%d") if e.last_seen else "",
        }) for e in urls[:5]],
    }

# ═══════════════════════════════════════════════════════
# NOVEL FEATURE 2 — WORD RISK HEATMAP (unchanged logic)
# ═══════════════════════════════════════════════════════
def generate_word_heatmap(raw_message: str) -> list:
    try:
        processed     = preprocess(raw_message)
        feature_names = vectorizer.get_feature_names_out()
        coefs         = lr.coef_[0]
        coef_lookup   = {feature_names[i]: coefs[i] for i in range(len(coefs))}
        tokens        = raw_message.split()
        result        = []
        for token in tokens:
            clean     = re.sub(r"[^\w]", "", token.lower())
            word_key  = f"word__{clean}"
            score     = coef_lookup.get(word_key, 0.0)
            proc_token = preprocess(token)
            for rt in ["suspicious_url","phone_number","money_amount",
                       "money_amount_usd","otp_number"]:
                if rt in proc_token:
                    rt_score = coef_lookup.get(f"word__{rt}", 0.0)
                    if rt_score > score:
                        score = rt_score
            if   score >= 0.5:  level = "high"
            elif score >= 0.15: level = "medium"
            elif score >= 0.01: level = "low"
            else:               level = "safe"
            result.append({"word": token, "score": round(float(score), 4), "level": level})
        return result
    except Exception:
        return [{"word": w, "score": 0.0, "level": "safe"} for w in raw_message.split()]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _safe_json(s, default):
    """Safely parse a JSON string; return default on error."""
    try:
        return json.loads(s) if s else default
    except Exception:
        return default

def _reconstruct_data(c: Complaint) -> dict:
    """Rebuild the data dict from a Complaint ORM object for PDF / NCRP helper."""
    fp = c.fraud_probability or 0.0
    ts = c.timestamp.strftime("%d-%m-%Y %H:%M:%S") if c.timestamp else ""
    thresh = 0.85 if c.risk_level == "FRAUD" else 0.45
    return {
        "complaint_id":        c.complaint_id,
        "timestamp":           ts,
        "message":             c.message,
        "complainant_name":    c.complainant_name    or "",
        "complainant_contact": c.complainant_contact or "",
        "prediction_result": {
            "risk_level":        c.risk_level or "UNKNOWN",
            "fraud_probability": fp,
            "legit_probability": round(1.0 - fp, 4),
            "threshold_used":    thresh,
        },
        "fraud_type": {
            "category":  c.fraud_category or "Unknown Fraud",
            "ncrp_code": c.ncrp_code      or "CAT-08",
        },
        "entities": {
            "phones":   _safe_json(c.phones_found,  []),
            "urls":     _safe_json(c.urls_found,    []),
            "amounts":  _safe_json(c.amounts_found, []),
            "otps":     _safe_json(c.otps_found,    []),
            "sender":   c.sender or "",
            "keywords": _safe_json(c.keywords,      []),
        },
        "velocity_alerts": _safe_json(c.velocity_alerts, []),
    }

# ─────────────────────────────────────────────
# HUMAN-IN-THE-LOOP
# ─────────────────────────────────────────────
def save_pending_review(complaint_id: str, message: str,
                        fraud_prob: float, user_id=None):
    pr = PendingReview(
        complaint_id      = complaint_id,
        user_id           = user_id,
        message           = message,
        fraud_probability = fraud_prob,
        admin_label       = "",
    )
    db.session.add(pr)
    db.session.commit()

def append_to_dataset(message: str, label: int):
    write_header = not DATASET.exists()
    with open(DATASET, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["text", "label"])
        w.writerow([message, label])

# ─────────────────────────────────────────────
# PDF GENERATOR v4
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

    # ── CyberShield Banner
    PURPLE = colors.HexColor("#5b21b6")
    ban = Table([[
        Paragraph("CYBER<br/>SHIELD",
                  ST("ic",fontSize=7,fontName="Helvetica-Bold",alignment=TA_CENTER,
                     textColor=colors.HexColor("#c4b5fd"),leading=9)),
        Paragraph("CyberShield<br/><b>AI-Powered Fraud Detection &amp; Complaint Intelligence Portal</b>",
                  ST("mt",fontSize=11,fontName="Helvetica-Bold",textColor=WHITE,
                     alignment=TA_CENTER,leading=16)),
        Paragraph("Powered by<br/>ML Pipeline",
                  ST("mh",fontSize=7.5,textColor=colors.HexColor("#c4b5fd"),
                     alignment=TA_CENTER,leading=10)),
    ]], colWidths=[2*cm,12*cm,3.4*cm])
    ban.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("BOX",(0,0),(-1,-1),0,NAVY),
        ("INNERGRID",(0,0),(-1,-1),0,NAVY),
    ]))
    story.append(ban)
    gs = Table([[""]],colWidths=[PW],rowHeights=[3])
    gs.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),PURPLE),
        ("BOX",(0,0),(-1,-1),0,PURPLE),
    ]))
    story.append(gs)
    sb = Table([[Paragraph(
        "FRAUD ANALYSIS REPORT  |  AUTO-GENERATED BY CYBERSHIELD ML SYSTEM  |  For reference only",
        ST("h2",fontSize=7.5,textColor=colors.HexColor("#FFD580"),
           fontName="Helvetica",alignment=TA_CENTER,leading=10)
    )]], colWidths=[PW])
    sb.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),SAFFRON),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("BOX",(0,0),(-1,-1),0,SAFFRON),
    ]))
    story.append(sb)
    story.append(Spacer(1,0.25*cm))

    risk     = data.get("prediction_result",{})
    rl       = risk.get("risk_level","—")
    rl_color = {"FRAUD":RED,"SUSPICIOUS":SAFFRON,"UNCERTAIN":GOLD,"LEGIT":GREEN}.get(rl,NAVY)
    ref_no   = data.get("complaint_id","N/A")
    ts       = data.get("timestamp","")

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
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),NAVY),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("BOX",(0,0),(-1,-1),0,NAVY),
        ]))
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
    kv_table([("Full Name:",cn),("Contact / Mobile:",cc)])

    sec("SECTION 2 — INCIDENT DETAILS")
    kv_table([("Incident Date:",ts.split(" ")[0]),
              ("Fraud Category:",ft.get("category","Unknown Fraud")),
              ("NCRP Category Code:",ft.get("ncrp_code","CAT-08")),
              ("Sender (if known):",entities.get("sender") or "Unknown"),
              ("Mode:","SMS / WhatsApp / Email")])

    sec("SECTION 3 — FRAUDULENT MESSAGE")
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
# ADMIN DECORATOR
# ─────────────────────────────────────────────
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login_page"))
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated

# ═══════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════
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

    user = db.session.get(User, uid)
    if not user:
        user = User(
            id        = uid,
            google_id = uid,
            email     = info.get("email", ""),
            name      = info.get("name", ""),
            picture   = info.get("picture", ""),
            is_admin  = info.get("email", "") in ADMIN_EMAILS,
        )
        db.session.add(user)
    else:
        user.last_login = datetime.datetime.utcnow()
        user.picture    = info.get("picture", user.picture)
        # Re-check admin on every login (in case ADMIN_EMAIL changed)
        user.is_admin   = info.get("email", "") in ADMIN_EMAILS

    db.session.commit()
    login_user(user, remember=True)   # persists across browser/server restarts
    # Admins go straight to the dashboard
    if user.is_admin:
        return redirect(url_for("admin_panel"))
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login_page"))

# ═══════════════════════════════════════════════════════
# MAIN ROUTES
# ═══════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html", user=current_user)

@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Auth guard: must be signed in to analyse ──
    if not current_user.is_authenticated:
        flash("Please sign in to analyse a message.", "error")
        return redirect(url_for("login_page"))

    raw_message         = request.form.get("message","").strip()
    sender              = request.form.get("sender","").strip()
    complainant_name    = request.form.get("complainant_name","").strip()
    complainant_contact = request.form.get("complainant_contact","").strip()

    if not raw_message:
        return render_template("index.html", error="Please enter a message.", user=current_user)

    # 1 — Predict
    result     = predict(raw_message)
    fraud_type = None
    if result["risk_level"] in ("FRAUD","SUSPICIOUS"):
        fraud_type = classify_fraud_type(result["processed_text"])
    elif result["risk_level"] == "UNCERTAIN":
        fraud_type = {"category": "Pending Admin Review", "ncrp_code": "TBD"}

    # 2 — Entities
    entities = extract_entities(raw_message, sender=sender)

    # 3 — Generate final complaint_id ONCE (used for velocity + store)
    complaint_id = str(uuid.uuid4())[:8].upper()
    timestamp    = datetime.datetime.now()

    # 4 — Velocity Tracker (DB-backed)
    velocity_alerts = []
    if result["risk_level"] in ("FRAUD","SUSPICIOUS"):
        update_velocity(complaint_id, entities["phones"], entities["urls"])
        velocity_alerts = get_velocity_alerts(entities["phones"], entities["urls"])

    # 5 — Word Heatmap
    word_heatmap = generate_word_heatmap(raw_message)

    # 6 — Save complaint to DB
    complaint = Complaint(
        complaint_id        = complaint_id,
        user_id             = current_user.id if current_user.is_authenticated else None,
        timestamp           = timestamp,
        message             = raw_message,
        risk_level          = result["risk_level"],
        fraud_probability   = result["fraud_probability"],
        fraud_category      = fraud_type["category"] if fraud_type else None,
        ncrp_code           = fraud_type["ncrp_code"] if fraud_type else None,
        phones_found        = json.dumps(entities["phones"]),
        urls_found          = json.dumps(entities["urls"]),
        amounts_found       = json.dumps(entities["amounts"]),
        otps_found          = json.dumps(entities["otps"]),
        keywords            = json.dumps(entities["keywords"]),
        complainant_name    = complainant_name,
        complainant_contact = complainant_contact,
        sender              = sender,
        velocity_alerts     = json.dumps(velocity_alerts),
    )
    db.session.add(complaint)
    db.session.commit()

    # 7 — Save to pending review if UNCERTAIN
    if result["risk_level"] == "UNCERTAIN":
        save_pending_review(
            complaint_id, raw_message, result["fraud_probability"],
            current_user.id if current_user.is_authenticated else None
        )

    ts_str = timestamp.strftime("%d-%m-%Y %H:%M:%S")
    return render_template("result.html",
        message=raw_message, result=result, fraud_type=fraud_type,
        entities=entities, complaint_id=complaint_id,
        show_pdf=result["risk_level"] in ("FRAUD","SUSPICIOUS"),
        timestamp=ts_str, user=current_user,
        velocity_alerts=velocity_alerts,
        word_heatmap=word_heatmap)

@app.route("/download-complaint/<complaint_id>")
def download_complaint(complaint_id):
    c = Complaint.query.filter_by(complaint_id=complaint_id).first_or_404()
    pdf_path = COMPLAINTS / f"complaint_{complaint_id}.pdf"
    if not pdf_path.exists():
        generate_complaint_pdf(_reconstruct_data(c), pdf_path)
        # Save path back to DB
        c.pdf_path = str(pdf_path.relative_to(BASE_DIR))
        db.session.commit()
    return send_file(str(pdf_path), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"NCRP_Complaint_{complaint_id}.pdf")

@app.route("/ncrp-helper/<complaint_id>")
def ncrp_helper(complaint_id):
    c = Complaint.query.filter_by(complaint_id=complaint_id).first_or_404()
    return render_template("ncrp_helper.html", data=_reconstruct_data(c), user=current_user)

@app.route("/model-details")
def model_details():
    return render_template("model_details.html", user=current_user)

# ─────────────────────────────────────────────
# HISTORY ROUTE (NEW v4)
# ─────────────────────────────────────────────
@app.route("/history")
def complaint_history():
    if not current_user.is_authenticated:
        flash("Please sign in to view your complaint history.", "error")
        return redirect(url_for("login_page"))
    page       = request.args.get("page", 1, type=int)
    pagination = (Complaint.query
                  .filter_by(user_id=current_user.id)
                  .order_by(Complaint.timestamp.desc())
                  .paginate(page=page, per_page=20, error_out=False))
    complaints = pagination.items

    # Per-risk counts
    all_c  = Complaint.query.filter_by(user_id=current_user.id).all()
    counts = {"FRAUD":0,"SUSPICIOUS":0,"UNCERTAIN":0,"LEGIT":0}
    for c in all_c:
        if c.risk_level in counts:
            counts[c.risk_level] += 1

    return render_template("history.html",
        user=current_user,
        complaints=complaints,
        pagination=pagination,
        counts=counts,
        total=len(all_c))

# ═══════════════════════════════════════════════════════
# ADMIN ROUTES
# ═══════════════════════════════════════════════════════
@app.route("/admin")
@require_admin
def admin_panel():
    # Load unlabeled pending reviews
    pending_rows = (PendingReview.query
                    .filter_by(admin_label="")
                    .order_by(PendingReview.created_at.desc())
                    .all())
    pending = [{
        "id":               p.complaint_id,
        "timestamp":        p.created_at.strftime("%d-%m-%Y %H:%M") if p.created_at else "",
        "message":          p.message,
        "fraud_probability":p.fraud_probability,
        "admin_label":      p.admin_label or "",
        "labeled_by":       p.labeled_by  or "",
        "labeled_at":       p.labeled_at.strftime("%d-%m-%Y %H:%M") if p.labeled_at else "",
    } for p in pending_rows]

    vstats = get_velocity_stats()

    # Admin stats
    total_uncertain = PendingReview.query.count()
    total_labeled   = PendingReview.query.filter(PendingReview.admin_label != "").count()

    return render_template("admin.html",
        pending=pending,
        user=current_user,
        vstats=vstats,
        total_uncertain=total_uncertain,
        awaiting_review=len(pending),
        total_labeled=total_labeled)

@app.route("/admin/label", methods=["POST"])
@require_admin
def admin_label():
    cid       = request.form.get("complaint_id")
    label_str = request.form.get("label")
    if not cid or label_str not in ("fraud","legit"):
        flash("Invalid submission.", "error")
        return redirect(url_for("admin_panel"))

    label   = 1 if label_str == "fraud" else 0
    pending = PendingReview.query.filter_by(complaint_id=cid).first()
    message = ""
    if pending:
        pending.admin_label = label_str
        pending.labeled_by  = current_user.email
        pending.labeled_at  = datetime.datetime.utcnow()
        message             = pending.message
        db.session.commit()
    # Fallback — look in complaints table
    if not message:
        c = Complaint.query.filter_by(complaint_id=cid).first()
        if c:
            message = c.message
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
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"]       = "1"
        subprocess.Popen(
            ["python", "-X", "utf8", str(BASE_DIR / "pipefinal.py")],
            cwd=str(BASE_DIR),
            stdout=open(BASE_DIR / "retrain.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        flash("🔄 Retraining started. Check log for progress.", "success")
    except Exception as e:
        flash(f"❌ Retrain failed: {e}", "error")
    return redirect(url_for("admin_panel"))

@app.route("/admin/retrain-status")
@require_admin
def retrain_status():
    log_path = BASE_DIR / "retrain.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] \
          if log_path.exists() else "No log yet."
    return jsonify({"log": log})

@app.route("/admin/velocity-db")
@require_admin
def velocity_db_view():
    """JSON endpoint — full velocity database."""
    phones = VelocityEntry.query.filter_by(indicator_type="phone").all()
    urls   = VelocityEntry.query.filter_by(indicator_type="url").all()
    return jsonify({
        "phones": {e.indicator_value: {
            "count": e.count,
            "complaint_ids": _safe_json(e.complaint_ids, []),
            "first_seen": e.first_seen.isoformat() if e.first_seen else "",
            "last_seen":  e.last_seen.isoformat()  if e.last_seen  else "",
        } for e in phones},
        "urls": {e.indicator_value: {
            "count": e.count,
            "complaint_ids": _safe_json(e.complaint_ids, []),
            "first_seen": e.first_seen.isoformat() if e.first_seen else "",
            "last_seen":  e.last_seen.isoformat()  if e.last_seen  else "",
        } for e in urls},
    })

# ─────────────────────────────────────────────
# USER MANAGEMENT ROUTES (NEW v4)
# ─────────────────────────────────────────────
@app.route("/admin/users")
@require_admin
def admin_users():
    results = (db.session.query(User, func.count(Complaint.id).label("complaint_count"))
               .outerjoin(Complaint, User.id == Complaint.user_id)
               .group_by(User.id)
               .order_by(func.count(Complaint.id).desc())
               .all())
    users_data = [{"user": u, "complaint_count": cnt} for u, cnt in results]

    total_users    = User.query.count()
    admin_users_n  = User.query.filter_by(is_admin=True).count()
    total_c        = Complaint.query.count()
    today          = datetime.datetime.utcnow().date()
    today_c        = Complaint.query.filter(
        func.date(Complaint.timestamp) == today).count()

    return render_template("admin_users.html",
        user=current_user,
        users_data=users_data,
        total_users=total_users,
        admin_users_n=admin_users_n,
        total_complaints=total_c,
        today_complaints=today_c)

@app.route("/admin/users/<user_id>/toggle-admin", methods=["POST"])
@require_admin
def toggle_admin(user_id):
    u = db.session.get(User, user_id)
    if not u:
        flash("User not found.", "error")
    elif u.id == current_user.id:
        flash("You cannot change your own admin status.", "error")
    else:
        u.is_admin = not u.is_admin
        db.session.commit()
        action = "promoted to Admin" if u.is_admin else "demoted from Admin"
        flash(f"✅ {u.email} {action}.", "success")
    return redirect(url_for("admin_users"))

@app.route("/complaint/<complaint_id>/delete", methods=["POST"])
def delete_complaint(complaint_id):
    if not current_user.is_authenticated:
        abort(401)
    c = Complaint.query.filter_by(complaint_id=complaint_id).first_or_404()
    if c.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    # Delete associated PDF
    if c.pdf_path:
        pdf = BASE_DIR / c.pdf_path
        if pdf.exists():
            pdf.unlink()
    # Delete pending review
    PendingReview.query.filter_by(complaint_id=complaint_id).delete()
    db.session.delete(c)
    db.session.commit()
    flash("🗑 Complaint deleted.", "success")
    return redirect(url_for("complaint_history"))

# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()          # Create tables if they don't exist
    print("\n" + "="*55)
    print("  CyberShield v4.0 — Database-backed")
    print("  Velocity Tracker + Heatmap + NCRP PDF + History")
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)