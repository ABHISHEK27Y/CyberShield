import pickle
import sys
import re
import numpy as np

# ============================================================
# 1️⃣ LOAD MODELS
# ============================================================

with open("final_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("final_models.pkl", "rb") as f:
    models = pickle.load(f)

lr = models["lr"]
nb = models["nb"]
svm = models["svm"]

# ============================================================
# 2️⃣ PREPROCESS FUNCTION (SAME AS TRAINING)
# ============================================================

def preprocess(text):
    text = str(text).lower()

    text = re.sub(r'http\S+|www\.\S+', ' SUSPICIOUS_URL ', text)
    text = re.sub(r'\b[789]\d{9}\b', ' PHONE_NUMBER ', text)
    text = re.sub(r'rs\.?\s*\d[\d,]+', ' MONEY_AMOUNT ', text)
    text = re.sub(r'\$\s*\d[\d,]+', ' MONEY_AMOUNT_USD ', text)
    text = re.sub(r'\b\d{4,6}\b', ' OTP_NUMBER ', text)

    text = re.sub(r'[^\w\s₹]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================================
# 3️⃣ ENSEMBLE FUNCTION
# ============================================================

def ensemble_proba(X):
    p1 = lr.predict_proba(X)
    p2 = nb.predict_proba(X)
    p3 = svm.predict_proba(X)

    return 0.5*p1 + 0.3*p3 + 0.2*p2

# ============================================================
# 4️⃣ CONVERSATIONAL DETECTION (GENERALIZED)
# ============================================================

def is_conversational(text):
    words = text.lower().split()

    casual_words = {"hi", "hello", "buddy", "morning", "night", "hey"}

    if len(words) <= 7 and any(w in casual_words for w in words):
        return True

    if "call" in words and "me" in words:
        return True

    if "how" in words and "you" in words:
        return True

    return False

# ============================================================
# 5️⃣ PREDICTION FUNCTION
# ============================================================

def predict_message(message):
    clean = preprocess(message)
    vec = vectorizer.transform([clean])

    prob = float(ensemble_proba(vec)[0][1])
    legit_prob = 1 - prob

    # Smart logic
    if is_conversational(message) and prob < 0.9:
        label = "✅ LEGITIMATE MESSAGE"
        risk = "Low Risk"

    elif prob >= 0.85:
        label = "🚨 HIGH RISK FRAUD / SPAM"
        risk = "Very Dangerous"

    elif prob >= 0.60:
        label = "⚠️ SUSPICIOUS MESSAGE"
        risk = "Medium Risk"

    elif prob >= 0.40:
        label = "⚠️ UNCERTAIN"
        risk = "Needs Manual Review"

    else:
        label = "✅ LEGITIMATE MESSAGE"
        risk = "Low Risk"

    return label, prob, legit_prob, risk

# ============================================================
# 6️⃣ UI HEADER
# ============================================================

print("=" * 60)
print("🚀 ADVANCED FRAUD DETECTION SYSTEM")
print("=" * 60)
print("Paste suspicious message below.")
print("Type END (new line) to finish.")
print("Type EXIT to quit.\n")

# ============================================================
# 7️⃣ MAIN LOOP
# ============================================================

while True:
    print("\nEnter suspicious message:")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "EXIT":
            print("Exiting... 👋")
            sys.exit()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    message = "\n".join(lines).strip()

    if not message:
        print("⚠️ No input provided.")
        continue

    # Predict
    label, fraud_prob, legit_prob, risk = predict_message(message)

    # Output
    print("\n" + "-" * 50)
    print(f"📌 Prediction: {label}")
    print(f"📊 Fraud Probability: {fraud_prob:.4f}")
    print(f"📊 Legitimate Probability: {legit_prob:.4f}")
    print(f"⚡ Risk Level: {risk}")
    print("-" * 50)