# ============================================================
# 🚀 FINAL FRAUD DETECTION SYSTEM (CORRECTED VERSION)
# ============================================================

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv("final_unified_dataset.csv")
df = df[['text', 'label']].dropna()

df['label'] = df['label'].str.lower().str.strip()
df['label'] = df['label'].map(lambda x: 1 if x in ['fraud', 'spam', '1'] else 0)

print(f"Dataset: {len(df)}")
print(f"Fraud: {df['label'].sum()} | Legit: {(df['label']==0).sum()}")

# ============================================================
# 2️⃣ PREPROCESSING
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

df["clean_text"] = df["text"].apply(preprocess)

# ============================================================
# 3️⃣ VECTORIZER
# ============================================================

word_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.85,
    max_features=20000,
    sublinear_tf=True,
    stop_words='english'
)

char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    max_features=5000
)

vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])

# ============================================================
# 4️⃣ SPLIT (TRAIN / VAL / TEST) ✅ FIXED
# ============================================================

X = df["clean_text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Splits:",
      len(X_train), len(X_val), len(X_test))

# ============================================================
# 5️⃣ VECTORIZATION (FIT ONLY ON TRAIN) ✅
# ============================================================

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ============================================================
# 6️⃣ MODELS (FIXED CLASS WEIGHT) ✅
# ============================================================

lr = LogisticRegression(
    max_iter=1000,
    C=0.5,
    class_weight='balanced'
)

nb = MultinomialNB()

svm = LinearSVC(class_weight='balanced')
svm_cal = CalibratedClassifierCV(svm, method='sigmoid', cv=2)

lr.fit(X_train_vec, y_train)
nb.fit(X_train_vec, y_train)
svm_cal.fit(X_train_vec, y_train)

# ============================================================
# 7️⃣ ENSEMBLE
# ============================================================

def ensemble_proba(X):
    p1 = lr.predict_proba(X)
    p2 = nb.predict_proba(X)
    p3 = svm_cal.predict_proba(X)
    return 0.5*p1 + 0.3*p3 + 0.2*p2

# ============================================================
# 8️⃣ BASELINE ON VALIDATION
# ============================================================

print("\n--- BASELINE (VALIDATION) ---")

val_probs = ensemble_proba(X_val_vec)
y_val_pred = (val_probs[:,1] > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_val, y_val_pred))

# ============================================================
# 9️⃣ HARD NEGATIVE RETRAINING (FROM VALIDATION ONLY) ✅
# ============================================================

hard_texts = []
hard_labels = []

for i in range(len(y_val)):
    if y_val.iloc[i] != y_val_pred[i]:
        hard_texts.append(X_val.iloc[i])
        hard_labels.append(y_val.iloc[i])

print("Hard examples:", len(hard_texts))

# Augment training
X_aug = list(X_train) + hard_texts
y_aug = list(y_train) + hard_labels

# 🚨 IMPORTANT: NO fit_transform
X_aug_vec = vectorizer.transform(X_aug)

# Retrain
lr.fit(X_aug_vec, y_aug)
nb.fit(X_aug_vec, y_aug)
svm_cal.fit(X_aug_vec, y_aug)

print("Retraining done ✔")

# ============================================================
# 🔟 FINAL EVALUATION ON TEST (CLEAN) ✅
# ============================================================

print("\n--- FINAL TEST RESULTS ---")

test_probs = ensemble_proba(X_test_vec)
y_test_pred = (test_probs[:,1] > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ============================================================
# 1️⃣1️⃣ SMART PREDICTION FUNCTION (FIXED RULE)
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


def predict_message(message):
    clean = preprocess(message)
    vec = vectorizer.transform([clean])

    prob = float(ensemble_proba(vec)[0][1])

    # safer override
    if is_conversational(message) and prob < 0.3:
        label = "LEGIT"
    else:
        if prob >= 0.85:
            label = "FRAUD"
        elif prob >= 0.60:
            label = "SUSPICIOUS"
        elif prob >= 0.40:
            label = "UNCERTAIN"
        else:
            label = "LEGIT"

    return {
        "prediction": label,
        "fraud_probability": round(prob,4)
    }

# ============================================================
# 1️⃣2️⃣ SAVE MODEL
# ============================================================

with open("final_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("final_models.pkl", "wb") as f:
    pickle.dump({
        "lr": lr,
        "nb": nb,
        "svm": svm_cal
    }, f)

print("\n✅ FINAL MODEL SAVED")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\nTraining completed successfully!")