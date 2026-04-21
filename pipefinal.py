# ============================================================
# FRAUD DETECTION SYSTEM -- FINAL VERSION
# ============================================================

import pandas as pd
import numpy as np
import re
import pickle
from itertools import product

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)

# ============================================================
# CONSTANTS
# ============================================================

CUSTOM_STOPWORDS = [
    "the","is","in","it","of","and","to","a","that","this","for","on",
    "are","with","as","at","be","by","from","or","an","was","were",
    "have","has","had","will","would","could","should","may","might",
    "do","did","does","been","being","i","me","my","we","our","you",
    "your","he","she","they","his","her","their","its","not","no",
    "so","but","if","about","which","who","what","when","where",
    "how","all","can","just","than","then","also","more","some","any",
    "there","here","into","up","out","said","get","got",
    "love","health","thanks","university","http","pm","hi","hello",
    "dear","sir","please","now","new","good","time","day","name",
    "old","years","am","year"
]

CONVERSATIONAL_PATTERNS = [
    r'^(hi|hey|hello|good morning|good night|gm|gn)\b.{0,40}$',
    r'^how are you',
    r'^(ok|okay|sure|thanks|thank you|noted|got it)\s*[.!]?$',
]

# ============================================================
# PURE FUNCTIONS
# ============================================================

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+',   ' SUSPICIOUS_URL ',   text)
    text = re.sub(r'\b[789]\d{9}\b',      ' PHONE_NUMBER ',     text)
    text = re.sub(r'rs\.?\s*\d[\d,]+',    ' MONEY_AMOUNT ',     text)
    text = re.sub(r'\$\s*\d[\d,]+',       ' MONEY_AMOUNT_USD ', text)
    text = re.sub(r'\b\d{4,6}\b',         ' OTP_NUMBER ',       text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_conversational(text):
    t = text.lower().strip()
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.match(pattern, t):
            return True
    return False


if __name__ == '__main__':

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # CHANGE 1: filename is now master_dataset.csv
    # CHANGE 2: labels are already 0/1 integers — no mapping needed
    # ----------------------------------------------------------

    df = pd.read_csv("master_dataset.csv")                  # CHANGED
    df = df[['text', 'label']].dropna().drop_duplicates()
    df['label'] = df['label'].astype(int)                   # CHANGED (was string mapping)

    print(f"Total samples  : {len(df)}")
    print(f"Fraud (1)      : {df['label'].sum()}")
    print(f"Legit (0)      : {(df['label'] == 0).sum()}")
    print(f"Fraud ratio    : {df['label'].mean():.2%}\n")

    # ----------------------------------------------------------
    # 2. PREPROCESS
    # ----------------------------------------------------------

    df["clean_text"] = df["text"].apply(preprocess)

    # ----------------------------------------------------------
    # 3. THREE-WAY SPLIT (train 70% / val 15% / test 15%)
    # ----------------------------------------------------------

    X = df["clean_text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}\n")

    # ----------------------------------------------------------
    # 4. VECTORIZE — fit ONCE on train only, frozen after this
    # ----------------------------------------------------------

    print("Vectorizing...")

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.85,     # CHANGE 3: min_df 5→3 (smaller dataset)
        stop_words=CUSTOM_STOPWORDS, max_features=20000, sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        max_features=5000, min_df=2                     # CHANGE 3: min_df 3→2 (smaller dataset)
    )
    vectorizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)

    # ----------------------------------------------------------
    # 5. TRAIN BASE MODELS
    # ----------------------------------------------------------

    print("Training base models...")

    lr = LogisticRegression(
        max_iter=1000, C=0.5,
        class_weight={0: 1.0, 1: 1.5},
        solver='lbfgs'
    )
    nb      = MultinomialNB(alpha=0.1)
    svm_cal = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, C=0.8),
        method='sigmoid', cv=3
    )

    lr.fit(X_train_vec, y_train)
    nb.fit(X_train_vec, y_train)
    svm_cal.fit(X_train_vec, y_train)

    print("Base models trained.\n")

    # ----------------------------------------------------------
    # 6. TUNE ENSEMBLE WEIGHTS ON VALIDATION SET
    # ----------------------------------------------------------

    print("Tuning ensemble weights on validation set...")

    p_lr_val  = lr.predict_proba(X_val_vec)
    p_nb_val  = nb.predict_proba(X_val_vec)
    p_svm_val = svm_cal.predict_proba(X_val_vec)

    best_f1      = 0.0
    best_weights = (0.5, 0.3, 0.2)
    candidates   = [round(w * 0.1, 1) for w in range(1, 8)]

    for w1, w2, w3 in product(candidates, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) > 0.01:
            continue
        blended = w1 * p_lr_val + w2 * p_svm_val + w3 * p_nb_val
        preds   = (blended[:, 1] > 0.5).astype(int)
        f1      = f1_score(y_val, preds, average='macro')
        if f1 > best_f1:
            best_f1      = f1
            best_weights = (w1, w2, w3)

    W_LR, W_SVM, W_NB = best_weights
    print(f"Best weights -> LR: {W_LR}, SVM: {W_SVM}, NB: {W_NB}  "
          f"(val macro-F1: {best_f1:.4f})\n")

    def ensemble_proba(X_vec):
        p1 = lr.predict_proba(X_vec)
        p2 = nb.predict_proba(X_vec)
        p3 = svm_cal.predict_proba(X_vec)
        return W_LR * p1 + W_SVM * p3 + W_NB * p2

    # ----------------------------------------------------------
    # 7. PRE-RETRAIN EVALUATION (val set)
    # ----------------------------------------------------------

    print("--- RESULTS BEFORE HARD NEGATIVE RETRAINING (on val set) ---")

    probs_before  = ensemble_proba(X_val_vec)
    y_pred_before = (probs_before[:, 1] > 0.5).astype(int)

    print(f"Accuracy  : {accuracy_score(y_val, y_pred_before):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_val, probs_before[:, 1]):.4f}")
    print(classification_report(y_val, y_pred_before, target_names=['Legit', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_before))
    print()

    # ----------------------------------------------------------
    # 8. HARD NEGATIVE RETRAINING
    # ----------------------------------------------------------

    print("Mining hard negatives from validation set...")

    hard_mask   = y_pred_before != y_val.values
    hard_texts  = X_val[hard_mask].tolist()
    hard_labels = y_val[hard_mask].tolist()

    print(f"Hard examples found: {len(hard_texts)}\n")

    X_aug     = list(X_train) + hard_texts
    y_aug     = list(y_train) + hard_labels
    X_aug_vec = vectorizer.transform(X_aug)

    lr.fit(X_aug_vec, y_aug)
    nb.fit(X_aug_vec, y_aug)
    svm_cal.fit(X_aug_vec, y_aug)

    print("Hard negative retraining complete.\n")

    # ----------------------------------------------------------
    # 9. FINAL EVALUATION ON UNTOUCHED TEST SET
    # ----------------------------------------------------------

    print("--- FINAL RESULTS ON TEST SET ---")

    probs_final  = ensemble_proba(X_test_vec)
    y_pred_final = (probs_final[:, 1] > 0.5).astype(int)

    print(f"Accuracy   : {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y_test, probs_final[:, 1]):.4f}")
    print(classification_report(y_test, y_pred_final, target_names=['Legit', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))
    print()

    print("--- THRESHOLD SENSITIVITY (on test set) ---")
    print(f"{'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1 (Fraud)':>12}")
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        preds  = (probs_final[:, 1] > threshold).astype(int)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        fraud  = report.get('1', {})
        print(f"{threshold:<12.2f} {fraud.get('precision',0):>10.4f} "
              f"{fraud.get('recall',0):>8.4f} {fraud.get('f1-score',0):>12.4f}")
    print()

    # ----------------------------------------------------------
    # 10. CROSS-VALIDATION
    # ----------------------------------------------------------

    print("Running 5-fold stratified cross-validation (LR baseline)...")

    cv_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.85,
            stop_words=CUSTOM_STOPWORDS, max_features=20000, sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000, C=0.5,
            class_weight={0: 1.0, 1: 1.5}, solver='lbfgs'
        ))
    ])

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        cv_pipeline, X, y,
        cv=skf, scoring='f1_macro',
        n_jobs=1
    )

    print(f"CV Macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Per-fold   : {[round(float(s), 4) for s in scores]}\n")

    # ----------------------------------------------------------
    # 11. SAVE MODEL ARTIFACTS
    # ----------------------------------------------------------

    with open("final_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("final_models.pkl", "wb") as f:
        pickle.dump({
            "lr"      : lr,
            "nb"      : nb,
            "svm"     : svm_cal,
            "weights" : best_weights
        }, f)

    print("Model saved -> final_vectorizer.pkl + final_models.pkl")

    # ----------------------------------------------------------
    # 12. PREDICTION FUNCTION
    # ----------------------------------------------------------

    def predict_message(message, threshold=0.45):
        clean = preprocess(message)
        vec   = vectorizer.transform([clean])
        prob  = float(ensemble_proba(vec)[0][1])

        if is_conversational(message) and prob < 0.35:
            label = "LEGIT"
        elif prob >= 0.85:
            label = "FRAUD"
        elif prob >= threshold:
            label = "SUSPICIOUS"
        elif prob >= 0.30:
            label = "UNCERTAIN"
        else:
            label = "LEGIT"

        return {
            "prediction"        : label,
            "fraud_probability" : round(prob, 4),
            "legit_probability" : round(1 - prob, 4),
            "risk_level"        : label,
            "threshold_used"    : threshold
        }

    # ----------------------------------------------------------
    # 13. SMOKE TEST
    # ----------------------------------------------------------

    test_messages = [
        "Congratulations! Your account has been selected. Click the link to claim Rs 50,000 now.",
        "Your OTP is 482910. Do not share this with anyone.",
        "Hey, are you coming to college tomorrow?",
        "Dear customer, verify your KYC immediately to avoid account suspension.",
        "Your salary of Rs 45,000 has been credited to your account.",
        "FREE entry: Call 9876543210 to win an iPhone. Limited offer!",
        "Your Aadhaar KYC verification is pending. Update now to avoid account block.",
        "PM Awas Yojana: House allotted. Pay Rs 5000 registration at pmawas.ml now.",
        "USPS: Your parcel is on hold. Confirm address at usps-delivery.com/verify",
    ]

    print("\n--- SMOKE TEST ---")
    print(f"{'Message':<55} {'Risk':<12} {'Fraud prob'}")
    print("-" * 80)
    for msg in test_messages:
        result = predict_message(msg)
        short  = msg[:52] + "..." if len(msg) > 55 else msg
        flag   = " <- known FP" if "salary" in msg.lower() else ""
        print(f"{short:<55} {result['risk_level']:<12} {result['fraud_probability']:.4f}{flag}")

    print("\nLimitation: MONEY_AMOUNT token fires on both fraud and legit bank SMS.")
    print("            Mention this in your project limitations section.\n")
    print("Training pipeline completed successfully.")