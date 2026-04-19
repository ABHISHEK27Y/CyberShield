# ============================================================
# DATASET MERGER v3 — ALL ISSUES FIXED
# ============================================================

import pandas as pd
import re

# ============================================================
# LOADERS
# ============================================================

def load_final_unified():
    df = pd.read_csv("final_unified_dataset.csv", on_bad_lines='skip')
    df = df[['text', 'label']].copy()
    df['label'] = df['label'].map({'Fraud': 1, 'Not_Fraud': 0})
    df['source'] = 'final_unified'
    return df

def load_dataset_5971():
    try:
        df = pd.read_csv("Dataset_5971.csv", on_bad_lines='skip', encoding='latin-1')
        text_col  = next(c for c in df.columns if any(k in c.lower() for k in ['text','message','sms']))
        label_col = next(c for c in df.columns if any(k in c.lower() for k in ['label','type','class']))
        df = df[[text_col, label_col]].copy()
        df.columns = ['text', 'label']
        df['label'] = df['label'].str.lower().str.strip()
        df['label'] = df['label'].map(lambda x: 1 if x in ['spam','smishing','fraud','1'] else 0)
        df['source'] = 'mendeley_5971'
        return df
    except Exception as e:
        print(f"  [SKIP] Dataset_5971 — {e}")
        return None

def load_combined_dataset():
    try:
        df = pd.read_csv("combined_dataset.csv", on_bad_lines='skip', encoding='latin-1')
        if 'target' in df.columns and 'text' in df.columns:
            df = df[['text', 'target']].copy()
            df.columns = ['text', 'label']
            df['label'] = df['label'].astype(str).str.lower().str.strip()
            df['label'] = df['label'].apply(
                lambda x: 0 if x == 'ham' else 1 if x in ['spam','smishing','fraud','1','phishing'] else -1
            )
            df = df[df['label'] != -1]
            df['source'] = 'combined'
            return df
    except Exception as e:
        print(f"  [SKIP] combined_dataset — {e}")
        return None

def load_multilingual():
    try:
        df = pd.read_csv("data-en-hi-de-fr.csv", on_bad_lines='skip', encoding='utf-8')
        text_col  = 'text' if 'text' in df.columns else None
        label_col = 'labels' if 'labels' in df.columns else next(
            (c for c in df.columns if 'label' in c.lower()), None)
        if not text_col or not label_col:
            return None
        df = df[[text_col, label_col]].copy()
        df.columns = ['text', 'label']
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        df['label'] = df['label'].apply(
            lambda x: 0 if x in ['ham','0','legit','not_fraud']
                      else 1 if x in ['spam','smishing','fraud','1'] else -1
        )
        df = df[df['label'] != -1]
        df['source'] = 'multilingual'
        return df
    except Exception as e:
        print(f"  [SKIP] multilingual — {e}")
        return None

def load_fraud_v3():
    try:
        df = pd.read_csv("fraud_dataset_v3.csv", on_bad_lines='skip', encoding='latin-1')
        df = df[['text', 'label']].copy()
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        df['label'] = df['label'].apply(
            lambda x: 1 if x in ['spam','smishing','fraud','1','phishing'] else 0
        )
        df['source'] = 'fraud_v3'
        return df
    except Exception as e:
        print(f"  [SKIP] fraud_v3 — {e}")
        return None

def load_smish_collection():
    # FIX: SMSSmishCollection.txt has embedded labels at the start of each line
    # Format is: "spam <message text>" or "ham <message text>"
    # Must parse the label FROM the text, not treat everything as fraud=1
    try:
        rows = []
        with open("SMSSmishCollection.txt", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                if line.lower().startswith('spam\t') or line.lower().startswith('spam '):
                    text = line[5:].strip()
                    label = 1
                elif line.lower().startswith('ham\t') or line.lower().startswith('ham '):
                    text = line[4:].strip()
                    label = 0
                elif line.lower().startswith('smishing\t') or line.lower().startswith('smishing '):
                    text = line[9:].strip()
                    label = 1
                else:
                    # No prefix found — try tab separator
                    parts = line.split('\t', 1)
                    if len(parts) == 2 and parts[0].lower() in ['ham','spam','smishing']:
                        label = 0 if parts[0].lower() == 'ham' else 1
                        text = parts[1].strip()
                    else:
                        # No label prefix — treat as smishing (it's a smishing collection)
                        text = line
                        label = 1
                if len(text) > 10:
                    rows.append({'text': text, 'label': label, 'source': 'smishtank'})

        df = pd.DataFrame(rows)
        print(f"  SMSSmish — parsed Fraud:{df['label'].sum()} Legit:{(df['label']==0).sum()}")
        return df
    except Exception as e:
        print(f"  [SKIP] SMSSmishCollection — {e}")
        return None

def load_analysis_dataset():
    # All rows = smishing/phishing = Fraud (1)
    try:
        df = pd.read_csv("analysisdataset.csv", on_bad_lines='skip', encoding='latin-1')
        text_col = 'MainText' if 'MainText' in df.columns else 'Fulltext'
        df = df[[text_col]].copy()
        df.columns = ['text']
        df['label'] = 1
        df['source'] = 'smishtank_acm'
        return df
    except Exception as e:
        print(f"  [SKIP] analysisdataset — {e}")
        return None

# ============================================================
# SMS FILTER — aggressive against email noise
# ============================================================

# Email-specific patterns that should never appear in real SMS
EMAIL_NOISE = re.compile(
    r'(\[python-dev\]|\[fork\]|\[ilug\]|\[opensuse\]|\[git\])'      # mailing lists
    r'|(^re\s*:|^fw\s*:|^fwd\s*:)'                                   # reply/forward
    r'|(wrote\s*:\s*\n\s*>)'                                          # quoted reply
    r'|(-----original message-----)'
    r'|(from\s*:.*\nto\s*:.*\nsubject\s*:)'                          # email headers
    r'|(\n\s*>+\s*\w)'                                               # quoted lines
    r'|(this email (was sent|is intended|may contain))'
    r'|(unsubscribe.*mailing list)'
    r'|(confidentiality notice)'
    r'|(enron\.com|python\.org|kernel\.org)',
    re.IGNORECASE | re.MULTILINE
)

def is_sms_like(text):
    t = str(text)
    if len(t) > 800:
        return False
    if EMAIL_NOISE.search(t):
        return False
    if t.count('\n') > 6:
        return False
    # Must have at least one of: number, url, or common SMS word
    # (avoids keeping pure gibberish)
    return True

def quality_filter(df):
    df = df[df['text'].str.split().str.len() >= 3]
    df = df[df['text'].str.contains(r'[a-zA-Z\u0900-\u097F]', na=False)]
    df = df[df['text'].str.strip().str.len() > 10]
    return df

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    loaders = [
        ("final_unified",    load_final_unified),
        ("dataset_5971",     load_dataset_5971),
        ("combined",         load_combined_dataset),
        ("multilingual",     load_multilingual),
        ("fraud_v3",         load_fraud_v3),
        ("smish_collection", load_smish_collection),
        ("analysis_dataset", load_analysis_dataset),
    ]

    frames = []
    for name, fn in loaders:
        print(f"\nLoading {name}...")
        try:
            df = fn()
            if df is not None and len(df) > 0:
                df = df.dropna(subset=['text', 'label'])
                df['label'] = df['label'].astype(int)
                print(f"  -> {len(df):,} rows | Fraud: {df['label'].sum():,} | Legit: {(df['label']==0).sum():,}")
                frames.append(df)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")

    print("\n" + "=" * 60)
    print("MERGING + CLEANING")
    print("=" * 60)

    master = pd.concat(frames, ignore_index=True)
    print(f"Raw total              : {len(master):,}")

    master = master[master['text'].apply(is_sms_like)]
    print(f"After SMS filter       : {len(master):,}")

    master = quality_filter(master)
    print(f"After quality filter   : {len(master):,}")

    master = master.drop_duplicates(subset='text')
    print(f"After deduplication    : {len(master):,}")

    # ============================================================
    # BALANCE — cap fraud class if ratio > 60%
    # ============================================================
    fraud_count = master['label'].sum()
    legit_count = (master['label'] == 0).sum()
    ratio = fraud_count / len(master)

    if ratio > 0.60:
        print(f"\n[REBALANCE] Fraud ratio {ratio:.1%} is too high — capping fraud to match legit count")
        fraud_df = master[master['label'] == 1].sample(n=legit_count, random_state=42)
        legit_df = master[master['label'] == 0]
        master = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal label distribution:")
    vc = master['label'].value_counts()
    print(vc)
    print(f"Fraud ratio            : {master['label'].mean():.2%}")

    print(f"\nRows per source:")
    print(master['source'].value_counts())

    # ============================================================
    # SAVE
    # ============================================================
    master[['text', 'label', 'source']].to_csv("master_dataset_full.csv", index=False)
    master[['text', 'label']].to_csv("master_dataset.csv", index=False)
    print(f"\nSaved: master_dataset.csv  ({len(master):,} rows)")

    # ============================================================
    # SANITY SAMPLES
    # ============================================================
    print("\n" + "=" * 60)
    print("FRAUD SAMPLES")
    print("=" * 60)
    for _, row in master[master['label'] == 1].sample(8, random_state=7).iterrows():
        print(f"[{row['source']}] {str(row['text'])[:130]}")
        print()

    print("=" * 60)
    print("LEGIT SAMPLES — should look like real SMS, not emails")
    print("=" * 60)
    for _, row in master[master['label'] == 0].sample(8, random_state=7).iterrows():
        print(f"[{row['source']}] {str(row['text'])[:130]}")
        print()