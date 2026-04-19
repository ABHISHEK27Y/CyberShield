# ============================================================
# VERIFY master_dataset.csv is ready, then retrain
# Run this from your fullPIPIELINE folder
# ============================================================

import pandas as pd

df = pd.read_csv("master_dataset.csv")

print("=" * 50)
print("MASTER DATASET SUMMARY")
print("=" * 50)
print(f"Total rows   : {len(df):,}")
print(f"Fraud (1)    : {df['label'].sum():,}  ({df['label'].mean():.1%})")
print(f"Legit (0)    : {(df['label']==0).sum():,}  ({1-df['label'].mean():.1%})")
print(f"Avg length   : {df['text'].str.len().mean():.0f} chars")
print(f"Short (<300) : {(df['text'].str.len()<300).sum():,} rows")
print(f"Duplicates   : {df.duplicated(subset='text').sum()}")
print(f"Nulls        : {df['text'].isna().sum()}")
print()

# Check for Indian fraud keywords now present
indian_kw = ['upi','otp','kyc','paytm','gpay','aadhaar','neft','imps',
             'pm awas','yojana','fedex','parcel','courier','crypto','qr']
print("Indian/modern fraud keyword coverage:")
fraud_texts = df[df['label']==1]['text'].str.lower()
for kw in indian_kw:
    count = fraud_texts.str.contains(kw, na=False).sum()
    bar = '#' * min(count // 5, 30)
    print(f"  {kw:12s}: {count:4d}  {bar}")

print()
print("Ready for training.")
print("In your pipefinal.py, change:")
print('  df = pd.read_csv("final_unified_dataset.csv")')
print("to:")
print('  df = pd.read_csv("master_dataset.csv")')
print()
print("Also update label mapping — master_dataset uses 0/1 directly:")
print("  REMOVE this line (no longer needed):")
print("  df['label'] = df['label'].map({'Fraud':1, 'Not_Fraud':0})")
