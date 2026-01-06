"""
Comparison Script: Original vs V2 (New Features)
Compares improved_bank_marketing.py vs improved_bank_marketing_v2.py

Original: age_campaign, is_first_contact
V2: contact_intensity, financial_stress
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
RESULTS_FILE = 'v1_vs_v2_comparison_results.txt'


def load_and_preprocess_v1(filepath):
    """Original preprocessing with age_campaign and is_first_contact."""
    df = pd.read_csv(filepath, sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Original features
    if 'age' in df.columns and 'campaign' in df.columns:
        df['age_campaign'] = df['age'] * df['campaign']
    if 'pdays' in df.columns:
        df['is_first_contact'] = (df['pdays'] == 999).astype(int)
    
    X = df.drop(['y', 'duration'], axis=1)
    y = df['y']
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    education_order = {
        'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
        'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': -1
    }
    
    if 'education' in X.columns:
        X['education'] = X['education'].map(education_order).fillna(-1)
        if 'education' in categorical_features:
            categorical_features.remove('education')
    
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X, y


def load_and_preprocess_v2(filepath):
    """V2 preprocessing with contact_intensity and financial_stress."""
    df = pd.read_csv(filepath, sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # New validated features
    if 'campaign' in df.columns and 'pdays' in df.columns:
        df['contact_intensity'] = df['campaign'] / (df['pdays'] + 1)
    
    if all(col in df.columns for col in ['housing', 'loan', 'default']):
        df['financial_stress'] = (
            (df['housing'] == 'yes').astype(int) + 
            (df['loan'] == 'yes').astype(int) + 
            (df['default'] == 'yes').astype(int)
        )
    
    X = df.drop(['y', 'duration'], axis=1)
    y = df['y']
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    education_order = {
        'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
        'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': -1
    }
    
    if 'education' in X.columns:
        X['education'] = X['education'].map(education_order).fillna(-1)
        if 'education' in categorical_features:
            categorical_features.remove('education')
    
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X, y


def evaluate_model(X, y):
    """Train and evaluate LightGBM with SMOTE using 5-fold CV."""
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    )
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return auc_scores


def main():
    filepath = 'data/bank-additional/bank-additional-full.csv'
    
    print("=" * 70)
    print("COMPARISON: Original (V1) vs New Features (V2)")
    print("=" * 70)
    
    print("\nV1 Features: age_campaign, is_first_contact")
    print("V2 Features: contact_intensity, financial_stress")
    
    # --- V1: Original ---
    print("\n[1/2] Evaluating V1 (Original)...")
    X_v1, y = load_and_preprocess_v1(filepath)
    auc_v1 = evaluate_model(X_v1, y)
    print(f"  Features: {X_v1.shape[1]}")
    print(f"  AUC scores: {auc_v1}")
    print(f"  Mean AUC: {auc_v1.mean():.4f} (+/- {auc_v1.std():.4f})")
    
    # --- V2: New Features ---
    print("\n[2/2] Evaluating V2 (New Features)...")
    X_v2, y = load_and_preprocess_v2(filepath)
    auc_v2 = evaluate_model(X_v2, y)
    print(f"  Features: {X_v2.shape[1]}")
    print(f"  AUC scores: {auc_v2}")
    print(f"  Mean AUC: {auc_v2.mean():.4f} (+/- {auc_v2.std():.4f})")
    
    # --- Comparison ---
    diff = auc_v2.mean() - auc_v1.mean()
    pct_diff = (diff / auc_v1.mean()) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Version':<40} | {'Mean AUC':<12} | {'Std':<10}")
    print("-" * 70)
    print(f"{'V1: Original (age_campaign, is_first_contact)':<40} | {auc_v1.mean():<12.4f} | {auc_v1.std():<10.4f}")
    print(f"{'V2: New (contact_intensity, financial_stress)':<40} | {auc_v2.mean():<12.4f} | {auc_v2.std():<10.4f}")
    print("-" * 70)
    print(f"Improvement: {diff:+.4f} ({pct_diff:+.2f}%)")
    
    # Fold-by-fold
    print("\nFold-by-Fold Comparison:")
    print(f"{'Fold':<10} | {'V1 (Original)':<15} | {'V2 (New)':<15} | {'Diff':<10}")
    print("-" * 55)
    for i, (v1, v2) in enumerate(zip(auc_v1, auc_v2)):
        print(f"Fold {i+1:<5} | {v1:<15.4f} | {v2:<15.4f} | {v2-v1:+.4f}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if diff > 0.003:
        verdict = f"V2 is SIGNIFICANTLY BETTER than V1 (+{diff:.4f} AUC, +{pct_diff:.2f}%)"
    elif diff > 0:
        verdict = f"V2 is slightly better than V1 (+{diff:.4f} AUC)"
    elif diff > -0.003:
        verdict = f"V1 and V2 are essentially equivalent ({diff:+.4f} AUC)"
    else:
        verdict = f"V1 is better than V2 ({diff:.4f} AUC)"
    
    print(verdict)
    
    # Win rate across folds
    wins = sum(1 for v1, v2 in zip(auc_v1, auc_v2) if v2 > v1)
    print(f"V2 wins in {wins}/5 folds")
    
    # --- Save Results ---
    with open(RESULTS_FILE, 'w') as f:
        f.write("COMPARISON: Original (V1) vs New Features (V2)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Feature Engineering Comparison:\n")
        f.write("-" * 70 + "\n")
        f.write("V1 (Original):\n")
        f.write("  - age_campaign = age * campaign\n")
        f.write("  - is_first_contact = (pdays == 999)\n\n")
        f.write("V2 (New):\n")
        f.write("  - contact_intensity = campaign / (pdays + 1)\n")
        f.write("  - financial_stress = (housing='yes') + (loan='yes') + (default='yes')\n\n")
        
        f.write("Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Version':<45} | {'Mean AUC':<12} | {'Std':<10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'V1: Original':<45} | {auc_v1.mean():<12.4f} | {auc_v1.std():<10.4f}\n")
        f.write(f"{'V2: New Features':<45} | {auc_v2.mean():<12.4f} | {auc_v2.std():<10.4f}\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(f"Improvement: {diff:+.4f} ({pct_diff:+.2f}%)\n\n")
        
        f.write("Fold-by-Fold Comparison:\n")
        f.write(f"{'Fold':<10} | {'V1':<12} | {'V2':<12} | {'Diff':<10}\n")
        f.write("-" * 50 + "\n")
        for i, (v1, v2) in enumerate(zip(auc_v1, auc_v2)):
            f.write(f"Fold {i+1:<5} | {v1:<12.4f} | {v2:<12.4f} | {v2-v1:+.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 70 + "\n")
        f.write(f"{verdict}\n")
        f.write(f"V2 wins in {wins}/5 folds\n")
    
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
