"""
Ablation Study: age_campaign Feature Value Analysis
Compares model AUC with and without the age_campaign interaction feature.
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
RESULTS_FILE = 'age_campaign_ablation_results.txt'

def load_and_preprocess(include_age_campaign=True):
    """Load and preprocess data, optionally including age_campaign feature."""
    
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
    
    # Map target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Feature Engineering
    if include_age_campaign and 'age' in df.columns and 'campaign' in df.columns:
        df['age_campaign'] = df['age'] * df['campaign']
    
    # is_first_contact (always include)
    if 'pdays' in df.columns:
        df['is_first_contact'] = (df['pdays'] == 999).astype(int)
    
    # Drop duration (leakage) and target
    X = df.drop(['y', 'duration'], axis=1)
    y = df['y']
    
    # Encode categoricals
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ordinal encoding for education
    education_order = {
        'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
        'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': -1
    }
    
    if 'education' in X.columns:
        X['education'] = X['education'].map(education_order).fillna(-1)
        if 'education' in categorical_features:
            categorical_features.remove('education')
    
    # One-hot encode remaining categoricals
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X, y


def evaluate_model(X, y, description):
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
    results = []
    
    print("=" * 60)
    print("ABLATION STUDY: age_campaign Feature")
    print("=" * 60)
    
    # --- Model WITH age_campaign ---
    print("\n[1/2] Evaluating model WITH age_campaign...")
    X_with, y = load_and_preprocess(include_age_campaign=True)
    auc_with = evaluate_model(X_with, y, "With age_campaign")
    
    print(f"  Features: {X_with.shape[1]}")
    print(f"  AUC scores: {auc_with}")
    print(f"  Mean AUC: {auc_with.mean():.4f} (+/- {auc_with.std():.4f})")
    
    # --- Model WITHOUT age_campaign ---
    print("\n[2/2] Evaluating model WITHOUT age_campaign...")
    X_without, y = load_and_preprocess(include_age_campaign=False)
    auc_without = evaluate_model(X_without, y, "Without age_campaign")
    
    print(f"  Features: {X_without.shape[1]}")
    print(f"  AUC scores: {auc_without}")
    print(f"  Mean AUC: {auc_without.mean():.4f} (+/- {auc_without.std():.4f})")
    
    # --- Comparison ---
    diff = auc_with.mean() - auc_without.mean()
    pct_diff = (diff / auc_without.mean()) * 100
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} | {'Mean AUC':<12} | {'Std':<10}")
    print("-" * 60)
    print(f"{'WITH age_campaign':<30} | {auc_with.mean():<12.4f} | {auc_with.std():<10.4f}")
    print(f"{'WITHOUT age_campaign':<30} | {auc_without.mean():<12.4f} | {auc_without.std():<10.4f}")
    print("-" * 60)
    print(f"Difference: {diff:+.4f} ({pct_diff:+.2f}%)")
    
    if abs(diff) < 0.001:
        conclusion = "NEGLIGIBLE IMPACT: The age_campaign feature provides no meaningful improvement."
    elif diff > 0:
        conclusion = f"POSITIVE IMPACT: The age_campaign feature improves AUC by {diff:.4f}."
    else:
        conclusion = f"NEGATIVE IMPACT: The age_campaign feature decreases AUC by {abs(diff):.4f}."
    
    print(f"\nConclusion: {conclusion}")
    
    # --- Save Results ---
    with open(RESULTS_FILE, 'w') as f:
        f.write("ABLATION STUDY: age_campaign Feature Value Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  - Model: LightGBM with SMOTE\n")
        f.write(f"  - Cross-Validation: 5-fold Stratified\n")
        f.write(f"  - Random State: {RANDOM_STATE}\n\n")
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<30} | {'Mean AUC':<12} | {'Std':<10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'WITH age_campaign':<30} | {auc_with.mean():<12.4f} | {auc_with.std():<10.4f}\n")
        f.write(f"{'WITHOUT age_campaign':<30} | {auc_without.mean():<12.4f} | {auc_without.std():<10.4f}\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Difference: {diff:+.4f} ({pct_diff:+.2f}%)\n\n")
        f.write(f"Fold-by-Fold Comparison:\n")
        f.write(f"{'Fold':<10} | {'With':<12} | {'Without':<12} | {'Diff':<12}\n")
        f.write("-" * 50 + "\n")
        for i, (w, wo) in enumerate(zip(auc_with, auc_without)):
            f.write(f"{'Fold ' + str(i+1):<10} | {w:<12.4f} | {wo:<12.4f} | {w-wo:+.4f}\n")
        f.write("\n")
        f.write(f"Conclusion: {conclusion}\n")
    
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
