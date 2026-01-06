"""
Ablation Study: Adding age_campaign to new features
Tests if age_campaign provides additional value on top of contact_intensity + financial_stress
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
RESULTS_FILE = 'age_campaign_with_new_features_results.txt'


def load_and_preprocess(include_age_campaign=False):
    """Load and preprocess with new features + optional age_campaign."""
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # New validated features (always include)
    df['contact_intensity'] = df['campaign'] / (df['pdays'] + 1)
    df['financial_stress'] = (
        (df['housing'] == 'yes').astype(int) + 
        (df['loan'] == 'yes').astype(int) + 
        (df['default'] == 'yes').astype(int)
    )
    
    # Optional: age_campaign
    if include_age_campaign:
        df['age_campaign'] = df['age'] * df['campaign']
    
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
    return cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)


def main():
    # Previous results (from new_features_ablation_results.txt)
    prev_all_features = {
        'auc_mean': 0.7884,
        'auc_std': 0.0055,
        'auc_scores': [0.7811, 0.7855, 0.7978, 0.7886, 0.7891]
    }
    
    print("=" * 70)
    print("ABLATION: Adding age_campaign to new features")
    print("=" * 70)
    print("\nBase: contact_intensity + financial_stress")
    print("Test: contact_intensity + financial_stress + age_campaign")
    print(f"\nUsing previous result for base (AUC: {prev_all_features['auc_mean']:.4f})")
    
    # Only run the new trial
    print("\n[1/1] Evaluating with age_campaign added...")
    X, y = load_and_preprocess(include_age_campaign=True)
    auc_with_age = evaluate_model(X, y)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  AUC scores: {auc_with_age}")
    print(f"  Mean AUC: {auc_with_age.mean():.4f} (+/- {auc_with_age.std():.4f})")
    
    # Compare
    diff = auc_with_age.mean() - prev_all_features['auc_mean']
    pct_diff = (diff / prev_all_features['auc_mean']) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<50} | {'Mean AUC':<10} | {'Std':<10}")
    print("-" * 75)
    print(f"{'contact_intensity + financial_stress':<50} | {prev_all_features['auc_mean']:<10.4f} | {prev_all_features['auc_std']:<10.4f}")
    print(f"{'contact_intensity + financial_stress + age_campaign':<50} | {auc_with_age.mean():<10.4f} | {auc_with_age.std():<10.4f}")
    print("-" * 75)
    print(f"Difference: {diff:+.4f} ({pct_diff:+.2f}%)")
    
    # Fold comparison
    print("\nFold-by-Fold:")
    prev_scores = prev_all_features['auc_scores']
    for i, (prev, new) in enumerate(zip(prev_scores, auc_with_age)):
        print(f"  Fold {i+1}: {prev:.4f} -> {new:.4f} ({new-prev:+.4f})")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if diff > 0.001:
        verdict = f"Adding age_campaign HELPS (+{diff:.4f} AUC)"
    elif diff < -0.001:
        verdict = f"Adding age_campaign HURTS ({diff:.4f} AUC)"
    else:
        verdict = "Adding age_campaign has NEGLIGIBLE effect"
    
    print(verdict)
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        f.write("ABLATION: Adding age_campaign to new features\n")
        f.write("=" * 70 + "\n\n")
        f.write("Question: Does age_campaign add value on top of contact_intensity + financial_stress?\n\n")
        f.write("Results:\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Configuration':<50} | {'Mean AUC':<10} | {'Std':<10}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'contact_intensity + financial_stress':<50} | {prev_all_features['auc_mean']:<10.4f} | {prev_all_features['auc_std']:<10.4f}\n")
        f.write(f"{'+ age_campaign':<50} | {auc_with_age.mean():<10.4f} | {auc_with_age.std():<10.4f}\n")
        f.write("-" * 75 + "\n\n")
        f.write(f"Difference: {diff:+.4f} ({pct_diff:+.2f}%)\n\n")
        f.write(f"Conclusion: {verdict}\n")
    
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
