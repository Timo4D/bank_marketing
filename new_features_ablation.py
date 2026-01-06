"""
Ablation Study: New Engineered Features
Tests: contact_intensity, previous_success_rate, financial_stress
Compares each individually and all together against baseline.
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
RESULTS_FILE = 'new_features_ablation_results.txt'


def load_and_preprocess(include_contact_intensity=False, 
                        include_previous_success=False,
                        include_financial_stress=False):
    """Load and preprocess data with optional new engineered features."""
    
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
    
    # Map target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # --- NEW ENGINEERED FEATURES ---
    
    # 1. Contact Intensity: campaign / (pdays + 1)
    # Measures how aggressively this lead is being pursued
    # For pdays=999 (never contacted), this becomes campaign/1000 (very low intensity)
    if include_contact_intensity:
        df['contact_intensity'] = df['campaign'] / (df['pdays'] + 1)
    
    # 2. Previous Success Rate: poutcome == 'success' (binary)
    # Simplifies the categorical poutcome to just the important signal
    if include_previous_success:
        df['previous_success'] = (df['poutcome'] == 'success').astype(int)
    
    # 3. Financial Stress: count of financial obligations
    # housing='yes' + loan='yes' + default='yes' (0-3 scale)
    if include_financial_stress:
        df['financial_stress'] = (
            (df['housing'] == 'yes').astype(int) + 
            (df['loan'] == 'yes').astype(int) + 
            (df['default'] == 'yes').astype(int)
        )
    
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


def run_experiment(name, **feature_flags):
    """Run a single experiment with given feature configuration."""
    print(f"\n  Evaluating: {name}...")
    X, y = load_and_preprocess(**feature_flags)
    auc_scores = evaluate_model(X, y)
    print(f"    Features: {X.shape[1]}, Mean AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    return {
        'name': name,
        'n_features': X.shape[1],
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'auc_scores': auc_scores
    }


def main():
    print("=" * 70)
    print("ABLATION STUDY: New Engineered Features")
    print("=" * 70)
    print("\nFeatures being tested:")
    print("  1. contact_intensity = campaign / (pdays + 1)")
    print("  2. previous_success = (poutcome == 'success')")
    print("  3. financial_stress = (housing='yes') + (loan='yes') + (default='yes')")
    
    results = []
    
    # Baseline (no new features)
    print("\n[1/5] Baseline (no new features)")
    results.append(run_experiment(
        "Baseline",
        include_contact_intensity=False,
        include_previous_success=False,
        include_financial_stress=False
    ))
    
    # Individual features
    print("\n[2/5] + contact_intensity only")
    results.append(run_experiment(
        "+ contact_intensity",
        include_contact_intensity=True,
        include_previous_success=False,
        include_financial_stress=False
    ))
    
    print("\n[3/5] + previous_success only")
    results.append(run_experiment(
        "+ previous_success",
        include_contact_intensity=False,
        include_previous_success=True,
        include_financial_stress=False
    ))
    
    print("\n[4/5] + financial_stress only")
    results.append(run_experiment(
        "+ financial_stress",
        include_contact_intensity=False,
        include_previous_success=False,
        include_financial_stress=True
    ))
    
    # All features combined
    print("\n[5/5] + ALL new features")
    results.append(run_experiment(
        "+ ALL (combined)",
        include_contact_intensity=True,
        include_previous_success=True,
        include_financial_stress=True
    ))
    
    # --- Results Summary ---
    baseline_auc = results[0]['auc_mean']
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} | {'Features':<10} | {'Mean AUC':<12} | {'Std':<10} | {'vs Baseline':<12}")
    print("-" * 75)
    
    for r in results:
        diff = r['auc_mean'] - baseline_auc
        diff_str = f"{diff:+.4f}" if r['name'] != 'Baseline' else "---"
        print(f"{r['name']:<25} | {r['n_features']:<10} | {r['auc_mean']:<12.4f} | {r['auc_std']:<10.4f} | {diff_str:<12}")
    
    print("-" * 75)
    
    # Find best individual feature
    individual_results = results[1:4]  # contact_intensity, previous_success, financial_stress
    best_individual = max(individual_results, key=lambda x: x['auc_mean'])
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    for r in results[1:]:
        diff = r['auc_mean'] - baseline_auc
        pct = (diff / baseline_auc) * 100
        
        if abs(diff) < 0.001:
            verdict = "NEGLIGIBLE"
        elif diff > 0.003:
            verdict = "STRONG POSITIVE"
        elif diff > 0:
            verdict = "WEAK POSITIVE"
        elif diff < -0.003:
            verdict = "STRONG NEGATIVE"
        else:
            verdict = "WEAK NEGATIVE"
            
        print(f"{r['name']:<25}: {diff:+.4f} ({pct:+.2f}%) - {verdict}")
    
    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    best_overall = max(results, key=lambda x: x['auc_mean'])
    print(f"Best configuration: {best_overall['name']} (AUC: {best_overall['auc_mean']:.4f})")
    print(f"Best individual feature: {best_individual['name']} (AUC: {best_individual['auc_mean']:.4f})")
    
    combined_vs_best_individual = results[4]['auc_mean'] - best_individual['auc_mean']
    print(f"Combined vs Best Individual: {combined_vs_best_individual:+.4f}")
    
    if combined_vs_best_individual > 0.001:
        print("Recommendation: Use ALL features together for best performance.")
    elif best_individual['auc_mean'] > baseline_auc + 0.001:
        print(f"Recommendation: Use only {best_individual['name']} - simpler with similar performance.")
    else:
        print("Recommendation: These features provide minimal benefit for tree-based models.")
    
    # --- Save Results ---
    with open(RESULTS_FILE, 'w') as f:
        f.write("ABLATION STUDY: New Engineered Features\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Features Tested:\n")
        f.write("  1. contact_intensity = campaign / (pdays + 1)\n")
        f.write("  2. previous_success = (poutcome == 'success')\n")
        f.write("  3. financial_stress = (housing='yes') + (loan='yes') + (default='yes')\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  - Model: LightGBM with SMOTE\n")
        f.write(f"  - Cross-Validation: 5-fold Stratified\n")
        f.write(f"  - Random State: {RANDOM_STATE}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Configuration':<25} | {'Features':<10} | {'Mean AUC':<12} | {'Std':<10} | {'vs Baseline':<12}\n")
        f.write("-" * 75 + "\n")
        
        for r in results:
            diff = r['auc_mean'] - baseline_auc
            diff_str = f"{diff:+.4f}" if r['name'] != 'Baseline' else "---"
            f.write(f"{r['name']:<25} | {r['n_features']:<10} | {r['auc_mean']:<12.4f} | {r['auc_std']:<10.4f} | {diff_str:<12}\n")
        
        f.write("-" * 75 + "\n\n")
        
        f.write("Fold-by-Fold Comparison:\n")
        f.write(f"{'Fold':<8}")
        for r in results:
            f.write(f" | {r['name'][:12]:<12}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        
        for i in range(5):
            f.write(f"Fold {i+1:<3}")
            for r in results:
                f.write(f" | {r['auc_scores'][i]:<12.4f}")
            f.write("\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best configuration: {best_overall['name']} (AUC: {best_overall['auc_mean']:.4f})\n")
        f.write(f"Best individual feature: {best_individual['name']} (AUC: {best_individual['auc_mean']:.4f})\n")
        f.write(f"Combined vs Best Individual: {combined_vs_best_individual:+.4f}\n")
    
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
