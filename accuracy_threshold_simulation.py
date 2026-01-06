"""
Accuracy Threshold Simulation - Business Impact Focus (Fixed)

This script simulates different accuracy levels for the Stage 1 duration classifier
to find the threshold at which the efficiency-based scheduling approach yields
MORE SALES than the simple probability-based scheduling.

IMPORTANT: Uses cross-validation predictions to avoid data leakage, matching
the methodology in improved_bank_marketing_v2.py for fair comparison.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
AVG_DURATION_SHORT = 101.3
AVG_DURATION_LONG = 417.0
TIME_LIMIT = 8 * 3600  # 8-hour shift in seconds
RESULTS_FILE = 'accuracy_threshold_results.txt'


def load_and_preprocess(filepath):
    """Load and preprocess the data."""
    df = pd.read_csv(filepath, sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Feature engineering (V2 style)
    if 'campaign' in df.columns and 'pdays' in df.columns:
        df['contact_intensity'] = df['campaign'] / (df['pdays'] + 1)
    
    if all(col in df.columns for col in ['housing', 'loan', 'default']):
        df['financial_stress'] = (
            (df['housing'] == 'yes').astype(int) + 
            (df['loan'] == 'yes').astype(int) + 
            (df['default'] == 'yes').astype(int)
        )
    
    # Duration binning (Short <= 180, Long > 180)
    bins = [-1, 180, float('inf')]
    labels = [0, 1]
    df['duration_class'] = pd.cut(df['duration'], bins=bins, labels=labels).astype(int)
    
    X = df.drop(['y', 'duration', 'duration_class'], axis=1)
    y = df['y']
    duration_target = df['duration_class']
    
    # Encode categorical features
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
    
    return X, y, duration_target, df


def get_real_duration_predictions(X, duration_target):
    """Train duration classifier and get CV predictions (no leakage)."""
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_proba = cross_val_predict(model, X, duration_target, cv=cv, method='predict_proba', n_jobs=1)
    
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    real_accuracy = accuracy_score(duration_target, y_pred_class)
    
    return y_pred_proba, real_accuracy


def simulate_accuracy_level(real_probs, true_labels, target_accuracy, random_state=42):
    """
    Simulate duration predictions at a specified accuracy level by mixing
    real predictions with oracle (perfect) predictions.
    """
    np.random.seed(random_state)
    n_samples = len(true_labels)
    
    real_preds = np.argmax(real_probs, axis=1)
    real_accuracy = accuracy_score(true_labels, real_preds)
    
    if target_accuracy <= real_accuracy:
        return real_probs.copy()
    
    # p_oracle = probability of using oracle prediction
    p_oracle = (target_accuracy - real_accuracy) / (1 - real_accuracy)
    p_oracle = np.clip(p_oracle, 0, 1)
    
    simulated_probs = real_probs.copy()
    use_oracle = np.random.random(n_samples) < p_oracle
    
    for i in range(n_samples):
        if use_oracle[i]:
            if true_labels.iloc[i] == 0:
                simulated_probs[i] = [1.0, 0.0]
            else:
                simulated_probs[i] = [0.0, 1.0]
    
    return simulated_probs


def simulate_business_impact(schedule_df):
    """Simulate 8-hour shift and count calls/sales."""
    total_duration = 0
    calls_made = 0
    sales = 0
    
    for _, row in schedule_df.iterrows():
        duration = row['duration']
        outcome = row['y']
        
        if total_duration + duration > TIME_LIMIT:
            break
        
        total_duration += duration
        calls_made += 1
        sales += outcome
    
    return calls_made, sales


def get_outcome_cv_predictions(X, y):
    """
    Get cross-validated outcome predictions - NO DATA LEAKAGE.
    This matches the methodology in improved_bank_marketing_v2.py.
    """
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
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
    # Use cross_val_predict to get out-of-sample predictions
    prob_success = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
    
    return prob_success


def run_simulation_for_accuracy(X, y, duration_probs, df_original):
    """
    Run business simulation for a given set of duration probabilities.
    Uses CV predictions for fair comparison (no data leakage).
    """
    # --- EFFICIENCY STRATEGY ---
    # 1. Train outcome model WITH duration features using CV predictions
    X_enhanced = X.copy()
    X_enhanced['prob_short'] = duration_probs[:, 0]
    X_enhanced['prob_long'] = duration_probs[:, 1]
    
    prob_success_enhanced = get_outcome_cv_predictions(X_enhanced, y)
    
    # 2. Calculate efficiency score and create schedule
    schedule_eff = df_original.copy()
    schedule_eff['prob_short'] = duration_probs[:, 0]
    schedule_eff['prob_long'] = duration_probs[:, 1]
    schedule_eff['prob_success'] = prob_success_enhanced
    schedule_eff['expected_duration'] = (
        schedule_eff['prob_short'] * AVG_DURATION_SHORT +
        schedule_eff['prob_long'] * AVG_DURATION_LONG
    )
    schedule_eff['efficiency_score'] = schedule_eff['prob_success'] / schedule_eff['expected_duration']
    schedule_eff = schedule_eff.sort_values('efficiency_score', ascending=False)
    
    eff_calls, eff_sales = simulate_business_impact(schedule_eff)
    
    # --- PROBABILITY STRATEGY ---
    # Train outcome model WITHOUT duration features using CV predictions
    prob_success_base = get_outcome_cv_predictions(X, y)
    
    schedule_prob = df_original.copy()
    schedule_prob['prob_success'] = prob_success_base
    schedule_prob = schedule_prob.sort_values('prob_success', ascending=False)
    
    prob_calls, prob_sales = simulate_business_impact(schedule_prob)
    
    return eff_calls, eff_sales, prob_calls, prob_sales


def main():
    print("=" * 70)
    print("ACCURACY THRESHOLD SIMULATION - BUSINESS IMPACT (FIXED)")
    print("Using Cross-Validation Predictions (No Data Leakage)")
    print("=" * 70)
    
    filepath = 'data/bank-additional/bank-additional-full.csv'
    X, y, duration_target, df = load_and_preprocess(filepath)
    
    print(f"\nDataset: {len(y)} samples")
    print(f"Total sales in dataset: {y.sum()}")
    
    # Get real model predictions
    print("\n--- Getting Real Duration Model Predictions ---")
    real_probs, real_accuracy = get_real_duration_predictions(X, duration_target)
    print(f"Real duration model accuracy: {real_accuracy:.2%}")
    
    # First, get baseline (probability-only) results - this is constant
    print("\n--- Computing Baseline (Probability-Only) Results ---")
    prob_success_base = get_outcome_cv_predictions(X, y)
    schedule_prob = df.copy()
    schedule_prob['prob_success'] = prob_success_base
    schedule_prob = schedule_prob.sort_values('prob_success', ascending=False)
    baseline_calls, baseline_sales = simulate_business_impact(schedule_prob)
    print(f"Baseline: {baseline_sales} sales from {baseline_calls} calls")
    
    # Test different accuracy levels
    accuracy_levels = [0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    results = []
    
    print("\n--- Simulating Different Accuracy Levels ---")
    print("(Running business simulation for each level - this may take a few minutes...)\n")
    
    for target_acc in accuracy_levels:
        print(f"[{target_acc:.0%} Accuracy]", end=" ", flush=True)
        
        if target_acc >= 0.999:
            # Perfect oracle
            simulated_probs = np.zeros((len(duration_target), 2))
            for i, label in enumerate(duration_target):
                if label == 0:
                    simulated_probs[i] = [1.0, 0.0]
                else:
                    simulated_probs[i] = [0.0, 1.0]
            actual_acc = 1.0
        else:
            simulated_probs = simulate_accuracy_level(real_probs, duration_target, target_acc)
            sim_preds = np.argmax(simulated_probs, axis=1)
            actual_acc = accuracy_score(duration_target, sim_preds)
        
        # Run simulation for efficiency strategy only (baseline is constant)
        X_enhanced = X.copy()
        X_enhanced['prob_short'] = simulated_probs[:, 0]
        X_enhanced['prob_long'] = simulated_probs[:, 1]
        
        prob_success_enhanced = get_outcome_cv_predictions(X_enhanced, y)
        
        schedule_eff = df.copy()
        schedule_eff['prob_short'] = simulated_probs[:, 0]
        schedule_eff['prob_long'] = simulated_probs[:, 1]
        schedule_eff['prob_success'] = prob_success_enhanced
        schedule_eff['expected_duration'] = (
            schedule_eff['prob_short'] * AVG_DURATION_SHORT +
            schedule_eff['prob_long'] * AVG_DURATION_LONG
        )
        schedule_eff['efficiency_score'] = schedule_eff['prob_success'] / schedule_eff['expected_duration']
        schedule_eff = schedule_eff.sort_values('efficiency_score', ascending=False)
        
        eff_calls, eff_sales = simulate_business_impact(schedule_eff)
        
        sales_diff = eff_sales - baseline_sales
        calls_diff = eff_calls - baseline_calls
        
        results.append({
            'target_accuracy': target_acc,
            'actual_accuracy': actual_acc,
            'efficiency_calls': eff_calls,
            'efficiency_sales': eff_sales,
            'probability_calls': baseline_calls,
            'probability_sales': baseline_sales,
            'sales_diff': sales_diff,
            'calls_diff': calls_diff,
            'efficiency_wins': eff_sales > baseline_sales
        })
        
        status = "✓" if eff_sales > baseline_sales else "✗"
        print(f"Eff: {eff_sales} sales ({eff_calls} calls) | Prob: {baseline_sales} sales ({baseline_calls} calls) | Diff: {sales_diff:+d} {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Accuracy':<10} | {'Efficiency':<18} | {'Probability':<18} | {'Difference':<15} | {'Winner'}")
    print(f"{'':10} | {'Calls':>7} {'Sales':>9} | {'Calls':>7} {'Sales':>9} | {'Calls':>6} {'Sales':>7} |")
    print("-" * 85)
    
    threshold_found = None
    
    for r in results:
        winner = "EFFICIENCY" if r['efficiency_wins'] else "PROBABILITY"
        acc_str = f"{r['target_accuracy']:.0%}"
        print(f"{acc_str:<10} | {r['efficiency_calls']:>7} {r['efficiency_sales']:>9} | {r['probability_calls']:>7} {r['probability_sales']:>9} | {r['calls_diff']:>+6} {r['sales_diff']:>+7} | {winner}")
        
        if r['efficiency_wins'] and threshold_found is None:
            threshold_found = r['target_accuracy']
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if threshold_found:
        print(f"\n🎯 THRESHOLD FOUND: ~{threshold_found:.0%} Stage 1 Accuracy")
        print(f"\nThe efficiency-based scheduling approach starts beating the")
        print(f"probability-only approach at approximately {threshold_found:.0%} duration classifier accuracy.")
        print(f"\nCurrent real accuracy: {real_accuracy:.2%}")
        print(f"Gap to threshold: {(threshold_found - real_accuracy)*100:.1f} percentage points")
    else:
        print("\n❌ Efficiency-based approach never beats probability-only in tested range.")
        print("This suggests the current two-stage approach needs higher accuracy to be effective.")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('accuracy_threshold_simulation_results.csv', index=False)
    
    # Detailed report
    with open(RESULTS_FILE, 'w') as f:
        f.write("ACCURACY THRESHOLD SIMULATION - BUSINESS IMPACT (FIXED)\n")
        f.write("Using Cross-Validation Predictions (No Data Leakage)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Goal: Find Stage 1 accuracy where efficiency scheduling beats probability scheduling\n")
        f.write(f"Time Budget: {TIME_LIMIT//3600} hours ({TIME_LIMIT:,} seconds)\n\n")
        
        f.write(f"Real Duration Model Accuracy: {real_accuracy:.2%}\n")
        f.write(f"Baseline (Prob-Only): {baseline_sales} sales from {baseline_calls} calls\n\n")
        
        f.write("Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Accuracy':<10} | {'Eff Sales':<10} | {'Prob Sales':<11} | {'Diff':<10} | {'Winner'}\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            winner = "EFFICIENCY" if r['efficiency_wins'] else "PROBABILITY"
            acc_str = f"{r['target_accuracy']:.0%}"
            f.write(f"{acc_str:<10} | {r['efficiency_sales']:<10} | {r['probability_sales']:<11} | {r['sales_diff']:+<10} | {winner}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 70 + "\n\n")
        
        if threshold_found:
            f.write(f"Threshold: ~{threshold_found:.0%}\n")
            f.write(f"Current Accuracy: {real_accuracy:.2%}\n")
            f.write(f"Gap: {(threshold_found - real_accuracy)*100:.1f} percentage points\n")
        else:
            f.write("Efficiency approach never beats probability-only in tested range.\n")
            f.write("The two-stage approach requires higher duration prediction accuracy.\n")
    
    print(f"\nResults saved to: accuracy_threshold_simulation_results.csv")
    print(f"Report saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
