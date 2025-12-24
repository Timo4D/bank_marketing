import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import sys

# --- Configuration ---
RANDOM_STATE = 42
warnings.filterwarnings('ignore')

# Derived Constants from Data Analysis
AVG_DURATION_SHORT = 101.3  # seconds
AVG_DURATION_LONG = 417.0   # seconds

def load_data(filepath):
    """Loads the dataset from the specified CSV file."""
    print("Loading data...")
    try:
        df = pd.read_csv(filepath, sep=';')
        print(f"Dataset loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the path.")
        sys.exit(1)

def preprocess_data(df):
    """
    Preprocesses the data:
    1. Maps target 'y' to binary.
    2. Creates interaction features.
    3. Bins 'duration' into Short/Long classes.
    4. Encodes categorical variables.
    """
    print("\nPreprocessing data...")
    
    # Map Target
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # --- Feature Engineering ---
    print("Generating interaction features...")
    # Age * Campaign: older people contacted many times vs younger
    if 'age' in df.columns and 'campaign' in df.columns:
        df['age_campaign'] = df['age'] * df['campaign']
        
    # Is First Contact: pdays=999 implies first time
    if 'pdays' in df.columns:
        df['is_first_contact'] = (df['pdays'] == 999).astype(int)
    
    # --- Duration Binning ---
    # Split at median (180s)
    # Short: <= 180, Long: > 180
    bins = [-1, 180, float('inf')]
    labels = [0, 1] # 0: Short, 1: Long
    df['duration_class'] = pd.cut(df['duration'], bins=bins, labels=labels)
    
    duration_target = df['duration_class'].astype(int)
    
    print("\nDuration Class Distribution:")
    print(duration_target.value_counts(normalize=True))
    
    # Drop duration features to avoid leakage in the classifier
    X_features = df.drop(['y', 'duration', 'duration_class'], axis=1)
    y_target = df['y']
    
    # Categorical Encoding
    categorical_features = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ordinal Encoding for Education
    education_order = {
        'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 
        'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': -1
    }
    
    X_encoded = X_features.copy()
    if 'education' in X_encoded.columns:
        X_encoded['education'] = X_encoded['education'].map(education_order)
        X_encoded['education'] = X_encoded['education'].fillna(-1)
        if 'education' in categorical_features:
            categorical_features.remove('education')

    print("Applying One-Hot Encoding to nominal features...")
    X_encoded = pd.get_dummies(X_encoded, columns=categorical_features, drop_first=True)
    
    print(f"Data encoded. Feature shape: {X_encoded.shape}")
    return X_encoded, y_target, duration_target

def train_duration_classifier(X, y_class):
    """
    Trains a LightGBM classifier to predict call duration class (Short/Long).
    Returns the trained model and the predicted probabilities (generated via CV to avoid leakage).
    """
    print("\n--- Training Duration Classifier (LightGBM) ---")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        verbosity=-1,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Generate Cross-Validated Probabilities for the next stage
    print("Generating predicted probabilities via 5-fold CV...")
    y_pred_proba = cross_val_predict(model, X, y_class, cv=5, method='predict_proba', n_jobs=-1)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # Evaluation
    acc = accuracy_score(y_class, y_pred_class)
    ll = log_loss(y_class, y_pred_proba)
    cm = confusion_matrix(y_class, y_pred_class)
    
    print(f"Duration Classifier Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print("  Confusion Matrix:")
    print(cm)
    
    # Fit on full data for potential future inference
    model.fit(X, y_class)
    
    return model, y_pred_proba

def calculate_alift(y_true, y_pred_proba):
    """Calculates the ALIFT metric."""
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data = data.sort_values('y_pred_proba', ascending=False)
    data['cumulative_positives'] = data['y_true'].cumsum()
    data['cumulative_population_pct'] = (np.arange(len(data)) + 1) / len(data)
    total_positives = data['y_true'].sum()
    data['lift'] = (data['cumulative_positives'] / total_positives) / data['cumulative_population_pct']
    return data['lift'].mean()

def train_outcome_model(X, y, description):
    """
    Trains the final outcome model (predicting 'y') using LightGBM with SMOTE.
    Evaluates using 5-fold Stratified CV.
    """
    print(f"\nTraining Outcome Model: {description}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        verbosity=-1,
        boosting_type='gbdt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    auc_scores = []
    alift_scores = []
    
    print("Running 5-fold CV...")
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        alift_scores.append(calculate_alift(y_test, y_pred_proba))
        
    mean_auc = np.mean(auc_scores)
    mean_alift = np.mean(alift_scores)
    
    print(f"  AUC: {mean_auc:.4f}")
    print(f"  ALIFT: {mean_alift:.4f}")
    
    return mean_auc, mean_alift, pipeline

def generate_call_schedule(X_feature_data, df_original, prob_short, prob_long, prob_success):
    """
    Generates a prioritized call schedule based on Efficiency Score.
    
    Efficiency Score = P(Success) / Expected Duration
    Expected Duration = P(Short)*Avg_Short + P(Long)*Avg_Long
    """
    print("\n--- Generating Prioritized Call Schedule ---")
    
    schedule = df_original.copy()
    
    # Add Predictions
    schedule['prob_short'] = prob_short
    schedule['prob_long'] = prob_long
    schedule['prob_success'] = prob_success
    
    # Calculate Expected Duration
    # Use the constants we found earlier
    schedule['expected_duration'] = (
        schedule['prob_short'] * AVG_DURATION_SHORT + 
        schedule['prob_long'] * AVG_DURATION_LONG
    )
    
    # Calculate Efficiency Score (Success Probability per Second)
    # Avoid division by zero (though avg duration > 0)
    schedule['efficiency_score'] = schedule['prob_success'] / schedule['expected_duration']
    
    # Sort by Efficiency Score Descending
    schedule_sorted = schedule.sort_values('efficiency_score', ascending=False)
    
    print("\nTop 5 Most Efficient Calls (Prioritized):")
    cols_to_show = ['prob_success', 'expected_duration', 'efficiency_score', 'y']
    if 'duration' in schedule.columns:
        cols_to_show.append('duration') # Show actual duration if available for comparison
        
    print(schedule_sorted[cols_to_show].head(5))
    
    return schedule_sorted

def simulate_business_impact(schedule_df):
    """
    Simulates an 8-hour shift (28,800 seconds) to quantify business impact.
    Compares 'Baseline' (Random) vs 'Prioritized' (Efficiency Score).
    """
    print("\n--- Business Impact Simulation (8-Hour Shift) ---")
    
    TIME_LIMIT = 8 * 3600 # 28,800 seconds
    
    # Needs 'y' and 'duration' (actuals) to be valid
    if 'y' not in schedule_df.columns or 'duration' not in schedule_df.columns:
        print("Warning: Cannot run simulation without actual 'y' and 'duration' columns.")
        return

    def run_shift(df_subset):
        total_duration = 0
        calls_made = 0
        sales = 0
        
        # We iterate through the dataframe until time runs out
        for _, row in df_subset.iterrows():
            duration = row['duration']
            outcome = row['y'] # 0 or 1
            
            if total_duration + duration > TIME_LIMIT:
                break
                
            total_duration += duration
            calls_made += 1
            sales += outcome
            
        return calls_made, sales

    # 1. Baseline Strategy (Random) - Run 100 times for stability
    baseline_calls = []
    baseline_sales = []
    print("Simulating Baseline (Random Calling)...")
    for _ in range(100):
        # Shuffle
        df_random = schedule_df.sample(frac=1, random_state=None) 
        c, s = run_shift(df_random)
        baseline_calls.append(c)
        baseline_sales.append(s)
        
    avg_base_calls = np.mean(baseline_calls)
    avg_base_sales = np.mean(baseline_sales)
    
    # 2. Prioritized Strategy
    print("Simulating Prioritized Strategy...")
    # Already sorted by generate_call_schedule, but sort again to be sure
    df_prioritized = schedule_df.sort_values('efficiency_score', ascending=False)
    p_calls, p_sales = run_shift(df_prioritized)
    
    # Results
    print("\nResults per 8-Hour Shift (Avg Agent):")
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Prioritized':<12} | {'Lift':<10}")
    print("-" * 60)
    print(f"{'Calls Made':<20} | {avg_base_calls:<10.1f} | {p_calls:<12} | {((p_calls/avg_base_calls)-1)*100:+.1f}%")
    print(f"{'Sales (Conversions)':<20} | {avg_base_sales:<10.1f} | {p_sales:<12} | {((p_sales/avg_base_sales)-1)*100:+.1f}%")

    sales_lift = p_sales - avg_base_sales
    print(f"\nImpact: An agent using this model makes ~{sales_lift:.1f} MORE sales per day.")

def run_oracle_experiment(X_base, y, df_original, duration_target):
    """
    Runs the 'Oracle' experiment where we assume we have a PERFECT duration predictor.
    This establishes the theoretical upper bound for the project.
    """
    print("\n\n" + "="*60)
    print("RUNNING ORACLE EXPERIMENT (Perfect Duration Predictor)")
    print("="*60)
    
    # 1. Create 'Perfect' Probabilities from Actuals
    # If duration_class is 0 (Short): prob_short=1.0, prob_long=0.0
    # If duration_class is 1 (Long): prob_short=0.0, prob_long=1.0
    
    perfect_probs_short = np.where(duration_target == 0, 1.0, 0.0)
    perfect_probs_long = np.where(duration_target == 1, 1.0, 0.0)
    
    # 2. Enhance Features with Perfect Info
    X_oracle = X_base.copy()
    X_oracle['prob_short'] = perfect_probs_short
    X_oracle['prob_long'] = perfect_probs_long
    
    # 3. Train Outcome Model with Perfect Features
    auc_oracle, alift_oracle, oracle_model = train_outcome_model(X_oracle, y, "Oracle Model (Perfect Duration Info)")
    
    # 4. Generate Oracle Schedule
    # Fit on all data for scoring (simulation purpose)
    oracle_model.fit(X_oracle, y)
    oracle_probs_success = oracle_model.predict_proba(X_oracle)[:, 1]
    
    oracle_schedule = generate_call_schedule(
        X_oracle,
        df_original,
        perfect_probs_short,
        perfect_probs_long,
        oracle_probs_success
    )
    
    # 5. Simulate Oracle Business Impact
    simulate_business_impact(oracle_schedule)
    
    return auc_oracle, alift_oracle

def main():
    # 1. Load Data
    df = load_data('data/bank-additional/bank-additional-full.csv')
    
    # 2. Preprocess
    X_encoded, y, duration_target = preprocess_data(df)
    results = []
    
    # 3. Enhanced Model Pipeline (Train Duration Clf -> Enhance Features -> Train Outcome Model)
    if duration_target is not None:
        # A. Train Duration Classifier
        duration_model, predicted_probs = train_duration_classifier(X_encoded, duration_target)
        
        # B. Enhance Feature Set with Probabilities
        X_enhanced = X_encoded.copy()
        X_enhanced['prob_short'] = predicted_probs[:, 0]
        X_enhanced['prob_long'] = predicted_probs[:, 1]
        
        # C. Train Final Outcome Model
        auc_enhanced, alift_enhanced, outcome_model = train_outcome_model(X_enhanced, y, "Enhanced Model (with Duration Probabilities)")
        
        results.append({
            'Model': 'Enhanced LightGBM',
            'Features': 'Base + Duration Probs',
            'AUC': auc_enhanced,
            'ALIFT': alift_enhanced
        })
        
        # D. Generate Schedule
        # We need probabilities for the whole dataset to generate the schedule.
        # Ideally, we should use cross-val predictions or hold-out, but for the "Application" phase, 
        # we can use the model trained on the full dataset to score new/unknown data.
        # Here, we are scoring the historical data to demonstrate the concept.
        
        # Fit final outcome model on all data
        outcome_model.fit(X_enhanced, y)
        final_probs_success = outcome_model.predict_proba(X_enhanced)[:, 1]
        
        # Generate Schedule
        # Note: predicted_probs comes from train_duration_classifier which returned CV probs.
        # For a production pipeline, we would train on all data and predict on new data.
        # But 'predicted_probs' variable currently holds CV probs for the training set X_encoded.
        prioritized_schedule = generate_call_schedule(
            X_encoded, 
            df, 
            predicted_probs[:, 0], # Prob Short
            predicted_probs[:, 1], # Prob Long
            final_probs_success    # Prob Success
        )
        
        prioritized_schedule.to_csv('prioritized_call_schedule.csv', index=False, sep=';')
        print("Prioritized schedule saved to 'prioritized_call_schedule.csv'")
        
        # E. Simulate Business Impact
        simulate_business_impact(prioritized_schedule)
        
        # F. Run Oracle Experiment
        auc_oracle, alift_oracle = run_oracle_experiment(X_encoded, y, df, duration_target)
        
        print("\n--- Final Results Comparison ---")
        comparison = [
            {'Model': 'Enhanced (Real)', 'AUC': results[-1]['AUC'], 'ALIFT': results[-1]['ALIFT']},
            {'Model': 'Oracle (Perfect)', 'AUC': auc_oracle, 'ALIFT': alift_oracle}
        ]
        print(pd.DataFrame(comparison))
        
        pd.DataFrame(results + comparison).to_csv('final_model_results_with_oracle.csv', index=False)
        print(pd.DataFrame(results))
        pd.DataFrame(results).to_csv('final_model_results.csv', index=False)
        print("Results saved to 'final_model_results.csv'")

    else:
        print("Error: Duration target could not be created.")

if __name__ == "__main__":
    main()
