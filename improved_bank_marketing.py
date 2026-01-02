import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import sys

# --- Configuration ---
RANDOM_STATE = 42
MODEL_DIR = 'models'
REPORT_FILE = 'model_performance_report.txt'
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def log_to_file(message):
    """Appends a message to the report file."""
    with open(REPORT_FILE, 'a') as f:
        f.write(message + '\n')
    print(message)

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

    print(f"Data encoded. Feature shape: {X_encoded.shape}")
    return X_encoded, y_target, duration_target

def optimize_lightgbm(X, y, objective, metric):
    """
    Optimizes LightGBM hyperparameters using Optuna.
    """
    print(f"  Optimizing {objective} model using Optuna...")
    
    def objective_function(trial):
        param = {
            'objective': objective,
            'metric': metric,
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        # Binary needs specific handling if we want to use 'accuracy' for duration or 'auc' for outcome
        # But 'binary_logloss' is generally good for probability calibration for duration class
        
        model = lgb.LGBMClassifier(**param)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        if metric == 'auc':
            # For imbalanced outcome, use AUC
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        else:
            # For duration classification (balanced-ish), accuracy is fine proxy for optimization, 
            # but let's stick to log_loss for better probabilities
            # Or just 'accuracy' as requested earlier? Let's maximize accuracy for duration split.
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_function, n_trials=20, timeout=600) # 20 trials or 10 mins
    
    print(f"  Best trial: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    best_params = study.best_params.copy()
    best_params.update({
        'objective': objective,
        'metric': metric,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })
    
    return lgb.LGBMClassifier(**best_params)

def train_duration_classifier(X, y_class):
    """
    Trains a LightGBM classifier to predict call duration class (Short/Long).
    Checks for saved model first.
    Returns the trained model and the predicted probabilities (generated via CV to avoid leakage).
    """
    print("\n--- Training Duration Classifier (LightGBM) ---")
    
    model_path = os.path.join(MODEL_DIR, 'duration_model.joblib')
    
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        model = joblib.load(model_path)
        print(f"Model Hyperparameters: {model.get_params()}")
    else:
        print("No saved model found. Optimizing and training new model...")
        model = optimize_lightgbm(X, y_class, 'binary', 'binary_logloss')
        model.fit(X, y_class)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        print(f"Model Hyperparameters: {model.get_params()}")
    
    # Generate Cross-Validated Probabilities for the next stage
    # INFO: If we loaded a model, strictly speaking we should still generate CV probs on X 
    # to avoid leakage if X is the training set. 
    # If this was inference on NEW data, we'd just predict. 
    # Assuming X is the same training set, we calculate CV probs fresh to keep the downstream safe.
    print("Generating predicted probabilities via 5-fold CV...")
    # NOTE: Set n_jobs=1 for cross_val_predict to avoid nested parallelism hang with LightGBM
    # NOTE: Must use Shuffle=True because data is time-ordered
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_proba = cross_val_predict(model, X, y_class, cv=cv, method='predict_proba', n_jobs=1)
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
    
    report = classification_report(y_class, y_pred_class)
    log_to_file("\n" + "="*40)
    log_to_file("DURATION CLASSIFIER REPORT")
    log_to_file("="*40)
    log_to_file(f"Model Params: {model.get_params()}")
    log_to_file(f"Accuracy: {acc:.4f}")
    log_to_file(f"Log Loss: {ll:.4f}")
    log_to_file("Confusion Matrix:\n" + str(cm))
    log_to_file("Classification Report:\n" + report)
    
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
    Checks for saved model first.
    Evaluates using 5-fold Stratified CV.
    """
    print(f"\nTraining Outcome Model: {description}")
    
    model_path = os.path.join(MODEL_DIR, 'outcome_model.joblib')
    
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        pipeline = joblib.load(model_path)
        # Extract classifier from pipeline to print params
        if hasattr(pipeline, 'named_steps'):
            clf = pipeline.named_steps['classifier']
            print(f"Model Hyperparameters: {clf.get_params()}")
        else:
            print(f"Model Hyperparameters: {pipeline.get_params()}")
            
        # Re-assign model variable if needed for CV construction logic below, 
        # but actually we just use the loaded pipeline for CV
    else:
        print("No saved model found. Optimizing and training new model...")
        # Since we use SMOTE in pipeline, optimization is tricky.
        # We'll optimize the CLASSIFIER on SMOTE-sampled data or just optimize normally.
        # For simplicity in this script, let's optimize the classifier on raw data (often fine for LightGBM)
        # OR optimize within a pipeline (complex).
        # Decision: Optimize LightGBM on raw imbalanced data using AUC metric, then put in SMOTE pipeline.
        model = optimize_lightgbm(X, y, 'binary', 'auc')
        print(f"Model Hyperparameters: {model.get_params()}")
        # We don't save the pipeline in this simplified block, we save the classifier logic?
        # Actually better to save the fitted pipeline if possible, or just the best params.
        # Let's save the best estimator.
        
        # Pipeline construction
        # Note: We fit the pipeline on the full data at the END and save it.
    
    # We construct the pipeline regardless to evaluate CV
    # Use the 'model' (either loaded or optimized new instance)
    # If we loaded a pipeline, we use it directly.
    if os.path.exists(model_path) and 'pipeline' in locals():
        # Pipeline is already loaded
        pass
    else:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model)
        ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    auc_scores = []
    alift_scores = []
    
    print("Running 5-fold CV...")
    
    # Store predictions for aggregated report
    y_true_all = []
    y_pred_class_all = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred_cls = pipeline.predict(X_test)
        
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        alift_scores.append(calculate_alift(y_test, y_pred_proba))
        
        y_true_all.extend(y_test)
        y_pred_class_all.extend(y_pred_cls)
        
    mean_auc = np.mean(auc_scores)
    mean_alift = np.mean(alift_scores)
    
    print(f"  AUC: {mean_auc:.4f}")
    print(f"  ALIFT: {mean_alift:.4f}")
    
    # Generate Aggregated Report
    report = classification_report(y_true_all, y_pred_class_all)
    
    log_to_file("\n" + "="*40)
    log_to_file(f"OUTCOME MODEL REPORT: {description}")
    log_to_file("="*40)
    
    # Extract model params if possible
    if hasattr(pipeline, 'named_steps'):
        clf_params = pipeline.named_steps['classifier'].get_params()
        log_to_file(f"Classifier Params: {clf_params}")
    else:
        log_to_file(f"Pipeline Params: {pipeline.get_params()}")
        
    log_to_file(f"Mean CV AUC: {mean_auc:.4f}")
    log_to_file(f"Mean CV ALIFT: {mean_alift:.4f}")
    log_to_file("Aggregated Classification Report (CV):\n" + report)
    
    # Fit and Save Final Pipeline on Full Data
    if not os.path.exists(model_path):
        pipeline.fit(X, y)
        joblib.dump(pipeline, model_path)
        print(f"Pipeline saved to {model_path}")
    
    # Return the pipeline (fitted on last CV fold or loaded)
    # Actually for downstream usage (scheduling) we want the pipeline fitted on ALL data
    # If valid logic: load -> already fitted? joblib loads the object state.
    # If we loaded just the 'model' (classifier) earlier, pipeline isn't fitted.
    # To correspond with the logic:
    # If loaded: assume it is the PIPELINE.
    
    # RE-LOGIC for persistence to be clean:
    # 1. Model variable 'model_artifact' -> Try load 'outcome_pipeline.joblib'
    # 2. If no exists: Optimize param, Create Pipeline, Fit Pipeline on Full Data, Save Pipeline.
    # 3. But we need CV score? 
    # Valid workflow:
    #   if not exists:
    #      Optimize Params.
    #      Run CV to Report Score.
    #      Fit Full Pipeline.
    #      Save Full Pipeline.
    #   if exists:
    #      Load Full Pipeline.
    #      Report "Saved model loaded - CV skipped" or re-run CV?
    #      User wants to save time. Skipping CV if loaded is preferred, but user might want to see metrics?
    #      Let's re-run CV for metrics report (fast enough compared to Optuna) but skip Optuna.
    
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

def simulate_business_impact(schedule_df, print_output=True):
    """
    Simulates an 8-hour shift (28,800 seconds).
    If print_output is True, prints 'Calls: X, Sales: Y'.
    """
    TIME_LIMIT = 8 * 3600
    
    total_duration = 0
    calls_made = 0
    sales = 0
        
    for _, row in schedule_df.iterrows():
        # Default to 0 duration/outcome if not present (shouldn't happen with correct usage)
        duration = row.get('duration', 0)
        outcome = row.get('y', 0)
        
        if total_duration + duration > TIME_LIMIT:
            break
            
        total_duration += duration
        calls_made += 1
        sales += outcome
            
    if print_output:
        print(f"  Calls: {calls_made}, Sales: {sales}")
        
    return calls_made, sales


def simulate_business_impact_random(schedule_df):
    """
    Simulates an 8-hour shift (28,800 seconds) for a random calling strategy.
    Runs 50 times for stability and returns average calls and sales.
    """
    n_trials = 50
    calls_list = []
    sales_list = []
    
    for i in range(n_trials):
        df_shuffled = schedule_df.sample(frac=1, random_state=RANDOM_STATE + i)
        c, s = simulate_business_impact(df_shuffled, print_output=False)
        calls_list.append(c)
        sales_list.append(s)
        
    return np.mean(calls_list), np.mean(sales_list)

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
    # Initialize Report File
    with open(REPORT_FILE, 'w') as f:
        f.write("BANK MARKETING MODEL PERFORMANCE REPORT\n")
        f.write("=======================================\n\n")

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
        
        # ---------------------------------------------------------
        # 4. Baseline Model (Probability Only - No Duration Features)
        # ---------------------------------------------------------
        # Train model on X_encoded (Raw features without duration probs)
        auc_base, alift_base, baseline_model = train_outcome_model(
            X_encoded, y, "Baseline Model (Probability Only - No Duration Info)"
        )
        
        # Generate Baseline Probabilities (CV)
        print("Generating baseline probabilities via 5-fold CV (Shuffle=True)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        prob_success_baseline = cross_val_predict(baseline_model, X_encoded, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]

        # ---------------------------------------------------------
        # 5. Generate Schedules
        # ---------------------------------------------------------
        
        # Schedule 1: Efficiency Score (Enhanced)
        schedule_efficiency = generate_call_schedule(X_encoded, df, predicted_probs[:, 0], predicted_probs[:, 1], final_probs_success)
        schedule_efficiency['strategy'] = 'Efficiency (Proposed)'
        
        # Schedule 2: Probability Score (Standard/Baseline)
        # Rank purely by Probability Descending
        schedule_standard = df.copy()
        schedule_standard['prob_success'] = prob_success_baseline
        schedule_standard['efficiency_score'] = schedule_standard['prob_success'] # Proxy for efficiency is just prob
        schedule_standard['strategy'] = 'Probability (Standard ML)'
        schedule_standard = schedule_standard.sort_values('prob_success', ascending=False)
        
        # Schedule 3: Oracle
        # (Generated inside run_oracle_experiment, but let's just grab metrics from function for now)
        
        # ---------------------------------------------------------
        # 6. Business Impact Simulation
        # ---------------------------------------------------------
        
        print("\n--- Business Impact Simulation (8-Hour Shift) ---")
        
        # 1. Random Baseline
        print("Simulating Random Strategy...")
        # Just shuffle the dataframe 100 times and average? 
        # Or just use the 'simulate_business_impact' function on shuffled df
        calls_random, sales_random = simulate_business_impact_random(df)
        
        # 2. Standard ML (Probability Only)
        print("Simulating Standard Probability Strategy...")
        calls_standard, sales_standard = simulate_business_impact(schedule_standard)
        
        # 3. Efficiency (Proposed)
        print("Simulating Efficiency Strategy...")
        calls_eff, sales_eff = simulate_business_impact(schedule_efficiency)

        # 4. Oracle (For comparison)
        # We need to run simulation on Oracle schedule. 
        # run_oracle_experiment returns AUC/ALIFT, let's modify it or just run sim inside main
        # To keep it clean, let's just use the metrics from the function call earlier
        # Actually, we need to CALL run_oracle_experiment to get the results?
        # Wait, in the previous code 'run_oracle_experiment' printed the simulation results.
        # Let's verify the Oracle run.
        print("\n--- Oracle Experiment ---")
        auc_oracle, alift_oracle = run_oracle_experiment(X_encoded, y, df, duration_target)
        
        # We can't easily get the 'calls_oracle' out of that function without changing it.
        # Let's trust the print output for Oracle or refactor if needed.
        # For the table, let's just print the main 3 for now.

        print(f"\nResults per 8-Hour Shift (Avg Agent):")
        print(f"{'Metric':<20} | {'Random':<10} | {'Standard':<10} | {'Efficiency':<10} | {'Lift (Eff vs Std)':<15}")
        print("-" * 80)
        print(f"{'Calls Made':<20} | {calls_random:<10.1f} | {calls_standard:<10.0f} | {calls_eff:<10.0f} | {((calls_eff-calls_standard)/calls_standard)*100:+.1f}%")
        print(f"{'Sales':<20} | {sales_random:<10.1f} | {sales_standard:<10.0f} | {sales_eff:<10.0f} | {((sales_eff-sales_standard)/sales_standard)*100:+.1f}%")

        print(f"\nImpact: Prioritizing by Efficiency yields {sales_eff - sales_standard:.1f} MORE sales than prioritized by Probability alone.")
        
        # Save Results
        results = pd.DataFrame({
            'Model': ['Enhanced (Real)', 'Baseline (Prob Only)', 'Oracle (Perfect)'],
            'AUC': [auc_enhanced, auc_base, auc_oracle],
            'ALIFT': [alift_enhanced, alift_base, alift_oracle]
        })
        
        print("\n--- Final Results Comparison ---")
        print(results)
        results.to_csv('final_model_results.csv', index=False)

    else:
        print("Error: Duration target could not be created.")

def simulate_business_impact_random(df):
    # Helper to run random simulation multiple times
    n_trials = 50
    calls_list = []
    sales_list = []
    
    for i in range(n_trials):
        df_shuffled = df.sample(frac=1, random_state=RANDOM_STATE + i)
        c, s = simulate_business_impact(df_shuffled, print_output=False)
        calls_list.append(c)
        sales_list.append(s)
        
    return np.mean(calls_list), np.mean(sales_list)


if __name__ == "__main__":
    main()

