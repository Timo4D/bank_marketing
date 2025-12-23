import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import sys

# Suppress warnings and Optuna logs
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random state
RANDOM_STATE = 42

def load_data(filepath):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath, sep=';')
        print(f"Dataset loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the path.")
        sys.exit(1)

def preprocess_data(df):
    print("\nPreprocessing data...")
    
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
    # Short: <= 180, Long: > 180
    bins = [-1, 180, float('inf')]
    labels = [0, 1] # 0: Short, 1: Long
    df['duration_class'] = pd.cut(df['duration'], bins=bins, labels=labels)
    
    duration_target = df['duration_class'].astype(int)
    
    print("\nDuration Class Distribution:")
    print(duration_target.value_counts(normalize=True))
    
    # Drop duration from features
    X_features = df.drop(['y', 'duration', 'duration_class'], axis=1)
    y_target = df['y']
    
    categorical_features = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
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

def compare_duration_models(X, y_class):
    print("\n--- Comparing Duration Classifiers (Fast Validation) ---")
    
    # Split for fast validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class)
    
    # Scaling for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        'LightGBM': lgb.LGBMClassifier(
            objective='binary', 
            metric='binary_logloss', 
            verbosity=-1, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ),
        'SVM (SGD)': SGDClassifier(
            loss='log_loss', 
            penalty='l2', 
            alpha=0.0001, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
    }
    
    results = {}
    best_score = -1
    best_model_name = None
    best_model = None
    best_is_scaled = False
    
    for name, model in models.items():
        print(f"Training {name}...")
        is_scaled = name == 'SVM (SGD)'
        
        X_t = X_train_scaled if is_scaled else X_train
        X_v = X_val_scaled if is_scaled else X_val
        
        model.fit(X_t, y_train)
        y_pred = model.predict(X_v)
        y_proba = model.predict_proba(X_v)
        
        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_proba)
        
        print(f"  {name} - Accuracy: {acc:.4f}, Log Loss: {ll:.4f}")
        results[name] = {'Accuracy': acc, 'Log Loss': ll}
        
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = model
            best_is_scaled = is_scaled
            
    print(f"\nBest Model: {best_model_name} (Accuracy: {best_score:.4f})")
    
    # Refit best model on full data
    print(f"Refitting {best_model_name} on full dataset...")
    if best_is_scaled:
        # For SVM we need to scale the full dataset (beware of data leakage in real prod, but ok for this standalone step)
        full_scaler = StandardScaler()
        X_scaled = full_scaler.fit_transform(X)
        best_model.fit(X_scaled, y_class)
        # We need to return probabilities for the full dataset (cross-val predicted or just predicted)
        # For simplicity and speed requested, we'll just predict on full data (overfitting risk acceptable for this 'feature generation' step logic if we accept it)
        # BETTER: Generate CV probabilities properly to avoid leakage features
        print("Generating CV probabilities for proper feature generation...")
        y_pred_proba = cross_val_predict(best_model, X_scaled, y_class, cv=5, method='predict_proba', n_jobs=-1)
    else:
        best_model.fit(X, y_class)
        print("Generating CV probabilities for proper feature generation...")
        y_pred_proba = cross_val_predict(best_model, X, y_class, cv=5, method='predict_proba', n_jobs=-1)

    return best_model, y_pred_proba

def calculate_alift(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data = data.sort_values('y_pred_proba', ascending=False)
    data['cumulative_positives'] = data['y_true'].cumsum()
    data['cumulative_population_pct'] = (np.arange(len(data)) + 1) / len(data)
    total_positives = data['y_true'].sum()
    data['lift'] = (data['cumulative_positives'] / total_positives) / data['cumulative_population_pct']
    return data['lift'].mean()

def evaluate_models(results_dict):
    print("\n--- Model Comparison Results ---")
    results_df = pd.DataFrame(results_dict)
    print(results_df)
    results_df.to_csv('improved_model_results.csv', index=False)
    print("Results saved to 'improved_model_results.csv'")

def train_outcome_model(X, y, description):
    print(f"\nTraining Outcome Model: {description}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        verbosity=-1,
        boosting_type='gbdt',
        random_state=RANDOM_STATE
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
    
    return mean_auc, mean_alift

def main():
    df = load_data('data/bank-additional/bank-additional-full.csv')
    X_encoded, y, duration_target = preprocess_data(df)
    
    results = []
    
    # Baseline (Updated with new features too)
    auc_base, alift_base = train_outcome_model(X_encoded, y, "Baseline (With New Interaction Features)")
    results.append({
        'Model': 'Baseline (New Features)',
        'Features': 'Standard + Interactions',
        'AUC': auc_base,
        'ALIFT': alift_base
    })
    
    if duration_target is not None:
        # Model Comparison for Duration
        duration_model, predicted_probs = compare_duration_models(X_encoded, duration_target)
        
        X_enhanced = X_encoded.copy()
        # For binary, predicted_probs has shape (N, 2). col 0 is Short (0), col 1 is Long (1).
        X_enhanced['prob_short'] = predicted_probs[:, 0]
        X_enhanced['prob_long'] = predicted_probs[:, 1]
        
        # Enhanced
        auc_enhanced, alift_enhanced = train_outcome_model(X_enhanced, y, "Enhanced (With Classified Duration Probs)")
        results.append({
            'Model': 'Enhanced (Classified)',
            'Features': 'Standard + Interactions + Prob(S/L)',
            'AUC': auc_enhanced,
            'ALIFT': alift_enhanced
        })
        
    else:
        print("Warning: 'duration' column not found.")

    evaluate_models(results)

if __name__ == "__main__":
    main()
