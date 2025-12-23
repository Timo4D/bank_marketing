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
    
    return mean_auc, mean_alift

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
        auc_enhanced, alift_enhanced = train_outcome_model(X_enhanced, y, "Enhanced Model (with Duration Probabilities)")
        
        results.append({
            'Model': 'Enhanced LightGBM',
            'Features': 'Base + Duration Probs',
            'AUC': auc_enhanced,
            'ALIFT': alift_enhanced
        })
        
        print("\n--- Final Results ---")
        print(pd.DataFrame(results))
        pd.DataFrame(results).to_csv('final_model_results.csv', index=False)
        print("Results saved to 'final_model_results.csv'")

    else:
        print("Error: Duration target could not be created.")

if __name__ == "__main__":
    main()
