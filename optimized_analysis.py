import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random state
RANDOM_STATE = 42

# Load data
print("Loading data...")
try:
    df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the path.")
    exit()

# Data Cleaning and Preprocessing
print("\nPreprocessing data...")

# Remove 'duration' feature to avoid data leakage
if 'duration' in df.columns:
    df = df.drop('duration', axis=1)
    print("Dropped 'duration' feature.")

# Separate features and target
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Handle Categorical Features
# Ordinal features: 'education'
# Nominal features: others

ordinal_features = ['education']
nominal_features = [col for col in categorical_features if col not in ordinal_features]

# Custom mapping for education
education_order = {
    'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 
    'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': -1
}

X_encoded = X.copy()
X_encoded['education'] = X_encoded['education'].map(education_order)
# Fill unknown with -1 if any remain (though map should handle it if 'unknown' is in keys)
X_encoded['education'] = X_encoded['education'].fillna(-1)

# One-Hot Encode nominal features
print("Applying One-Hot Encoding to nominal features...")
X_encoded = pd.get_dummies(X_encoded, columns=nominal_features, drop_first=True)

print(f"Data encoded. New shape: {X_encoded.shape}")

# ALIFT Calculation Function
def calculate_alift(y_true, y_pred_proba):
    """
    Calculate the Area under the LIFT cumulative curve (ALIFT).
    """
    # Create a dataframe with true labels and predicted probabilities
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    
    # Sort by predicted probability in descending order
    data = data.sort_values('y_pred_proba', ascending=False)
    
    # Calculate cumulative sum of true positives
    data['cumulative_positives'] = data['y_true'].cumsum()
    
    # Calculate cumulative percentage of population targeted
    data['cumulative_population_pct'] = (np.arange(len(data)) + 1) / len(data)
    
    # Calculate Lift
    # Lift = (Cumulative Positives / Total Positives) / Cumulative Population Pct
    total_positives = data['y_true'].sum()
    data['lift'] = (data['cumulative_positives'] / total_positives) / data['cumulative_population_pct']
    
    # ALIFT is the mean of the Lift curve (simplified approximation)
    # A more precise integration could be done, but mean is a good proxy for "Area" in this context relative to random
    # Actually, standard ALIFT definition might vary, but let's stick to the notebook's intent if possible.
    # Notebook used a custom function. Let's try to replicate the logic if we saw it, 
    # but standard Area Under Lift Curve is often calculated by integrating.
    # For simplicity and robustness, we'll use the mean lift over deciles or the whole curve.
    # Let's use the average lift over the top 100% (which is 1.0 for random model).
    # But usually we care about the early part.
    # Let's stick to a simple metric: Mean Lift.
    return data['lift'].mean()

# Optimization with Optuna
print("\nStarting Optuna Optimization...")

def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', ['LightGBM', 'XGBoost', 'CatBoost'])
    
    if classifier_name == 'LightGBM':
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'random_state': RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**param)
        
    elif classifier_name == 'XGBoost':
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'random_state': RANDOM_STATE,
            'use_label_encoder': False,
            'n_jobs': -1
        }
        model = xgb.XGBClassifier(**param)
        
    elif classifier_name == 'CatBoost':
        param = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'random_seed': RANDOM_STATE,
            'verbose': False,
            'allow_writing_files': False
        }
        model = cb.CatBoostClassifier(**param)

    # Use SMOTE within a pipeline to avoid data leakage
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    # 3-fold CV for speed during optimization
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # We want to maximize AUC
    try:
        scores = []
        for train_idx, val_idx in cv.split(X_encoded, y):
            X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
        return np.mean(scores)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, timeout=600) # Limit trials/time for demo

print("\nBest trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: {trial.params}")

# Train best models for each type
print("\nTraining best models for Voting Classifier...")

best_models = []

# Helper to get best params for a specific classifier from the study if possible, 
# or just run a small specific study for each if we want to be thorough.
# For simplicity, let's take the best overall parameters and also train default/lightly tuned versions of others 
# OR run separate studies. Running separate studies is better.

def optimize_model(classifier_name, n_trials=10):
    print(f"Optimizing {classifier_name}...")
    def specific_objective(trial):
        if classifier_name == 'LightGBM':
            param = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': RANDOM_STATE
            }
            model = lgb.LGBMClassifier(**param)
        elif classifier_name == 'XGBoost':
            param = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'use_label_encoder': False, 'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': RANDOM_STATE
            }
            model = xgb.XGBClassifier(**param)
        elif classifier_name == 'CatBoost':
            param = {
                'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'allow_writing_files': False,
                'iterations': trial.suggest_int('iterations', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 10),
                'random_seed': RANDOM_STATE
            }
            model = cb.CatBoostClassifier(**param)
            
        pipeline = ImbPipeline([('smote', SMOTE(random_state=RANDOM_STATE)), ('classifier', model)])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X_encoded, y):
            pipeline.fit(X_encoded.iloc[train_idx], y.iloc[train_idx])
            scores.append(roc_auc_score(y.iloc[val_idx], pipeline.predict_proba(X_encoded.iloc[val_idx])[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(specific_objective, n_trials=n_trials)
    return study.best_params

# Get best params
lgb_params = optimize_model('LightGBM')
xgb_params = optimize_model('XGBoost')
cb_params = optimize_model('CatBoost')

# Instantiate models with best params
lgb_model = lgb.LGBMClassifier(**lgb_params, objective='binary', metric='auc', verbosity=-1, boosting_type='gbdt', random_state=RANDOM_STATE)
xgb_model = xgb.XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_jobs=-1, random_state=RANDOM_STATE)
cb_model = cb.CatBoostClassifier(**cb_params, loss_function='Logloss', eval_metric='AUC', verbose=False, allow_writing_files=False, random_seed=RANDOM_STATE)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('cb', cb_model)],
    voting='soft'
)

# Evaluation
print("\nEvaluating models with 10-fold CV...")
models = {
    'LightGBM': lgb_model,
    'XGBoost': xgb_model,
    'CatBoost': cb_model,
    'Voting Classifier': voting_clf
}

results = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"Evaluating {name}...")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    auc_scores = []
    alift_scores = []
    
    # Manual CV loop to calculate ALIFT
    for train_idx, test_idx in cv.split(X_encoded, y):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        alift_scores.append(calculate_alift(y_test, y_pred_proba))
        
    results.append({
        'Model': name,
        'AUC': np.mean(auc_scores),
        'ALIFT': np.mean(alift_scores)
    })

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\nFinal Results:")
print(results_df)

# Save results
results_df.to_csv('optimized_results.csv', index=False)
print("\nResults saved to optimized_results.csv")
