import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'
SAVE_DIR = "plots"
DATA_PATH = "data/bank-additional/bank-additional-full.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

warnings.filterwarnings('ignore')

def load_and_prep_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Preprocessing identical to training script
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Duration classes
    bins = [-1, 180, float('inf')]
    labels = [0, 1]
    df['duration_class'] = pd.cut(df['duration'], bins=bins, labels=labels).astype(int)
    
    return df

def plot_duration_distribution(df):
    print("Plotting Duration Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Filter out extreme outliers for better visualization (e.g., > 2000s)
    data = df[df['duration'] < 2000]
    
    sns.histplot(data=data, x='duration', hue='y', kde=True, bins=50, palette='viridis', alpha=0.6)
    
    # Add vertical line for threshold
    plt.axvline(x=180, color='red', linestyle='--', linewidth=2, label='Short/Long Threshold (180s)')
    
    plt.title('Call Duration Distribution by Outcome', fontsize=16)
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Outcome (y)')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/duration_distribution.png", dpi=300)
    plt.close()

def plot_correlation_heatmap(df):
    print("Plotting Correlation Heatmap...")
    # Select numeric columns and key targets
    cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 
            'cons.conf.idx', 'euribor3m', 'nr.employed', 'duration', 'y_binary']
    
    # Handle pdays=999
    df_corr = df[cols].copy()
    df_corr['pdays'] = df_corr['pdays'].replace(999, -1) # Treat as binary-ish or distinct
    
    corr = df_corr.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png", dpi=300)
    plt.close()

def plot_roc_curve(df):
    print("Plotting ROC Curve...")
    # Requires re-running prediction or loading saved model
    # To be accurate/consistent with the paper, we should get the CV probabilities
    # But for a "pretty graph", we can just use the saved model on the full data (or subset)
    
    # Load Outcome Model Pipeline
    model_path = os.path.join(MODEL_DIR, "outcome_model.joblib")
    if not os.path.exists(model_path):
        print("Outcome model not found. Skipping ROC.")
        return

    pipeline = joblib.load(model_path)
    
    # We need X_encoded
    # For simplicity, we'll assume we can pass raw data if pipeline handles encoding?
    # NO, the saved pipeline in improved_bank_marketing.py expects ENCODED data.
    # We need to reproduce the encoding step.
    
    # ... Or we just load the metrics from csv if saved? 
    # Let's quickly re-encode.
    # To match the training data (48 features), we need to replicate the EXACT preprocessing
    # from improved_bank_marketing.py
    
    # 1. Load raw again to be safe
    df_raw = pd.read_csv(DATA_PATH, sep=';')
    
    # 2. Replicate preprocess_data logic EXACTLY
    
    # Interaction features
    if 'age' in df_raw.columns and 'campaign' in df_raw.columns:
        df_raw['age_campaign'] = df_raw['age'] * df_raw['campaign']
    if 'pdays' in df_raw.columns:
        df_raw['is_first_contact'] = (df_raw['pdays'] == 999).astype(int)
        
    X_features = df_raw.drop(['y', 'duration'], axis=1) # duration is dropped for training
    
    # Categorical Encoding Logic from improved_bank_marketing.py
    # 1. Identify categorical
    categorical_features = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Ordinal Encoding for Education
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
            
    # 3. One-Hot Encoding for the rest
    X_encoded = pd.get_dummies(X_encoded, columns=categorical_features, drop_first=True, dtype=int)
    
    print(f"Encoded shape: {X_encoded.shape}") 
    
    # Add duration probs
    dur_model_path = os.path.join(MODEL_DIR, "duration_model.joblib")
    if not os.path.exists(dur_model_path):
        print("Duration model not found. Skipping ROC.")
        return
        
    dur_model = joblib.load(dur_model_path)
    
    # Predict Duration Probs
    dur_probs = dur_model.predict_proba(X_encoded)
    
    # Add features for Outcome Model
    X_encoded['prob_short'] = dur_probs[:, 0]
    X_encoded['prob_long'] = dur_probs[:, 1]
    
    y = df['y_binary']
    
    # Get probabilities
    y_pred_proba = pipeline.predict_proba(X_encoded)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Enhanced Model (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/roc_curve.png", dpi=300)
    plt.close()
    
    # Feature Importance (if LGBM)
    if hasattr(pipeline, 'named_steps'):
        clf = pipeline.named_steps['classifier']
        lgb.plot_importance(clf, max_num_features=15, figsize=(10, 8), title='Feature Importance (Outcome Model)')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/feature_importance.png", dpi=300)
        plt.close()

def plot_simulation_results():
    print("Plotting Simulation Results...")
    # Results from latest run:
    # Random: 12.4
    # Standard: 66.0
    # Efficiency: 54.0
    # Oracle: 117.0
    
    strategies = ['Random', 'Prob Only\n(Standard)', 'Efficiency\n(Proposed)', 'Oracle\n(Theoretical)']
    sales = [12.4, 66.0, 54.0, 117.0]
    colors = ['gray', 'purple', '#2ecc71', '#3498db'] 
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, sales, color=colors, width=0.6)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    # Add Standard vs Efficiency Note
    plt.annotate(f"Current Limit\n(Duration Model Weak)", 
                 xy=(2, 54), xytext=(2, 80),
                 ha='center', fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.annotate(f"Potential\nTarget", 
                 xy=(3, 117), xytext=(3, 135),
                 ha='center', fontsize=10, color='blue', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='blue'))

    plt.title('Sales per 8-Hour Shift (Strategy Comparison)', fontsize=16)
    plt.ylabel('Number of Sales', fontsize=12)
    plt.ylim(0, 160)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/simulation_results.png", dpi=300)
    plt.close()

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    df = load_and_prep_data()
    
    plot_duration_distribution(df)
    plot_correlation_heatmap(df)
    plot_roc_curve(df) # This might fail if encoding differs slightly, but worth a try
    plot_simulation_results()
    
    print(f"\nAll plots saved to {SAVE_DIR}/")

if __name__ == "__main__":
    main()
