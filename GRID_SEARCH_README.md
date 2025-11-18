# Hyperparameter Optimization via Grid Search

## Overview

The `optimized_auc_alift_analysis.ipynb` notebook now includes comprehensive hyperparameter optimization using GridSearchCV for all algorithms. This document explains the grid search implementation and how to interpret the results.

## What is Grid Search?

Grid search is an exhaustive search over specified parameter values for an estimator. It:
1. Tests all possible combinations of hyperparameters
2. Evaluates each combination using cross-validation
3. Selects the combination with the best performance (AUC in our case)
4. Returns the optimal model trained on the full dataset

## Implementation Details

### Cross-Validation Strategy
- **5-fold Stratified CV** during grid search (for speed)
- **10-fold Stratified CV** for final evaluation
- Stratification maintains class distribution across folds

### Pipeline Architecture
All models use `imblearn.Pipeline` to prevent data leakage:
```python
Pipeline([
    ('smote', SMOTE()),           # Step 1: Oversample minority class
    ('scaler', StandardScaler()), # Step 2: Scale features (if needed)
    ('classifier', Model())       # Step 3: Train classifier
])
```

**Key benefit**: SMOTE is applied ONLY to training data in each fold, not to test data.

### Scoring Metric
- **Primary**: `roc_auc` (Area Under ROC Curve)
- **Why AUC?**: Optimal for ranking customers by subscription probability
- Directly optimizes for the ALIFT metric

## Hyperparameter Grids

### 1. Decision Tree (J48)
```python
Parameters tested: 288 combinations
- criterion: ['gini', 'entropy']
- max_depth: [None, 5, 10, 15, 20, 25]
- min_samples_split: [2, 5, 10, 20]
- min_samples_leaf: [1, 2, 5, 10]
- max_features: [None, 'sqrt', 'log2']
```

**Purpose**:
- `criterion`: Split quality measure
- `max_depth`: Prevents overfitting
- `min_samples_split/leaf`: Controls tree granularity
- `max_features`: Feature subset randomization

### 2. Random Forest
```python
Parameters tested: 864 combinations
- n_estimators: [50, 100, 200, 300]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2', None]
- bootstrap: [True, False]
```

**Purpose**:
- `n_estimators`: Number of trees (more = better but slower)
- `max_depth`: Tree depth limit
- `bootstrap`: Whether to use bootstrap samples
- Other params: Control overfitting and diversity

### 3. Logistic Regression
```python
Parameters tested: ~120 combinations (some invalid)
- penalty: ['l1', 'l2', 'elasticnet', None]
- C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- solver: ['saga']
- l1_ratio: [0.0, 0.25, 0.5, 0.75, 1.0]
```

**Purpose**:
- `penalty`: Regularization type
- `C`: Inverse of regularization strength (smaller = stronger)
- `l1_ratio`: Elasticnet mix (0 = L2, 1 = L1)

### 4. Naive Bayes
```python
Parameters tested: 7 combinations
- var_smoothing: [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
```

**Purpose**:
- `var_smoothing`: Portion of largest variance added to variances for stability

### 5. XGBoost
```python
Parameters tested: 3,024 combinations
- n_estimators: [50, 100, 200, 300]
- max_depth: [3, 5, 7, 9]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]
- min_child_weight: [1, 3, 5]
- gamma: [0, 0.1, 0.2]
```

**Purpose**:
- `n_estimators`: Number of boosting rounds
- `max_depth`: Tree depth
- `learning_rate`: Step size shrinkage
- `subsample`: Fraction of samples for each tree
- `colsample_bytree`: Fraction of features for each tree
- `min_child_weight`: Minimum sum of instance weight in child
- `gamma`: Minimum loss reduction for split

## Expected Runtime

| Algorithm | Grid Size | Estimated Time |
|-----------|-----------|----------------|
| Decision Tree | 288 | 2-5 minutes |
| Random Forest | 864 | 10-20 minutes |
| Logistic Regression | ~120 | 3-7 minutes |
| Naive Bayes | 7 | < 1 minute |
| XGBoost | 3,024 | 15-30 minutes |
| **Total** | **~4,300** | **30-60 minutes** |

*Times vary based on hardware (CPU cores, RAM)*

## Outputs Generated

### 1. Grid Search Summary CSV
**File**: `results/grid_search_summary.csv`

Contains:
- Model name
- Best CV AUC score
- Train AUC score
- Overfitting gap (Train - CV)
- Time taken (minutes)

### 2. Best Hyperparameters JSON
**File**: `results/best_hyperparameters.json`

Contains optimal hyperparameters for each model in JSON format.

### 3. Visualizations
**File**: `results/grid_search_summary.png`

Two plots:
- **Left**: Best AUC scores (CV vs Train) for each model
- **Right**: Overfitting gap analysis (color-coded)
  - Green: < 0.05 (good)
  - Orange: 0.05-0.1 (moderate)
  - Red: > 0.1 (high overfitting)

## Interpreting Results

### Best CV AUC Score
- **What it means**: Model's performance on unseen data
- **Higher is better**: Range 0.5-1.0
- **Good performance**: > 0.90 for this dataset

### Overfitting Gap
```
Gap = Train AUC - CV AUC
```

- **< 0.05**: Excellent generalization
- **0.05-0.1**: Good, acceptable gap
- **> 0.1**: Model may be overfitting

### Example Interpretation
```
Model: Random Forest
Best CV AUC: 0.9450
Train AUC: 0.9852
Overfitting Gap: 0.0402
```

**Analysis**:
- Excellent AUC score (94.5%)
- Low overfitting gap (4%)
- Model generalizes well
- Safe to deploy

## How Grid Search Prevents Overfitting

1. **Cross-Validation**: Tests on multiple train/test splits
2. **Separate Test Set**: Grid search NEVER sees final test set
3. **SMOTE in Pipeline**: Applied per-fold, preventing leakage
4. **Regularization Parameters**: Explicitly tested (C, max_depth, etc.)
5. **Overfitting Monitoring**: Track train vs validation gap

## Using the Results

### In the Notebook

After grid search completes:

```python
# Best models are stored in 'best_models' dictionary
best_rf_model = best_models['Random Forest']

# Grid search results in 'grid_search_results' dictionary
rf_best_params = grid_search_results['Random Forest']['best_params']
rf_best_score = grid_search_results['Random Forest']['best_score']

# All subsequent evaluations use optimized models
models_optimized  # Contains grid search winners
```

### For Production

1. Load best hyperparameters from JSON:
```python
import json
with open('results/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)
```

2. Create model with optimal parameters:
```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Extract Random Forest params (remove 'classifier__' prefix)
rf_params = {k.replace('classifier__', ''): v
             for k, v in best_params['Random Forest'].items()}

# Create optimized pipeline
pipeline = Pipeline([
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier(**rf_params))
])
```

## Advantages of This Approach

✅ **No Manual Tuning**: Automates hyperparameter selection
✅ **Reproducible**: Same results every time (fixed random seed)
✅ **No Data Leakage**: Pipeline prevents test data contamination
✅ **Optimal Performance**: Tests thousands of combinations
✅ **Overfitting Control**: Monitors generalization gap
✅ **Production Ready**: Best params saved for deployment

## Common Questions

### Q: Why 5-fold for grid search but 10-fold for evaluation?
**A**: 5-fold is faster for hyperparameter search (less time). 10-fold gives more reliable final performance estimate.

### Q: Can I add more hyperparameters?
**A**: Yes! Just add to the `param_grid` dictionaries. Be aware: grid size grows multiplicatively (e.g., adding 3 values = 3x time).

### Q: Why use Pipeline?
**A**: Prevents data leakage. SMOTE must be applied ONLY to training data in each fold, never to test data.

### Q: What if grid search takes too long?
**A**: Options:
1. Reduce parameter ranges
2. Use RandomizedSearchCV instead
3. Use fewer CV folds (e.g., 3)
4. Run on more powerful hardware
5. Use n_jobs=-1 for parallelization (already enabled)

### Q: How do I know if overfitting?
**A**: Check the overfitting gap in summary. If > 0.1, consider:
- Stronger regularization
- Simpler model
- More data
- Feature reduction

## Next Steps

After grid search:
1. ✅ Review `grid_search_summary.csv`
2. ✅ Check overfitting gaps
3. ✅ Compare AUC scores across models
4. ✅ Proceed with 10-fold CV evaluation
5. ✅ Generate final LIFT curves
6. ✅ Deploy best model

## Troubleshooting

### Issue: "Memory Error"
**Solution**: Reduce grid size or use fewer CV folds

### Issue: "Too Slow"
**Solution**:
- Reduce parameter ranges
- Use RandomizedSearchCV
- Run overnight

### Issue: "Invalid parameter combination"
**Solution**: Grid search automatically skips invalid combinations (e.g., l1_ratio with non-elasticnet penalty)

### Issue: "No improvement over baseline"
**Solution**:
- Check if data is properly prepared
- Verify SMOTE is working
- Try different parameter ranges

## Technical Notes

- **Random State**: Set to 42 for reproducibility
- **n_jobs=-1**: Uses all CPU cores
- **verbose=1**: Shows progress during search
- **return_train_score=True**: Enables overfitting analysis
- **scoring='roc_auc'**: Primary optimization metric

## References

- [Scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Imbalanced-learn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
