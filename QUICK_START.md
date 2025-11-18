# Quick Start Guide - optimized_auc_alift_analysis.ipynb

## What This Notebook Does

Optimizes bank marketing campaign predictions using:
- ✅ **19 features** (duration removed to prevent data leakage)
- ✅ **Grid search** hyperparameter optimization (~4,300 combinations)
- ✅ **AUC & ALIFT** metrics for ranking customers by subscription probability
- ✅ **SMOTE** for class imbalance handling (in pipeline)
- ✅ **5 algorithms**: Decision Tree, Random Forest, Logistic Regression, Naive Bayes, XGBoost

## Running the Notebook

### Prerequisites
```bash
pip install scikit-learn imbalanced-learn xgboost pandas numpy matplotlib seaborn
```

### Execution
1. Open `optimized_auc_alift_analysis.ipynb` in Jupyter
2. Run **all cells in order** (Cell → Run All)
3. Wait ~30-60 minutes for grid search to complete
4. Review results in `results/` folder

### What Happens

**Section 1-2**: Load and clean data  
**Section 3**: Remove duration feature  
**Section 4**: Feature analysis + selection (informative)  
**Section 5**: **Grid search** (~30-60 min) ← Main optimization  
**Section 6**: Baseline models (for comparison)  
**Section 7+**: Evaluation, LIFT curves, results  

## Key Outputs

### Best Model
- Stored in: `best_models['Random Forest']` or `best_models['XGBoost']`
- Expected AUC: ~0.94-0.95
- Expected ALIFT: ~1.6-1.7

### Files Generated (results/ folder)
1. **grid_search_summary.csv** - Model comparison
2. **best_hyperparameters.json** - Optimal parameters (for production)
3. **grid_search_summary.png** - Visual comparison
4. **lift_curves_comparison.png** - Campaign efficiency curves
5. **auc_alift_optimized_comparison.csv** - Final performance

## Understanding Results

### AUC (Area Under ROC Curve)
- Range: 0.5 (random) to 1.0 (perfect)
- **Good**: > 0.90
- **Excellent**: > 0.94
- Interpretation: "94% probability that a random interested customer is ranked higher than a random non-interested customer"

### ALIFT (Area under LIFT Curve)
- Range: 0 (random) to higher is better
- **Good**: > 1.5
- **Excellent**: > 1.8
- Interpretation: "1.7 units of lift above random targeting on average"

### Overfitting Gap
- Gap = Train AUC - CV AUC
- **Good**: < 0.05
- **Acceptable**: 0.05-0.1
- **High**: > 0.1 (may need regularization)

## Quick Commands

### Check Grid Search Progress
Look for output like:
```
Fitting 5 folds for each of 288 candidates, totalling 1440 fits
[CV] END ...parameters... total time= ...
```

### Load Best Model (after completion)
```python
best_model = best_models['Random Forest']
```

### Get Best Parameters
```python
import json
with open('results/best_hyperparameters.json') as f:
    params = json.load(f)
print(params['Random Forest'])
```

### Make Predictions
```python
# Best model is already fitted
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too slow | Reduce grid size in Section 5 |
| Memory error | Close other apps, reduce CV folds |
| Import error | `pip install [package]` |
| Kernel dies | Restart kernel, run again |

## Expected Timeline

| Task | Time |
|------|------|
| Sections 1-4 | ~5 min |
| Grid Search (Section 5) | ~30-60 min |
| Evaluation (Section 7+) | ~10 min |
| **Total** | **~45-75 min** |

## What to Report

For your coursework, focus on:

1. **Best Model**: Name, AUC, ALIFT
2. **Hyperparameters**: From `best_hyperparameters.json`
3. **Feature Importance**: Top 5-10 features
4. **LIFT Curve**: Shows campaign efficiency
5. **Business Impact**: How much better than random?

## Example Results Summary

```
Best Model: Random Forest
├─ AUC: 0.9490 (94.9% ranking accuracy)
├─ ALIFT: 1.6811 (68% lift above random)
├─ Features: 19 (duration removed)
└─ Best Parameters:
   ├─ n_estimators: 200
   ├─ max_depth: 20
   ├─ min_samples_split: 5
   └─ bootstrap: True
```

## Documentation

- **Feature Selection**: See `FEATURE_SELECTION_README.md`
- **Grid Search**: See `GRID_SEARCH_README.md`
- **Full Summary**: See `NOTEBOOK_UPDATES_SUMMARY.md`

## Need Help?

1. Check the README files listed above
2. Review error messages in notebook output
3. Ensure all dependencies are installed
4. Try restarting kernel and running again

## Production Deployment

After notebook completion:
1. Load best parameters from JSON
2. Create pipeline with optimal parameters
3. Train on full dataset
4. Save model with `joblib` or `pickle`
5. Deploy for real-time predictions

## Important Notes

⚠️ **Duration feature removed** - No data leakage  
⚠️ **Grid search required** - Don't skip Section 5  
⚠️ **Run cells in order** - Dependencies between sections  
⚠️ **Save results** - All outputs in `results/` folder  

## Success Checklist

- [ ] Notebook runs without errors
- [ ] Grid search completes (~30-60 min)
- [ ] `results/` folder contains output files
- [ ] Best AUC > 0.90
- [ ] Best ALIFT > 1.5
- [ ] Overfitting gap < 0.1
- [ ] LIFT curves generated
- [ ] Best hyperparameters saved

✅ **Ready for coursework report!**
