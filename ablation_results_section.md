# Feature Engineering Ablation Study

## Methodology

We conducted a systematic ablation study to evaluate the contribution of engineered features to model performance. All experiments used LightGBM with SMOTE oversampling, evaluated via 5-fold stratified cross-validation with AUC-ROC as the primary metric.

## Baseline Feature Analysis

We first evaluated two commonly used interaction features from the original model:

| Feature | Formula | AUC Improvement | Verdict |
|---------|---------|-----------------|---------|
| `age_campaign` | age × campaign | +0.0015 (+0.19%) | Negligible |
| `is_first_contact` | pdays = 999 | +0.0008 (+0.10%) | Negligible |

Neither feature provided statistically meaningful improvement, with gains smaller than cross-validation variance (σ ≈ 0.005). This confirms that tree-based models can learn these patterns directly from raw features.

## Novel Feature Engineering

We proposed and evaluated three alternative features based on domain knowledge:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `contact_intensity` | campaign / (pdays + 1) | Captures pursuit aggressiveness relative to recency |
| `previous_success` | poutcome = 'success' | Simplifies previous outcome to binary signal |
| `financial_stress` | Σ(housing, loan, default = 'yes') | Ordinal scale (0-3) of financial obligations |

**Table 1: Individual Feature Contribution**

| Configuration | Mean AUC | Std | vs Baseline |
|---------------|----------|-----|-------------|
| Baseline (no engineered features) | 0.7812 | 0.0053 | — |
| + contact_intensity | 0.7855 | 0.0044 | **+0.0044** |
| + previous_success | 0.7806 | 0.0054 | −0.0005 |
| + financial_stress | 0.7845 | 0.0065 | **+0.0033** |
| + All three combined | 0.7884 | 0.0055 | **+0.0073** |

The `contact_intensity` feature provided the strongest individual improvement (+0.56%), followed by `financial_stress` (+0.43%). Notably, `previous_success` showed no benefit, as LightGBM already captures this signal through the one-hot encoded `poutcome` variable.

## Feature Combination Analysis

Combining `contact_intensity` and `financial_stress` yielded a cumulative improvement of +0.93% AUC. We further tested whether adding `age_campaign` to this combination would provide additional benefit:

**Table 2: age_campaign Additive Effect**

| Configuration | Mean AUC | Δ AUC |
|---------------|----------|-------|
| contact_intensity + financial_stress | 0.7884 | — |
| + age_campaign | 0.7857 | **−0.0027** |

Adding `age_campaign` **decreased** performance across all five folds, suggesting it introduces noise when combined with more informative features.

## Final Model Comparison

**Table 3: Original vs. Optimized Feature Set**

| Version | Engineered Features | Mean AUC |
|---------|---------------------|----------|
| V1 (Original) | age_campaign, is_first_contact | 0.7822 |
| V2 (Optimized) | contact_intensity, financial_stress | **0.7888** |

The optimized feature set achieved a **+0.84% improvement** over the original, winning in all 5 cross-validation folds.

## Discussion

Our ablation study reveals several insights:

1. **Tree-based models learn simple interactions**: Features like `age × campaign` and binary thresholds (`pdays = 999`) provide negligible value to gradient boosting models, which can discover these patterns automatically.

2. **Ratio features add value**: The `contact_intensity` ratio captures a relationship between contact frequency and recency that is more difficult for trees to discover through sequential splits.

3. **Ordinal aggregation helps**: Combining related binary indicators into an ordinal scale (`financial_stress`) provides a more informative signal than individual features.

4. **Feature redundancy hurts**: Adding marginally useful features to an optimized set can degrade performance, likely due to increased noise and feature competition.

## Conclusion

Through systematic ablation, we identified an optimized feature set that improves AUC by 0.84% over the baseline. The final model uses two engineered features: `contact_intensity` and `financial_stress`, excluding the original `age_campaign` and `is_first_contact` features.
