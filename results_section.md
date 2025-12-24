# Results and Conclusion

## 1. Experimental Setup & Metrics
Our experiments evaluated the impact of incorporating call duration predictions into the lead prioritization strategy. We evaluated four configurations:
1.  **Baseline (Random)**: Random calling (simulating a standard list-based approach).
2.  **Standard ML (Probability)**: Prioritizing leads by highest $P(Success)$ (Standard Industry Practice).
3.  **Efficiency (Proposed)**: Prioritization based on Efficiency Score ($P(Success) \div E[Time]$).
4.  **Oracle (Upper Bound)**: Prioritization using *perfect* (ground-truth) duration knowledge.

## 2. Model Performance
### Duration Classification
Attempts to classify pre-call duration yielded modest signal.
*   **Accuracy**: **57.56%** (vs 50% random).
*   **Log Loss**: 0.6727.

### Outcome Prediction
Integrating duration probabilities yielded a slight improvement in AUC over the Standard model.
*   **Standard (Prob Only) AUC**: 0.7746
*   **Enhanced (Duration Props) AUC**: 0.7755
*   **Oracle (Perfect Info) AUC**: 0.8646

## 3. Business Impact Simulation (The 8-Hour Shift)
We simulated an 8-hour agent shift (28,800 seconds) to verify the real-world impact.

| Strategy | Metric Implied | Sales (Conversions) | Lift vs Random | Lift vs Standard |
| :--- | :--- | :--- | :--- | :--- |
| **Random** | - | 12.4 | - | - |
| **Standard ML** | Highest Prob | **66.0** | +432% | - |
| **Efficiency (Proposed)** | Best ROI | 54.0 | +335% | -18% |
| **Oracle (Theoretical)** | Best ROI | **117.0** | +843% | **+77%** |

### Key Findings
1.  **Current Performance Gap**: The *Proposed Efficiency Strategy* currently underperforms the *Standard Probability Strategy* (54 vs 66 Sales).
    *   **Reason**: The Duration Classifier (57% accuracy) is not yet precise enough. It likely misclassifies some high-probability long calls as "inefficient", causing them to be skipped.
2.  **Theoretical Dominance (Novelty)**: The *Oracle* experiment proves that if duration could be predicted perfectly, the Efficiency Strategy would yield **117 Sales**, nearly **double** the Standard Strategy (66 Sales).
    *   This confirms that the **concept** of "Return on Time" is superior to "Return on Call," but its realization depends entirely on the quality of the duration predictor.

## 4. Conclusion
We proposed a novel "Efficiency Score" that prioritizes leads by *revenue per second* rather than *revenue per call*.
*   **Novelty**: Integrating pre-call duration prediction to operationalize the "forbidden variable" of sales time.
*   **Status**: While the current duration model (Activity-based features) is too weak to beat standard probability scoring, the approach has a massive theoretical ceiling (+77% over standard state-of-the-art). Future work should focus on richer data (e.g., audio analytics, historical user profile) to bridge this gap.
