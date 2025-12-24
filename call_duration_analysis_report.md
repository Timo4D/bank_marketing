# Walkthrough - Binary Call Duration Classification

I have implemented a binary classification model to predict whether a call will be "Short" (<= 180s) or "Long" (> 180s), used these predictions to enhance the final outcome model, and developed a **Lead Prioritization Scheduling Model** to maximize agent efficiency.

## Setup

- **Duration Split**: Median (180 seconds).
- **Classes**:
  - `0`: Short (<= 180s)
  - `1`: Long (> 180s)
- **Model**: LightGBM (Gradient Boosting Decision Tree).

## Results

### 1. Duration Classification (Model Selection)

I compared three models for classifying duration. LightGBM was the most effective.

| Model | Accuracy | Log Loss |
| :--- | :--- | :--- |
| **LightGBM** | **57.82%** | **0.6735** |
| Random Forest | 57.68% | 0.6734 |
| SVM (SGD) | 55.07% | 0.6947 |

### 2. Final Impact on Outcome Prediction

Using the LightGBM classifier to generate "Short/Long" probabilities for the final model resulted in a clear improvement over the baseline.

| Model | Features | AUC | ALIFT |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard + Interactions | 0.7860 | 2.1967 |
| **Enhanced** | Standard + Interactions + Prob(S/L) | **0.7964** | **2.2114** |

## Lead Prioritization Model (Scheduling)

To improve agent efficiency, I implemented an **Efficiency Score** to prioritize calls:

$$ \text{Efficiency Score} = \frac{P(\text{Success})}{E[\text{Duration}]} $$

Where expected duration is calculated using the predicted Short/Long probabilities and historical averages (Short=101s, Long=417s).

### Top 5 Prioritized Calls
The algorithm identifies high-probability, low-duration calls effectively:

| Prob(Success) | Expected Duration (s) | Efficiency Score | Actual Result (y) |
| :--- | :--- | :--- | :--- |
| **92.5%** | **124s** | 0.0074 | **Yes** |
| **88.3%** | **119s** | 0.0074 | **Yes** |
| **88.3%** | **119s** | 0.0074 | **Yes** |
| **85.3%** | **116s** | 0.0074 | **Yes** |
| **87.7%** | **120s** | 0.0073 | **Yes** |

## Business Impact Analysis (The 8-Hour Shift)

To quantify the real-world value, I simulated an agent working an 8-hour shift (28,800 seconds) using two strategies:
1.  **Baseline**: Random calling (Standard Approach).
2.  **Prioritized**: Calling leads in order of the **Efficiency Score**.

### Simulation Methodology
The simulation models a single agent's workday constraint—**Time**.
1.  **Constraint**: 8 hours (28,800 seconds).
2.  **Process**:
    -   We iterate through the sorted (or shuffled) list of leads.
    -   For each lead contacted, we subtract their **actual** call duration from the remaining time.
    -   We count 1 "Sale" if the actual outcome was `y=yes`.
    -   The simulation stops when 8 hours of talk-time have elapsed.
3.  **Stability**: The Baseline strategy is averaged over 100 random trials to ensure a fair comparison.
    
### Simulation Results

| Metric | Baseline (Random) | Prioritized (Model) | Lift |
| :--- | :--- | :--- | :--- |
| **Calls Made** | 109.3 | 95.0 | -13.1% |
| **Sales (Conversions)** | **12.8** | **51.0** | **+299.7%** |

### Impact Summary
> [!IMPORTANT]
> **Tripled Productivity**: An agent using this prioritization model generates **~38 more sales per day** than an agent calling randomly.
>
> Surprisingly, they make *fewer* calls (-13%) but spend their time talking to highly likely buyers, drastically increasing the "Revenue per Hour".

## Oracle Duration Analysis (The "What If" Scenario)

To test if improving duration prediction is worth further investment, I ran an **Oracle Experiment** assuming a Perfect Duration Predictor (100% accuracy).

### The Gap Analysis

| Metric | Current Model | Oracle (Perfect) | Gap (Potential) |
| :--- | :--- | :--- | :--- |
| **Model AUC** | 0.7964 | 0.8733 | +0.077 |
| **Sales per Shift** | 51.0 | 141.0 | **+176% over current** |

> [!TIP]
> **Investment Recommendation**:
> While the current model provides a massive **3x lift** over random calling, a perfect duration predictor would yield a **10x lift** (141 sales vs 12.8). This huge gap proves that finding new features to predict call duration (e.g., customer personality, historical talk time) has extremely high ROI.

## Conclusion

Predicting call duration classification (Short/Long) provides valuable signal:
1.  **Model Lift**: Improves AUC by ~0.01 (0.786 -> 0.796).
2.  **Operational Efficiency**: Enables a rigorous "Return on Time" scheduling model that prioritizes quick wins.

The final script [`improved_bank_marketing.py`](file:///home/timo/bank_marketing/improved_bank_marketing.py) generates the [`prioritized_call_schedule.csv`](file:///home/timo/bank_marketing/prioritized_call_schedule.csv) automatically and runs this simulation to verify performance.
