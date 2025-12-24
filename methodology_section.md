# Methodology

## 1. Data Preprocessing and Feature Engineering
The dataset used was the *Bank Marketing* dataset (Moro et al., 2014). To enable predictive modeling, categorical variables (e.g., job, marital status) were one-hot encoded, and ordinal variables (education) were mapped to integer scales.

A key preprocessing step involved the `duration` variable. Since call duration is unknown before a call is made, it cannot be used directly as a feature for predictive modeling in a real-time setting. However, it serves as a critical proxy for cost (agent time). We discretized the `duration` variable into two binary classes using the median split (180 seconds):
*   **Short**: $t \le 180s$
*   **Long**: $t > 180s$

## 2. Two-Stage Predictive Modeling Framework
We proposed a two-stage modeling approach to optimize call center efficiency:

### Stage 1: Pre-Call Duration Classification
We trained a supervised learning model to predict the probability of a call being "Short" or "Long" based solely on pre-call features (e.g., customer demographics, past campaign interactions). We evaluated three algorithms: **LightGBM**, **Random Forest**, and **Support Vector Machines (SVM)**. LightGBM was selected as the final model due to superior accuracy ($57.8\%$) and training speed.

This model outputs a probability vector for each lead:
$$ \hat{P}_{duration} = [P(\text{Short}), P(\text{Long})] $$

### Stage 2: Enhanced Outcome Prediction
In the second stage, we trained a binary classifier (LightGBM) to predict the conversion outcome ($y$). To capture the latent relationship between expected interaction time and success probability, we enriched the feature space with the predicted duration probabilities from Stage 1. This "Enhanced Model" demonstrated a measurable improvement in AUC ($0.796$ vs. $0.786$ baseline) and Lift.

## 3. Lead Prioritization Strategy (Efficiency Scoring)
Traditional lead scoring ranks customers solely by conversion probability ($P(y=1)$). We introduced a cost-aware **Efficiency Score** ($S_{eff}$) to maximize the *Return on Time Invested (ROTI)*:

$$ S_{eff} = \frac{P(y=1)}{E[\text{Duration}]} $$

Where the expected duration $E[\text{Duration}]$ is calculated using the stage-1 predictions:
$$ E[\text{Duration}] = P(\text{Short}) \cdot \mu_{short} + P(\text{Long}) \cdot \mu_{long} $$
*   $\mu_{short} \approx 101s$ (historical average for short calls)
*   $\mu_{long} \approx 417s$ (historical average for long calls)

This metric prioritizes "Quick Wins"—clients with a high probability of conversion who require minimal agent time.

## 4. Business Impact Simulation
To quantify the operational impact in a realistic setting, we simulated an 8-hour agent shift (28,800 seconds).
*   **Protocol**: We systematically "dialed" leads from a hold-out test set according to two strategies: (1) Random Order (Baseline) and (2) Priority Order (Efficiency Score).
*   **Constraint**: For each call made, the *actual* historical duration was subtracted from the agent's time budget.
*   **Metric**: The total number of successful conversions ($y=yes$) achieved before time expired.

We further established a theoretical upper bound by running an **Oracle Experiment**, where the "Stage 1" predictor was replaced with perfect ground-truth duration labels, allowing us to measure the maximum potential efficiency gain attainable through perfect duration forecasting.
