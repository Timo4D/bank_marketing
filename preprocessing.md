# Preprocessing and Data Exploration

## 1. Dataset Overview
This study utilizes the **Bank Marketing Data Set** (UCI Machine Learning Repository), which contains data from direct marketing campaigns of a Portuguese banking institution.
*   **Source**: `bank-additional-full.csv`
*   **Instances**: 41,188
*   **Features**: 20 inputs (demographic, socio-economic, and campaign-related) + 1 target (`y`: term deposit subscription).

## 2. Exploratory Data Analysis (EDA)
### Target Distribution
The dataset is imbalanced, with the majority of clients declining the term deposit.
*   **No (Rejection)**: ~88.7%
*   **Yes (Conversion)**: ~11.3%

### Call Duration Analysis
Call duration is the primary focus of this research as a cost proxy.
*   **Distribution**: Heavily right-skewed. Most calls are short (< 3 min), but successful calls tend to be significantly longer.
*   **Median Duration**: ~180 seconds.
*   **novelty**: Unlike standard approaches that discard `duration` to avoid leakage, we retained it to create a **Target Variable for Stage 1** (Duration Classification).

## 3. Data Preprocessing
### Feature Engineering
To enhance the predictive power of the models, we engineered the following interaction features:
1.  **`age_campaign`**: Interaction between `age` and `campaign` (number of contacts). This captures the potentially varying patience levels or interest of different age groups across repeated contact attempts.
2.  **`is_first_contact`**: A binary flag derived from `pdays=999`. It explicitly indicates whether a client was previously contacted, which is a strong behavioral signal.

### Categorical Encoding
We applied a hybrid encoding strategy to preserve information while managing dimensionality:
1.  **Ordinal Encoding**: Applied to `education` to preserve the natural hierarchy (e.g., *illiterate < basic.4y < ... < university.degree*).
    *   Mapping: `illiterate`: 0 $\to$ `university.degree`: 6.
2.  **One-Hot Encoding**: Applied to nominal categorical variables (e.g., `job`, `marital`, `housing`, `loan`). `drop_first=True` was used to prevent multicollinearity.

### Duration Binning (Target Generation)
For the Duration Classification Model (Stage 1), we transformed the continuous `duration` variable into a binary target based on the median split:
*   **Short Label (0)**: Duration $\le$ 180 seconds.
*   **Long Label (1)**: Duration $>$ 180 seconds.

This balanced discretization (approx. 50/50 split) allows the classifier to learn the distinct profiles of customers who engage in detailed conversations versus those who disengage quickly.

## 4. Data Leakage Prevention
Strict measures were taken to prevent data leakage, particularly regarding the `duration` variable:
1.  **Training**: The `duration` feature was **dropped** from the input feature set ($X$) for all models.
2.  **Prediction**: The Output Model (Stage 2) uses **predicted probabilities** of duration (generated via Cross-Validation) rather than actual duration values.
