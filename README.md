# Bank Term Deposit Subscription Prediction: Supervised Classification

An end-to-end supervised machine learning project in Python (`scikit-learn`) predicting whether retail banking clients will subscribe to a term deposit (`SUBSCRIBED`) **prior to placing an outbound telemarketing call**. Built on **39,883 client records** (`38,912` unique observations after cleaning), this benchmark addresses class imbalance (~90% `no` / ~10% `yes`), post-call target leakage, multicollinearity, variable selection, and leak-free out-of-fold threshold optimization across **6 classifier families**.

---

## 1. Problem Formulation & Objectives

In outbound bank telemarketing, dialing every customer wastes call-center capacity, while missing high-propensity subscribers sacrifices deposit revenue. Because only **~10.2%** of clients subscribe, naive classifiers may produce a score of ~90% accuracy simply by predicting `"no"` while failing to detect actual subscribers.

This project implements a leak-free classification workflow to:
- Build a realistic **pre-call** predictive pipeline by removing post-call target leakage (`duration`).
- Benchmark **6 classifier families** spanning baseline, distance-based, $L_1$-regularized parametric, probabilistic, bagged tree-ensemble, and gradient-boosted architectures.
- Prevent overfitting and model-selection bias using **three-way stratified partitioning** (Train / Test / Held-Out Validation) and a shared **10-fold stratified cross-validation** splitter (`StratifiedKFold`).
- Move from hard classification to **soft classification**, running out-of-fold probability threshold optimization to maximize minority-class **F1-score**, alongside threshold-independent ranking diagnostics (**Precision–Recall Curves / Average Precision** and **ROC–AUC**).

---

## 2. Dataset, Data Hygiene & Feature Engineering

The raw dataset contains **39,883 rows and 22 columns** (10 numeric predictors, 11 categorical predictors, and the binary target `SUBSCRIBED`) with **0 missing values**.

### Key Preprocessing & Cleaning Steps
1. **Leakage Variable Removal (`duration`):** Call duration is only observed *after* a phone call concludes (and `duration = 0` deterministically implies `no`). It was removed prior to modeling so every feature is available before dialing.
2. **Sentinel-Code Engineering (`pdays` → `pdays_was_contacted`):** The raw `pdays` variable uses `999` as a special code for clients never previously contacted (**97.36%** of records). To prevent artificial numeric distortion, `pdays` was converted into a binary indicator `pdays_was_contacted` (`0`: 38,831 [97.36%], `1`: 1,052 [2.64%]) and the raw column was dropped.
3. **Standardization & Deduplication:** Normalized string casing (`otherproduct`), cast categorical attributes to `category`, and removed duplicate records, yielding **38,912 unique observations** (`34,943` `no` [89.80%] / `3,969` `yes` [10.20%]).
4. **Multicollinearity Pruning ($\vert{}r\vert{} > 0.85$):** Pearson correlation diagnostics identified severe collinearity among macroeconomic variables:
   - `emp.var.rate` vs. `euribor3m`: **$r = 0.98$**
   - `euribor3m` vs. `nr.employed`: **$r = 0.96$**
   - `emp.var.rate` vs. `nr.employed`: **$r = 0.95$**
   - Dropped `euribor3m` and `nr.employed` while retaining `emp.var.rate`, `cons.price.idx`, and `cons.conf.idx`, producing **18 clean predictors** (7 numeric + 11 categorical).
5. **Outlier Audit (IQR Rule):** Quantified numeric outliers via $1.5 \times \text{IQR}$ (`age`: 359 [0.92%], `campaign`: 2,384 [6.13%], `pdays_was_contacted`: 1,050 [2.70%], `previous`: 4,816 [12.38%], `cons.conf.idx`: 442 [1.14%]; `emp.var.rate` and `cons.price.idx`: 0 [0.00%]).
6. **Leak-Free Pipelines:** All scaling (`StandardScaler`), encoding (`OneHotEncoder` / `OrdinalEncoder`), variance filtering (`VarianceThreshold`), and feature selection steps are contained within `ColumnTransformer` and `Pipeline` objects so every transformation is fit strictly inside training folds.

---

## 3. Exploratory Data Analysis (EDA)

Distributional histograms, contingency tables (`pd.crosstab`), pairwise scatterplots ($n = 5,000$ random sample), and hand-crafted decision-rule visuals (`age` vs. `euribor3m`) highlighted clear conversion regimes:

- **Prior Campaign Success (`poutcome`):** Clients with a previous `success` subscribe at **60.2%** (582 / 967), versus **12.4%** after a prior `failure` and **8.3%** when `nonexistent`.
- **Contact Channel (`contact`):** `cellular` outreach converts at **13.0%** (3,253 / 25,089)—more than double `telephone` at **5.0%** (734 / 14,794).
- **Demographic Segments (`job`):** `student` (**28.8%**) and `retired` (**21.8%**) clients convert highest, whereas `blue-collar` (**6.5%**) and `services` (**7.5%**) convert lowest.
- **Macroeconomic & Seasonal Regimes:** Low-interest periods (`euribor3m < 1.5`) and lower-volume months (`mar`: **50.5%**, `dec`: **48.9%**, `oct`: **43.2%**, `sep`: **39.7%**) outperform high-volume `may` (**6.4%**) by more than 6×.

---

## 4. Methodology & Statistical Classification Framework

### Three-Way Stratified Partitioning (Overfitting Control)
Following the three-way holdout convention, the 38,912 cleaned records were partitioned with stratified sampling (`random_state=27`), preserving the exact **89.8% / 10.2%** class ratio across all splits:
- **Training Set (60% — `23,347` rows):** Used for shared 10-fold stratified CV (`StratifiedKFold(n_splits=10, shuffle=True, random_state=27)`), hyperparameter tuning, feature selection, and out-of-fold threshold optimization.
- **Test Set (20% — `7,782` rows; `6,988` `no` / `794` `yes`):** Used strictly for cross-model comparison and selecting the winning classifier.
- **Held-Out Validation Set (20% — `7,783` rows; `6,989` `no` / `794` `yes`):** Kept completely untouched until final generalization testing of the winning model.

### Classifier Families Benchmarked
1. **Dummy Classifier (Stratified Baseline):** Class-prior random benchmark establishing the minimum performance floor.
2. **$k$-Nearest Neighbors (`KNeighborsClassifier`):** Non-parametric distance classifier over scaled continuous features, tuned over odd $k \in \{3, 5, \dots, 101\}$ via 10-fold `GridSearchCV` (optimal **$k = 3$**, 10-fold CV F1 = `0.3134`). Accompanied by a 2D decision boundary visualization (`age` vs. `cons.conf.idx`).
3. **LASSO Logistic Regression ($L_1$-Penalized Logit):** Parametric linear log-odds model with $L_1$ shrinkage for automatic feature selection. Tuned via 10-fold `GridSearchCV` over $C \in \{0.01, \dots, 10\}$ and `class_weight` (optimal **`C = 0.5`, `class_weight = "balanced"`**, lifting 10-fold CV F1 from `0.2533` to `0.3952`). Kept **51 of 62** encoded features and shrank **11** to exact zero.
4. **Gaussian Naive Bayes + `VarianceThreshold` + Forward SFS:** Applies Bayes' rule under conditional independence. Near-constant one-hot indicators are filtered via `VarianceThreshold(1e-4)` with mild variance smoothing (`var_smoothing=1e-2`), followed by forward `SequentialFeatureSelector` (5-fold CV, F1 scoring) selecting the **15** most informative predictors.
5. **Random Forest (`RandomForestClassifier`):** Bagged ensemble of decorrelated trees with bootstrap **Out-of-Bag (OOB)** validation (`OOB Accuracy = 0.8329`). Tuned via 10-fold `RandomizedSearchCV` (`n_iter=10`, F1 scoring; optimal **`max_leaf_nodes = 200`, `max_features = 4`, `class_weight = "balanced"`**, 10-fold CV F1 = `0.4179`).
6. **Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`):** Sequential additive tree ensemble with `OrdinalEncoder` and early stopping (`max_bins=100`, `max_iter=1000`). Tuned via 10-fold `RandomizedSearchCV` (`n_iter=25`, F1 scoring; optimal **`learning_rate = 0.0324`, `max_depth = 11`, `max_leaf_nodes = 100`, `l2_regularization = 0.275`, `class_weight = "balanced"`**, 10-fold CV F1 = `0.4096`). Diagnosed bias–variance behavior via `LearningCurveDisplay`.

### Leak-Free Threshold Optimization (Soft Classification)
Because `class_weight="balanced"` upweights the minority class ~9× and shifts predicted probabilities upward, the default `0.50` cutoff is suboptimal. For every tunable model, out-of-fold training probabilities (`cross_val_predict(..., cv=skf, method="predict_proba")`) were scanned over $\tau \in [0.10, 0.90)$ to select the F1-maximizing cutoff $\tau^*$ strictly on `x_train` before freezing it for `x_test`.

---

## 5. Benchmark Results

### Model Comparison on the Test Set ($N = 7,782$, Positive Support = `794`)

| Model | Default F1 (`0.50`) | Optimal Threshold ($\tau^*$) | Test Accuracy | Test Precision (Class 1) | Test Recall (Class 1) | Test F1 (Tuned) | Test AP (PR-AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.4098 | **0.62** | 0.8741 | 0.4061 | 0.5063 | **0.4507** | 0.3667 |
| **Gradient Boosting (`HistGradientBoosting`)** | 0.4061 | **0.63** | 0.8723 | 0.4002 | 0.5050 | **0.4465** | 0.3776 |
| **LASSO Logistic Regression ($L_1$, Balanced)** | 0.3798 | **0.68** | 0.8712 | 0.3964 | 0.5013 | **0.4427** | **0.3834** |
| Gaussian Naive Bayes + SFS (15 features) | 0.3783 | 0.50 | 0.8834 | 0.4150 | 0.3476 | 0.3783 | 0.3356 |
| KNN ($k = 3$, Numeric Only) | 0.3026 | 0.34 | 0.8863 | 0.4042 | 0.2418 | 0.3026 | 0.2121 |
| Dummy Baseline (Stratified) | 0.1015 | 0.50 | 0.8111 | 0.0986 | 0.1045 | 0.1015 | 0.1017 |

#### Key Comparative Takeaways
- **Large Threshold-Tuning Gains:** Out-of-fold threshold tuning lifted Test F1 across all three balanced leaders: **LASSO (+6.29 pts)**, **Random Forest (+4.09 pts)**, and **Gradient Boosting (+4.04 pts)**.
- **Tight Top-Three Convergence:** Once evaluated fairly with tuned thresholds, Random Forest (`0.4507`), HistGradientBoosting (`0.4465`), and LASSO (`0.4427`) all landed within `0.008` F1—while LASSO achieved the highest overall probability ranking score (**Test AP = `0.3834`**).
- **Accuracy Paradox Confirmed:** Numeric KNN ($k = 3$) achieved the highest raw accuracy (`88.63%`) yet missed **75.8%** of actual subscribers (`Recall = 0.2418`), demonstrating why F1 and PR-AUC are essential under 90/10 class imbalance.

---

### Final Out-of-Sample Evaluation on the Held-Out Validation Set ($N = 7,783$)

Evaluating the winning **Random Forest ($\tau^* = 0.62$)** once on the completely untouched held-out validation split confirms strong generalization with virtually zero overfitting:

| Dataset Split | Sample Size ($N$) | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Set** | 7,782 | 0.8741 | 0.4061 | 0.5063 | **0.4507** |
| **Held-Out Validation Set** | 7,783 | **0.8756** | **0.4110** | **0.5063** | **0.4537** |

#### Held-Out Validation Confusion Matrix ($N = 7,783$)

| Actual \ Predicted | Predicted No (`0`) | Predicted Yes (`1`) | Total Support |
| :--- | :---: | :---: | :---: |
| **Actual No (`0`)** | 6,413 (TN) | 576 (FP) | 6,989 |
| **Actual Yes (`1`)** | 392 (FN) | 402 (TP) | 794 |
| **Total Predicted** | 6,805 | 978 | 7,783 |

- **Operational Impact:** Dialing only the top **12.6%** of validation leads (`978 / 7,783`) captures **50.6%** of all true term-deposit subscribers (`402 / 794`)—delivering an effective **4.03× lift** over random dialing.
- **Zero Overfitting:** Validation F1 (`0.4537`) matches Test F1 (`0.4507`) and Train OOF F1 (`0.4582`) within `0.005`, confirming a leak-free modeling pipeline.

---

## 6. Key Predictive Drivers & Feature Importance

Across $L_1$ LASSO coefficients, Naive Bayes SFS selection, and Random Forest Gini impurity importance, three signal groups consistently dominate pre-call subscription propensity:
- **Macroeconomic Climate:** `emp.var.rate` (strongest negative LASSO weight: `-1.84`), `cons.price.idx` (strongest positive weight: `+1.30`), and `cons.conf.idx`.
- **Prior Engagement History:** `poutcome_success` (`+0.54`), `poutcome_failure` (`-0.38`), and `pdays_was_contacted` (`+0.13`).
- **Campaign Timing & Client Profile:** Campaign months (`month_aug`: `+0.84`, `month_mar`: `+0.64`, `month_oct`: `+0.34` vs. `month_jun`: `-1.20`, `month_may`: `-1.04`, `month_apr`: `-0.73`), contact channel (`cellular` vs. `telephone`), and life-stage occupations (`student`: `+0.43`, `retired`: `+0.29`).

---

## 7. Repository Structure

```text
├── Bank_Classification_Project.ipynb    # Executed Jupyter/Databricks notebook (43 cells, tables & plots)
├──  bank.csv                         # Bank telemarketing dataset (39,883 records)
├── images/                              # Key exported visual results (PR curves, feature importance, ROC)
├── requirements.txt                     # Python dependencies for local reproducibility
└── README.md                            # Project documentation & benchmark summary
```

---

## 8. How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lambrossklivas/Bank-Classification-Python-Version.git
   cd Bank-Classification-Python-Version
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the notebook:**
   ```bash
   jupyter notebook Bank_Classification_Project.ipynb
   ```

---

## Tech Stack

`Python 3.12` · `scikit-learn` · `pandas` · `NumPy` · `SciPy` · `Matplotlib` · `Seaborn` · `Databricks`

---

## Author

**Lampros Sklivas**  
MSc in Statistics — Athens University of Economics and Business (AUEB)  
BSc in Mathematics — University of Patras
GitHub: [@lambrossklivas](https://github.com/lambrossklivas)
GitHub: [@lambrossklivas](https://github.com/lambrossklivas)
