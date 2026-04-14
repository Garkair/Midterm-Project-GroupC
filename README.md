# Social Media Addiction & Productivity
### Group C — Data Science with Python | Midterm Project

A full data science pipeline analyzing the relationship between social media usage and student productivity. Covers data cleaning, exploratory analysis, feature engineering, and two machine learning models (Linear Regression + Random Forest Classifier).

---

## 📁 Project Structure

```
project/
│
├── README.md                            ← You are here
│
├── data/
│   ├── social_media_productivity_6000.csv     ← Raw dataset (source of truth)
│   └── cleaned_social_media_productivity.csv  ← Output of Cleaning.py
│
├── scripts/
│   ├── Cleaning.py                      ← Step 1: Clean & save the dataset
│   ├── social_media_analysis.py         ← Step 2: EDA + 9 visualizations (saves PNGs)
│   └── Python_script_v3.py             ← Step 3: Full ML pipeline (models + dashboard)
│
├── batch/
│   ├── 3runCleaning.bat                 ← Double-click to run Step 1 (Windows)
│   ├── 1runAnalysis.bat                 ← Double-click to run Step 2 (Windows)
│   └── 2runPipeline.bat                 ← Double-click to run Step 3 (Windows)
│
├── outputs/
│   ├── eda_visualizations.png           ← 9-panel EDA figure
│   ├── model_evaluation.png             ← LR actual/predicted, residuals, RF confusion matrix
│   ├── feature_importance.png           ← Random Forest feature importance bar chart
│   └── Social_media_addtion_plot_new.png
│
└── Group_C_Report.docx                  ← Final written report
```

> **Note:** The CSV files are expected in the **same directory** as whichever script you are running. If you keep the flat structure (all files together), everything works out of the box.

---

## 🔧 Requirements

- **Python 3.10+**
- The following packages:

| Package | Version Used |
|---|---|
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

### Install all dependencies at once

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Double-click (Windows, recommended for beginners)

Three `.bat` scripts are included. Run them **in order** by double-clicking:

| Script | Step | What it runs |
|---|---|---|
| `3runCleaning.bat` | Step 1 | `Cleaning.py` — cleans the raw data |
| `1runAnalysis.bat` | Step 2 | `social_media_analysis.py` — EDA + visualizations |
| `2runPipeline.bat` | Step 3 | `Python_script_v3.py` — full ML pipeline |

> Each script opens a terminal window, shows live output while running, and displays a summary of saved output files when finished. **The window stays open** until you press any key.

---

### Option B — Command line (all platforms)

All three scripts are **standalone** — run them independently in order.

### Step 1 — Clean the Data

```bash
python Cleaning.py
```

**What it does:**
- Loads `social_media_productivity_6000.csv`
- Removes duplicates
- Fills missing values (median for numeric columns, mode for `addiction_level`)
- Drops `daily_screen_time` (high collinearity with `social_media_hours`)
- Removes rows with any remaining negative numeric values
- Saves the result to `cleaned_social_media_productivity.csv`

**Output:**
```
Loaded dataset: 6000 rows, 9 columns
Duplicates removed: 0
  Filled 120 missing in 'age' with median
  ...
Final shape after cleaning: (6000, 8)
Cleaned dataset saved as 'cleaned_social_media_productivity.csv'
```

---

### Step 2 — Exploratory Data Analysis

```bash
python social_media_analysis.py
```

**What it does:**
- Loads the raw CSV, applies cleaning + IQR outlier clipping
- Engineers one feature (`social_media_ratio`)
- Produces a **9-panel EDA figure** saved to `eda_visualizations.png`
- Trains both models and saves evaluation figures:
  - `model_evaluation.png` — LR actual vs predicted, residual plot, RF confusion matrix
  - `feature_importance.png` — Random Forest feature importance

**Visualizations produced:**

| Panel | Chart Type | What it shows |
|---|---|---|
| 1 | Bar chart | Addiction level distribution |
| 2 | Box plot | Productivity score by addiction level |
| 3 | Scatter + trend | Social media hours vs productivity |
| 4 | Histogram | Sleep hours by addiction group |
| 5 | Heatmap | Feature correlation matrix |
| 6 | Scatter | Focus score vs productivity |
| 7 | Violin plot | Notifications per day by addiction level |
| 8 | Scatter | Daily screen time vs study hours |
| 9 | Histogram | Age distribution of respondents |

> Charts are **saved as PNG files** — no display window required. Safe to run headless.

---

### Step 3 — Machine Learning Pipeline

```bash
python Python_script_v3.py
```

**What it does:**
- Runs the complete pipeline end-to-end: load → inspect → clean → engineer features → EDA → train models → evaluate → dashboard
- **Model 1:** Linear Regression predicting `productivity_score`
  - 5-fold cross-validation on training set
  - Reports MAE, RMSE, R²
- **Model 2:** Random Forest Classifier predicting `addiction_level`
  - GridSearchCV hyperparameter tuning (n_estimators, max_depth)
  - Reports accuracy, precision, recall, F1, confusion matrix
- Saves `dashboard.png` — a 6-panel summary combining EDA and model results

**Expected console output (abbreviated):**
```
Loaded dataset: 6000 rows, 9 columns

================ DATA INSPECTION ================
...
Class distribution (addiction_level):
  Medium: 3184 (53.1%)
  High:   1857 (31.0%)
  Low:     959 (16.0%)

================ MODEL 1: LINEAR REGRESSION ================
5-Fold CV  R²:  0.8445  ±  0.0083
5-Fold CV  MAE: 8.2840  ±  0.1802

Test Set Results:
  MAE:  8.2562
  RMSE: ~11.0
  R²:   0.8405

================ MODEL 2: RANDOM FOREST CLASSIFIER ================
GridSearchCV best params:  {'max_depth': 10, 'n_estimators': 100}
GridSearchCV CV accuracy:  0.9806
Test Set Accuracy: 0.9875
...
Dashboard saved -> dashboard.png
```

---

## 📊 Dataset

**Source:** Shamim, A. (2024). *Students Social Media Addiction* [Data set]. Kaggle.
https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Respondent age (15–39) |
| `daily_screen_time` | numeric | Total daily screen time (hours) |
| `social_media_hours` | numeric | Daily social media usage (hours) |
| `study_hours` | numeric | Daily study time (hours) |
| `sleep_hours` | numeric | Nightly sleep (hours) |
| `notifications_per_day` | numeric | App notifications received per day |
| `focus_score` | numeric | Self-reported focus ability (0–100) |
| `addiction_level` | categorical | Low / Medium / High |
| `productivity_score` | numeric | Productivity score (0–100) |

**6,000 rows | 9 columns | 120 missing values per column (2%) | 0 duplicates**

---

## 🧠 Key Results

| Model | Task | CV Score | Test Score |
|---|---|---|---|
| Linear Regression | Predict `productivity_score` | R² = 0.8445 ± 0.0083 | R² = 0.8405, MAE = 8.26 |
| Random Forest | Predict `addiction_level` | Accuracy = 98.06% | Accuracy = 98.75%, F1 ≈ 0.99 |

> ⚠️ The Random Forest's high accuracy is partly explained by target-feature collinearity — `addiction_level` is near-deterministically derivable from `social_media_hours`. This is documented as a known limitation in the report (Section 4.2).

---

## ⚠️ Known Limitations

- **Target-feature collinearity:** `social_media_hours` accounts for ~70% of Random Forest feature importance because `addiction_level` is effectively derived from usage hours in this dataset.
- **Linear assumption:** Linear Regression assumes linearity between features and `productivity_score`. The mild heteroskedasticity in the residual plot suggests a non-linear model (e.g., Gradient Boosting) could improve predictions.
- **Self-reported data:** All variables are self-reported, introducing potential response bias.

---

## 👥 Group Contributions

| Member | Responsibility |
|---|---|
| Member 1 | Data acquisition, `Cleaning.py`, data cleaning section of report |
| Member 2 | EDA, all 9 visualizations, `social_media_analysis.py`, EDA section of report |
| Member 3 | Feature engineering, ML pipeline, `Python_script_v3.py`, ML sections of report |
| All Members | Code review, report editing, final QA |

---

## 📄 Report

The full written report is in `Group_C_Report.docx`. It covers:
1. Introduction to the Dataset
2. Data Cleaning & Preprocessing Steps
3. Exploratory Data Analysis & Key Insights
4. Model Building & Evaluation
5. Conclusion
6. References
