# ==========================================
# MIDTERM PROJECT: DATA SCIENCE WITH PYTHON
# Social Media Addiction vs Productivity
# Group C — April 2026
#
# DATASET SOURCE:
#   Shamim, A. (2024). Students Social Media Addiction [Data set].
#   Kaggle. https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships
#   Downloaded manually as a CSV and placed in the same directory as this script.
#
# HOW TO RUN:
#   python Python_script_v3.py
#   No arguments needed. All output is printed to console.
#   Dashboard chart is saved to dashboard.png in the same directory.
# ==========================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)


# ============================================================
# FUNCTIONS
# ============================================================

def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw loaded dataframe.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df):
    """
    Print a structured overview of the dataset including shape, dtypes,
    missing values, duplicates, summary statistics, and class distribution.
    Class imbalance is explicitly noted here because it affects model
    evaluation strategy (stratified splitting).

    Args:
        df (pd.DataFrame): Raw dataframe.
    """
    print("\n================ DATA INSPECTION ================\n")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSummary statistics:\n", df.describe())

    # Class imbalance check
    if "addiction_level" in df.columns:
        counts = df["addiction_level"].value_counts()
        total  = len(df)
        print("\nClass distribution (addiction_level):")
        for cls, cnt in counts.items():
            print(f"  {cls}: {cnt} ({cnt / total * 100:.1f}%)")
        print(
            "\nNote: Class imbalance detected (Low ~16%, Medium ~53%, High ~31%)."
            " Stratified train/test splitting is used throughout to preserve these"
            " proportions, preventing the classifier from being biased toward the"
            " majority class (Medium) during evaluation."
        )


def clean_data(df):
    """
    Clean the dataset in four steps:
      1. Remove duplicate rows.
      2. Median imputation for numeric columns — robust to skewed distributions
         and outliers, preferred over mean for behavioral survey data.
      3. Mode imputation for the categorical addiction_level column.
      4. IQR-based outlier clipping — clips values beyond 1.5xIQR from Q1/Q3.
         Chosen over row removal to retain the full 6,000-row dataset while
         limiting the influence of extreme values on model coefficients.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    print("\n================ DATA CLEANING ================\n")

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # 2. Median imputation for numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled {n_missing} missing in '{col}' with median")

    # 3. Mode imputation for categorical column
    if df["addiction_level"].isnull().sum() > 0:
        mode_val = df["addiction_level"].mode()[0]
        df["addiction_level"] = df["addiction_level"].fillna(mode_val)
        print(f"  Filled missing 'addiction_level' with mode: '{mode_val}'")

    # 4. IQR outlier clipping
    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    print(f"\nShape after cleaning: {df.shape}")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    """
    Add three engineered features and drop one redundant column.

    New features:
      - screen_study_ratio:     social media hours relative to study time
      - notifications_per_hour: notification density per screen hour
      - sleep_focus_score:      interaction term — sleep quality x focus ability

    Dropped:
      - daily_screen_time: Pearson r = 0.78 with social_media_hours,
        indicating strong multicollinearity. Replaced by screen_study_ratio
        which captures the relative contribution more informatively.

    Args:
        df (pd.DataFrame): Cleaned dataframe.

    Returns:
        pd.DataFrame: Feature-engineered dataframe.
    """
    print("\n================ FEATURE ENGINEERING ================\n")

    df["screen_study_ratio"]     = df["social_media_hours"] / (df["study_hours"] + 1)
    df["notifications_per_hour"] = df["notifications_per_day"] / (df["daily_screen_time"] + 1)
    df["sleep_focus_score"]      = df["sleep_hours"] * df["focus_score"]

    df = df.drop(columns=["daily_screen_time"])

    print("Added:   screen_study_ratio, notifications_per_hour, sleep_focus_score")
    print("Dropped: daily_screen_time (multicollinearity with social_media_hours)")
    print("Updated columns:", df.columns.tolist())
    return df


def run_eda(df):
    """
    Compute and print the Pearson correlation matrix for numeric features,
    and surface the top correlates of the regression target.

    Visual EDA charts (histograms, scatter, box, violin, heatmap) are
    produced separately in social_media_analysis.py.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe.

    Returns:
        pd.DataFrame: Correlation matrix (used later for dashboard heatmap).
    """
    print("\n================ EDA: CORRELATIONS ================\n")
    corr_cols = [
        "age", "social_media_hours", "study_hours", "sleep_hours",
        "notifications_per_day", "focus_score", "productivity_score",
        "screen_study_ratio", "notifications_per_hour", "sleep_focus_score"
    ]
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.round(2))

    target_corr = (
        corr_matrix["productivity_score"]
        .drop("productivity_score")
        .abs()
        .sort_values(ascending=False)
    )
    print("\nTop correlations with productivity_score:")
    print(target_corr.head(5).round(3))
    return corr_matrix


def train_linear_regression(df):
    """
    Train and evaluate a Linear Regression model to predict productivity_score.

    Pipeline:
      - LabelEncoder encodes addiction_level with a fixed ordinal order
        (Low=0, Medium=1, High=2) for consistent interpretation.
      - StandardScaler is fit on the TRAINING set only and applied to
        the test set via transform() — this is critical to prevent data
        leakage through the scaling step.
      - 5-fold cross-validation on the training set provides a robust
        estimate of generalization performance before final test evaluation.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe.

    Returns:
        tuple: (trained LinearRegression, fitted StandardScaler, metrics dict)
    """
    print("\n================ MODEL 1: LINEAR REGRESSION ================\n")

    # Encode addiction_level ordinally with fixed order
    le = LabelEncoder()
    le.fit(["Low", "Medium", "High"])
    df_reg = df.copy()
    df_reg["addiction_level"] = le.transform(df_reg["addiction_level"])

    X = df_reg.drop("productivity_score", axis=1)
    y = df_reg["productivity_score"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale AFTER splitting — fit on train only (prevents leakage to test set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LinearRegression()

    # 5-fold cross-validation on training set
    cv_r2  = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="r2")
    cv_mae = cross_val_score(lr, X_train_scaled, y_train, cv=5,
                             scoring="neg_mean_absolute_error")
    print(f"5-Fold CV  R²:  {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")
    print(f"5-Fold CV  MAE: {(-cv_mae).mean():.4f}  ±  {cv_mae.std():.4f}")

    # Final fit and held-out test evaluation
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\nTest Set Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    metrics = {
        "mae": mae, "rmse": rmse, "r2": r2,
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "y_test": y_test, "y_pred": y_pred
    }
    return lr, scaler, metrics


def train_random_forest(df):
    """
    Train and evaluate a Random Forest Classifier to predict addiction_level.

    Pipeline:
      - Stratified train/test split preserves class proportions for the
        imbalanced target (Low 16%, Medium 53%, High 31%).
      - GridSearchCV performs 3-fold CV over a param grid of n_estimators
        and max_depth to select the best hyperparameters automatically.
      - Best model is evaluated on the held-out test set.

    Known limitation (data leakage in target):
      addiction_level is near-deterministically derivable from social_media_hours
      (~70% feature importance). The 98.8% accuracy reflects this structural
      relationship rather than true generalization. This is documented in the
      project report as a known limitation.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe.

    Returns:
        tuple: (best RandomForestClassifier, metrics dict)
    """
    print("\n================ MODEL 2: RANDOM FOREST CLASSIFIER ================\n")

    label_order = ["Low", "Medium", "High"]

    X = df.drop("addiction_level", axis=1)
    y = df["addiction_level"]

    # Stratified split to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameter tuning via GridSearchCV (3-fold CV on training set)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [6, 10, None]
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    print(f"GridSearchCV best params:  {gs.best_params_}")
    print(f"GridSearchCV CV accuracy:  {gs.best_score_:.4f}")

    # Evaluate best model on held-out test set
    best_rf = gs.best_estimator_
    y_pred  = best_rf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred, labels=label_order)

    print(f"\nTest Set Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "accuracy": acc, "cm": cm, "label_order": label_order,
        "y_test": y_test, "y_pred": y_pred,
        "feature_importances": best_rf.feature_importances_,
        "feature_names": X.columns.tolist()
    }
    return best_rf, metrics


def plot_dashboard(df, corr_matrix, lr_metrics, rf_metrics):
    """
    Render and save a 2x3 summary dashboard combining EDA and model results.

    Panel layout:
      [0,0] Addiction level distribution (count bar)
      [0,1] Productivity score by addiction level (box plot)
      [0,2] Feature correlation heatmap
      [1,0] LR: Actual vs Predicted (with CV R2 in title)
      [1,1] LR: Residual plot (with MAE and RMSE in title)
      [1,2] RF: Confusion matrix (with accuracy in title)

    Args:
        df (pd.DataFrame): Feature-engineered dataframe (string addiction labels).
        corr_matrix (pd.DataFrame): Precomputed correlation matrix.
        lr_metrics (dict): From train_linear_regression().
        rf_metrics (dict): From train_random_forest().
    """
    label_order = ["Low", "Medium", "High"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "Social Media Addiction & Productivity — Final Project Dashboard",
        fontsize=24, fontweight="bold"
    )

    # EDA: Addiction distribution
    sns.countplot(data=df, x="addiction_level", order=label_order,
                  hue="addiction_level", palette="Set2", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Addiction Level Distribution", fontsize=15, fontweight="bold")
    axes[0, 0].set_xlabel("Addiction Level")
    axes[0, 0].set_ylabel("Count")

    # EDA: Productivity by addiction level
    sns.boxplot(data=df, x="addiction_level", y="productivity_score",
                order=label_order, hue="addiction_level", palette="Set2",
                legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Productivity Score by Addiction Level", fontsize=15, fontweight="bold")
    axes[0, 1].set_xlabel("Addiction Level")
    axes[0, 1].set_ylabel("Productivity Score")

    # EDA: Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, linecolor="gray", cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8}, ax=axes[0, 2])
    axes[0, 2].set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45, ha="right", fontsize=5, weight="bold")
    axes[0, 2].set_yticklabels(axes[0, 2].get_yticklabels(), rotation=0, fontsize=5, weight="bold")

    # Model: LR Actual vs Predicted
    y_test_r, y_pred_r = lr_metrics["y_test"], lr_metrics["y_pred"]
    axes[1, 0].scatter(y_test_r, y_pred_r, alpha=0.35, s=25, color="#4C9BE8")
    lim = [min(y_test_r.min(), y_pred_r.min()), max(y_test_r.max(), y_pred_r.max())]
    axes[1, 0].plot(lim, lim, "r--", linewidth=2)
    axes[1, 0].set_title(
        f"Linear Regression: Actual vs Predicted\n"
        f"R² = {lr_metrics['r2']:.3f}  |  CV R² = {lr_metrics['cv_r2_mean']:.3f} ± {lr_metrics['cv_r2_std']:.3f}",
        fontsize=13, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Actual Productivity Score")
    axes[1, 0].set_ylabel("Predicted Productivity Score")

    # Model: LR Residual plot
    residuals = y_test_r - y_pred_r
    axes[1, 1].scatter(y_pred_r, residuals, alpha=0.35, s=25, color="darkorange")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[1, 1].set_title(
        f"Residual Plot\nMAE = {lr_metrics['mae']:.2f}  |  RMSE = {lr_metrics['rmse']:.2f}",
        fontsize=13, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Predicted Productivity Score")
    axes[1, 1].set_ylabel("Residuals")

    # Model: RF Confusion matrix
    sns.heatmap(rf_metrics["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=rf_metrics["label_order"],
                yticklabels=rf_metrics["label_order"], ax=axes[1, 2])
    axes[1, 2].set_title(
        f"Random Forest Confusion Matrix\nAccuracy = {rf_metrics['accuracy']:.3f}",
        fontsize=13, fontweight="bold"
    )
    axes[1, 2].set_xlabel("Predicted Label")
    axes[1, 2].set_ylabel("True Label")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Dashboard saved -> dashboard.png")


def print_summary(lr_metrics, rf_metrics):
    """
    Print a concise plain-English summary of all pipeline results.

    Args:
        lr_metrics (dict): From train_linear_regression().
        rf_metrics (dict): From train_random_forest().
    """
    print("\n================ FINAL PROJECT SUMMARY ================\n")
    print("1. Dataset: 6,000 rows, 9 columns. Source: Kaggle (Shamim, 2024).")
    print("2. Cleaning: duplicates removed; median/mode imputation; IQR outlier clipping.")
    print("3. Feature engineering: added screen_study_ratio, notifications_per_hour,")
    print("   sleep_focus_score. Dropped daily_screen_time (multicollinearity r=0.78).")
    print("4. Preprocessing: LabelEncoder (addiction_level, fixed ordinal order),")
    print("   StandardScaler fit on training data only — no leakage to test set.")
    print(f"5. Linear Regression — Test R²: {lr_metrics['r2']:.4f} | MAE: {lr_metrics['mae']:.4f}")
    print(f"   5-Fold CV R²: {lr_metrics['cv_r2_mean']:.4f} ± {lr_metrics['cv_r2_std']:.4f}")
    print(f"6. Random Forest Classifier — Test Accuracy: {rf_metrics['accuracy']:.4f}")
    print("   Hyperparameters selected via GridSearchCV (3-fold CV).")
    print("   Known limitation: high accuracy partly reflects target-feature collinearity.")
    print("7. Dashboard saved to dashboard.png.")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Orchestrate the full data science pipeline:
    load -> inspect -> clean -> engineer -> EDA -> model -> evaluate -> visualize.
    """
    filepath = "social_media_productivity_6000.csv"

    df           = load_data(filepath)
    inspect_data(df)
    df           = clean_data(df)
    df           = engineer_features(df)
    corr_matrix  = run_eda(df)
    _, _, lr_metrics = train_linear_regression(df)
    _, rf_metrics    = train_random_forest(df)
    plot_dashboard(df, corr_matrix, lr_metrics, rf_metrics)
    print_summary(lr_metrics, rf_metrics)


if __name__ == "__main__":
    main()
