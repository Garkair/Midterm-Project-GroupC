# ==========================================
# MIDTERM PROJECT: DATA SCIENCE WITH PYTHON
# Social Media Addiction vs Productivity
# ==========================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv("social_media_productivity_6000.csv")

print("\n================ DATA LOADED ================\n")
print(df.head())

# -----------------------------
# 3. DATA INSPECTION
# -----------------------------
print("\n================ DATA INSPECTION ================\n")
print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe())

# -----------------------------
# 4. DATA CLEANING
# -----------------------------
# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing addiction_level with mode if needed
if "addiction_level" in df.columns:
    df["addiction_level"] = df["addiction_level"].fillna(df["addiction_level"].mode()[0])

print("\n================ AFTER CLEANING ================\n")
print("Shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# 5. ADD / DROP DATA
# -----------------------------
# Add engineered features
df["screen_study_ratio"] = df["social_media_hours"] / (df["study_hours"] + 1)
df["notifications_per_hour"] = df["notifications_per_day"] / (df["daily_screen_time"] + 1)
df["sleep_focus_score"] = df["sleep_hours"] * df["focus_score"]

# Drop one redundant column to satisfy add/drop requirement
df = df.drop(columns=["daily_screen_time"])

print("\n================ AFTER ADD / DROP ================\n")
print("Updated columns:")
print(df.columns.tolist())

# -----------------------------
# 6. EDA PREP
# -----------------------------
# Keep a plotting copy with original addiction labels
df_plot = df.copy()

# Make sure addiction levels appear in logical order
label_order = ["Low", "Medium", "High"]

# Correlation matrix for numeric features
corr_cols = [
    "age",
    "social_media_hours",
    "study_hours",
    "sleep_hours",
    "notifications_per_day",
    "focus_score",
    "productivity_score",
    "screen_study_ratio",
    "notifications_per_hour",
    "sleep_focus_score"
]
corr_matrix = df[corr_cols].corr()

print("\n================ CORRELATIONS ================\n")
print(corr_matrix)

# -----------------------------
# 7. PREPROCESSING FOR REGRESSION
# -----------------------------
# Encode addiction_level for regression only
addiction_map = {"Low": 0, "Medium": 1, "High": 2}
df_reg = df.copy()
df_reg["addiction_level"] = df_reg["addiction_level"].map(addiction_map)

# Features and target for regression
X_reg = df_reg.drop("productivity_score", axis=1)
y_reg = df_reg["productivity_score"]

# Train/test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale regression features
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# -----------------------------
# 8. MODEL 1: LINEAR REGRESSION
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg_scaled)

# Regression metrics
lr_mae = mean_absolute_error(y_test_reg, y_pred_lr)
lr_mse = mean_squared_error(y_test_reg, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test_reg, y_pred_lr)

print("\n================ LINEAR REGRESSION RESULTS ================\n")
print(f"MAE:  {lr_mae:.4f}")
print(f"MSE:  {lr_mse:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R2:   {lr_r2:.4f}")

# -----------------------------
# 9. MODEL 2: RANDOM FOREST CLASSIFIER
# -----------------------------
# Predict addiction level as classification target
X_cls = df.drop("addiction_level", axis=1)
y_cls = df["addiction_level"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_classifier.fit(X_train_cls, y_train_cls)
y_pred_rf_cls = rf_classifier.predict(X_test_cls)

# Classification metrics
rf_accuracy = accuracy_score(y_test_cls, y_pred_rf_cls)
cm = confusion_matrix(y_test_cls, y_pred_rf_cls, labels=label_order)

print("\n================ RANDOM FOREST CLASSIFICATION RESULTS ================\n")
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_rf_cls))

# -----------------------------
# 10. FINAL DASHBOARD (ONE PULL)
# -----------------------------
residuals = y_test_reg - y_pred_lr

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(
    "Social Media Addiction & Productivity - Final Project Dashboard",
    fontsize=24,
    fontweight="bold"
)

# ---- Top Row: EDA ----

# 1. Addiction Level Distribution
sns.countplot(
    data=df_plot,
    x="addiction_level",
    order=label_order,
    hue="addiction_level",
    palette="Set2",
    legend=False,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Addiction Level Distribution", fontsize=15, fontweight="bold")
axes[0, 0].set_xlabel("Addiction Level")
axes[0, 0].set_ylabel("Count")

# 2. Productivity by Addiction Level
sns.boxplot(
    data=df_plot,
    x="addiction_level",
    y="productivity_score",
    order=label_order,
    hue="addiction_level",
    palette="Set2",
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Productivity Score by Addiction Level", fontsize=15, fontweight="bold")
axes[0, 1].set_xlabel("Addiction Level")
axes[0, 1].set_ylabel("Productivity Score")

# 3. Correlation Heatmap

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",                 # cleaner numbers
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="gray",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},     # smaller text inside boxes
    ax=axes[0, 2]
)

axes[0, 2].set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")

# Rotate labels for readability
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45, ha="right")
axes[0, 2].set_yticklabels(axes[0, 2].get_yticklabels(), rotation=0)

# ---- Bottom Row: Model Results ----

# 4. Linear Regression Actual vs Predicted
axes[1, 0].scatter(y_test_reg, y_pred_lr, alpha=0.35, s=25, color="#4C9BE8")
min_val = min(y_test_reg.min(), y_pred_lr.min())
max_val = max(y_test_reg.max(), y_pred_lr.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
axes[1, 0].set_title(
    f"Linear Regression: Actual vs Predicted\nR² = {lr_r2:.3f}",
    fontsize=15,
    fontweight="bold"
)
axes[1, 0].set_xlabel("Actual Productivity Score")
axes[1, 0].set_ylabel("Predicted Productivity Score")

# 5. Residual Plot
axes[1, 1].scatter(y_pred_lr, residuals, alpha=0.35, s=25, color="darkorange")
axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=2)
axes[1, 1].set_title(
    f"Residual Plot\nMAE = {lr_mae:.2f}",
    fontsize=15,
    fontweight="bold"
)
axes[1, 1].set_xlabel("Predicted Productivity Score")
axes[1, 1].set_ylabel("Residuals")

# 6. Random Forest Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_order,
    yticklabels=label_order,
    ax=axes[1, 2]
)
axes[1, 2].set_title(
    f"Random Forest Confusion Matrix\nAccuracy = {rf_accuracy:.3f}",
    fontsize=15,
    fontweight="bold"
)
axes[1, 2].set_xlabel("Predicted Label")
axes[1, 2].set_ylabel("True Label")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -----------------------------
# 11. FINAL SUMMARY
# -----------------------------
print("\n================ FINAL PROJECT SUMMARY ================\n")
print("1. The dataset was loaded and inspected successfully.")
print("2. Duplicate rows were removed and missing values were handled.")
print("3. Feature engineering was performed by adding three new variables.")
print("4. One original feature was dropped to reduce redundancy.")
print("5. Two preprocessing techniques were used: encoding and scaling.")
print("6. A Linear Regression model predicted productivity_score.")
print("7. A Random Forest Classifier predicted addiction_level.")
print("8. The dashboard summarizes EDA, regression performance, and classification results.")