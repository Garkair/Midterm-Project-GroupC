"""
Social Media Addiction & Productivity – Full Data Science Pipeline
Covers: Data Cleaning → EDA → Visualization → Modeling → Evaluation

HOW TO RUN:
  1. Place this .py file in the SAME FOLDER as social_media_productivity_6000.csv
  2. Run:  python social_media_analysis.py
  Output charts are saved to that same folder automatically.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ── All paths resolve relative to wherever THIS script is saved ───────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "social_media_productivity_6000.csv")
OUT_EDA  = os.path.join(BASE_DIR, "eda_visualizations.png")
OUT_EVAL = os.path.join(BASE_DIR, "model_evaluation.png")
OUT_FEAT = os.path.join(BASE_DIR, "feature_importance.png")

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE = ["#4C9BE8", "#F4845F", "#56C596"]
ACCENT  = "#4C9BE8"
BG      = "#F7F9FC"
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# =============================================================================
# 1. LOAD
# =============================================================================
print(f"Loading data from: {CSV_PATH}\n")
df_raw = pd.read_csv(CSV_PATH)
print(f"Raw shape: {df_raw.shape}")
print(df_raw.dtypes, "\n")

# =============================================================================
# 2. DATA CLEANING
# =============================================================================
df = df_raw.copy()

# (a) Drop rows that are entirely empty
df = df.dropna(how="all")

# (b) Median imputation for numeric columns  (pandas 3.0 safe — assign back)
num_cols = df.select_dtypes(include="number").columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# (c) Mode imputation for the categorical column  (pandas 3.0 safe)
df["addiction_level"] = df["addiction_level"].fillna(
    df["addiction_level"].mode()[0]
)

# (d) Drop any rows still missing after imputation
df = df.dropna()

# (e) IQR outlier clipping
for col in num_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR,
                           upper=Q3 + 1.5 * IQR)

print(f"Clean shape : {df.shape}")
print(f"Missing left: {df.isnull().sum().sum()}\n")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
df["social_media_ratio"] = (
    df["social_media_hours"] / df["daily_screen_time"].replace(0, np.nan)
)
df["social_media_ratio"] = df["social_media_ratio"].fillna(
    df["social_media_ratio"].median()
)

# Encode addiction_level for modeling
le = LabelEncoder()
df["addiction_encoded"] = le.fit_transform(df["addiction_level"])

# =============================================================================
# 4. EDA VISUALIZATIONS
# =============================================================================
fig = plt.figure(figsize=(18, 20), facecolor=BG)
fig.suptitle("Social Media Addiction & Productivity  -  Exploratory Analysis",
             fontsize=17, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

order      = ["Low", "Medium", "High"]
colors_map = {"Low": PALETTE[2], "Medium": PALETTE[0], "High": PALETTE[1]}

# 4.1 Addiction level distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df["addiction_level"].value_counts().reindex(order)
bars = ax1.bar(order, counts,
               color=[colors_map[o] for o in order],
               edgecolor="white", linewidth=1.2)
for b in bars:
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
             f"{int(b.get_height()):,}", ha="center",
             fontsize=9, fontweight="bold")
ax1.set_title("Addiction Level Distribution")
ax1.set_xlabel("Addiction Level")
ax1.set_ylabel("Count")

# 4.2 Productivity score by addiction level
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df, x="addiction_level", y="productivity_score",
            order=order, palette=colors_map, ax=ax2, linewidth=1.2)
ax2.set_title("Productivity Score by Addiction Level")
ax2.set_xlabel("Addiction Level")
ax2.set_ylabel("Productivity Score")

# 4.3 Social media hours vs productivity
ax3 = fig.add_subplot(gs[0, 2])
for lvl, grp in df.groupby("addiction_level"):
    ax3.scatter(grp["social_media_hours"], grp["productivity_score"],
                alpha=0.25, s=12, label=lvl, color=colors_map[lvl])
tmp  = df[["social_media_hours", "productivity_score"]].dropna()
m, b = np.polyfit(tmp["social_media_hours"], tmp["productivity_score"], 1)
xs   = np.linspace(df["social_media_hours"].min(),
                   df["social_media_hours"].max(), 100)
ax3.plot(xs, m * xs + b, color="black", lw=1.8,
         linestyle="--", label="Trend")
ax3.set_title("Social Media Hours vs Productivity")
ax3.set_xlabel("Social Media Hours / Day")
ax3.set_ylabel("Productivity Score")
ax3.legend(fontsize=8, markerscale=2)

# 4.4 Sleep hours distribution
ax4 = fig.add_subplot(gs[1, 0])
for lvl in order:
    sub = df[df["addiction_level"] == lvl]["sleep_hours"]
    ax4.hist(sub, bins=20, alpha=0.6, label=lvl,
             color=colors_map[lvl], edgecolor="white")
ax4.set_title("Sleep Hours Distribution")
ax4.set_xlabel("Sleep Hours")
ax4.set_ylabel("Frequency")
ax4.legend(fontsize=8)

# 4.5 Correlation heatmap
ax5       = fig.add_subplot(gs[1, 1])
corr_cols = ["daily_screen_time", "social_media_hours", "study_hours",
             "sleep_hours", "notifications_per_day", "focus_score",
             "productivity_score"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax5,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
ax5.set_title("Feature Correlation Heatmap")
ax5.tick_params(axis="x", rotation=45, labelsize=8)
ax5.tick_params(axis="y", labelsize=8)

# 4.6 Focus score vs productivity
ax6 = fig.add_subplot(gs[1, 2])
for lvl, grp in df.groupby("addiction_level"):
    ax6.scatter(grp["focus_score"], grp["productivity_score"],
                alpha=0.25, s=12, label=lvl, color=colors_map[lvl])
ax6.set_title("Focus Score vs Productivity")
ax6.set_xlabel("Focus Score")
ax6.set_ylabel("Productivity Score")
ax6.legend(fontsize=8, markerscale=2)

# 4.7 Notifications per day by addiction level
ax7 = fig.add_subplot(gs[2, 0])
sns.violinplot(data=df, x="addiction_level", y="notifications_per_day",
               order=order, palette=colors_map, ax=ax7, linewidth=1)
ax7.set_title("Notifications / Day by Addiction Level")
ax7.set_xlabel("Addiction Level")
ax7.set_ylabel("Notifications per Day")

# 4.8 Daily screen time vs study hours
ax8 = fig.add_subplot(gs[2, 1])
for lvl, grp in df.groupby("addiction_level"):
    ax8.scatter(grp["daily_screen_time"], grp["study_hours"],
                alpha=0.25, s=12, label=lvl, color=colors_map[lvl])
ax8.set_title("Daily Screen Time vs Study Hours")
ax8.set_xlabel("Daily Screen Time (hrs)")
ax8.set_ylabel("Study Hours / Day")
ax8.legend(fontsize=8, markerscale=2)

# 4.9 Age distribution
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(df["age"], bins=20, color=ACCENT,
         edgecolor="white", linewidth=1.1)
ax9.set_title("Age Distribution of Respondents")
ax9.set_xlabel("Age")
ax9.set_ylabel("Count")

plt.savefig(OUT_EDA, dpi=150, bbox_inches="tight")
plt.close()
print(f"EDA figure saved  ->  {OUT_EDA}\n")

# =============================================================================
# 5. MODEL 1 – Linear Regression  (predict productivity_score)
# =============================================================================
features_reg = ["daily_screen_time", "social_media_hours", "study_hours",
                "sleep_hours", "notifications_per_day", "focus_score",
                "social_media_ratio", "addiction_encoded"]
X_reg = df[features_reg]
y_reg = df["productivity_score"]

scaler       = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42)

lr       = LinearRegression()
lr.fit(X_train_r, y_train_r)
y_pred_r = lr.predict(X_test_r)

mae_r = mean_absolute_error(y_test_r, y_pred_r)
r2_r  = r2_score(y_test_r, y_pred_r)
print(f"Linear Regression  ->  MAE: {mae_r:.2f}  |  R2: {r2_r:.4f}\n")

# =============================================================================
# 6. MODEL 2 – Random Forest Classifier  (predict addiction_level)
# =============================================================================
features_clf = ["age", "daily_screen_time", "social_media_hours",
                "study_hours", "sleep_hours", "notifications_per_day",
                "focus_score", "productivity_score", "social_media_ratio"]
X_clf = df[features_clf]
y_clf = df["addiction_encoded"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

rf = RandomForestClassifier(n_estimators=150, max_depth=8,
                             random_state=42, n_jobs=-1)
rf.fit(X_train_c, y_train_c)
y_pred_c = rf.predict(X_test_c)

print("Random Forest Classifier Report:")
print(classification_report(y_test_c, y_pred_c,
      target_names=le.classes_, zero_division=0))

# =============================================================================
# 7. EVALUATION FIGURES
# =============================================================================
fig2, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig2.suptitle("Model Evaluation", fontsize=16, fontweight="bold")

# 7.1 Actual vs Predicted
ax = axes[0]
ax.scatter(y_test_r, y_pred_r, alpha=0.3, s=12, color=ACCENT)
lims = [min(y_test_r.min(), y_pred_r.min()),
        max(y_test_r.max(), y_pred_r.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
ax.set_title(f"Linear Regression\nActual vs Predicted  (R2={r2_r:.3f})")
ax.set_xlabel("Actual Productivity Score")
ax.set_ylabel("Predicted Productivity Score")
ax.legend(fontsize=9)

# 7.2 Residual plot
ax        = axes[1]
residuals = y_test_r - y_pred_r
ax.scatter(y_pred_r, residuals, alpha=0.3, s=12, color=PALETTE[1])
ax.axhline(0, color="black", lw=1.5, linestyle="--")
ax.set_title(f"Residual Plot\n(MAE = {mae_r:.2f})")
ax.set_xlabel("Predicted Productivity Score")
ax.set_ylabel("Residuals")

# 7.3 Confusion matrix
ax   = axes[2]
cm   = confusion_matrix(y_test_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Random Forest\nConfusion Matrix")

plt.tight_layout()
plt.savefig(OUT_EVAL, dpi=150, bbox_inches="tight")
plt.close()
print(f"Evaluation figure saved  ->  {OUT_EVAL}\n")

# Feature importance
fig3, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
importances = pd.Series(rf.feature_importances_,
                        index=features_clf).sort_values()
importances.plot(kind="barh", color=ACCENT, edgecolor="white", ax=ax)
ax.set_title("Random Forest - Feature Importance\n(Predicting Addiction Level)",
             fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(OUT_FEAT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Feature importance saved  ->  {OUT_FEAT}")
print("\nAll done.")