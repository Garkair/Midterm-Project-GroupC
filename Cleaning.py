"""
Cleaning.py — Standalone Data Cleaning Module
Group C Midterm Project | April 2026

Loads the raw dataset, applies all cleaning steps, and saves the
cleaned output as cleaned_social_media_productivity.csv.

No user input required. All decisions are hardcoded and documented
below to ensure full reproducibility.

HOW TO RUN:
    python Cleaning.py
"""

import pandas as pd


# Column to drop after cleaning.
# daily_screen_time is removed because it is highly collinear with
# social_media_hours (Pearson r = 0.78), making it redundant as a
# predictor once social_media_hours is included in the model.
COLUMN_TO_DROP = "daily_screen_time"


def load_dataset(file_name):
    """
    Load a CSV file safely with error handling.

    Args:
        file_name (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset, or empty DataFrame on failure.
    """
    try:
        df = pd.read_csv(file_name)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return pd.DataFrame()


def clean_dataset(df):
    """
    Apply all cleaning steps in a reproducible, non-interactive way.

    Steps:
      1. Remove duplicate rows.
      2. Median imputation for numeric columns.
         Rationale: median is robust to skewed distributions and outliers,
         making it more appropriate than mean or forward fill for
         behavioral survey data where row order is arbitrary.
      3. Mode imputation for the categorical addiction_level column.
      4. Drop the COLUMN_TO_DROP column (see module constant above).
      5. Remove any rows with remaining negative numeric values.
         These would represent physically impossible readings
         (e.g. negative sleep hours or screen time).

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    print("=== CLEANING STEPS ===\n")
    print(f"Initial shape: {df.shape}")

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}  ({before - len(df)} removed)")

    # 2. Median imputation for numeric columns
    print("\nMissing values BEFORE imputation:")
    print(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  '{col}': filled {n_missing} missing values with median ({median_val:.4f})")

    # 3. Mode imputation for categorical column
    if "addiction_level" in df.columns and df["addiction_level"].isnull().sum() > 0:
        mode_val = df["addiction_level"].mode()[0]
        n_missing = df["addiction_level"].isnull().sum()
        df["addiction_level"] = df["addiction_level"].fillna(mode_val)
        print(f"  'addiction_level': filled {n_missing} missing values with mode ('{mode_val}')")

    print("\nMissing values AFTER imputation:")
    print(df.isnull().sum())

    # 4. Drop redundant column
    if COLUMN_TO_DROP in df.columns:
        df = df.drop(columns=[COLUMN_TO_DROP])
        print(f"\nDropped column: '{COLUMN_TO_DROP}' (multicollinearity with social_media_hours)")
    else:
        print(f"\nColumn '{COLUMN_TO_DROP}' not found — skipping drop step.")

    # 5. Remove rows with negative numeric values (physically impossible)
    numeric_cols_remaining = df.select_dtypes(include=["int64", "float64"]).columns
    before_neg = len(df)
    for col in numeric_cols_remaining:
        df = df[df[col] >= 0]
    removed_neg = before_neg - len(df)
    if removed_neg > 0:
        print(f"Removed {removed_neg} rows with negative numeric values.")

    print(f"\nFinal shape after cleaning: {df.shape}")
    return df


def main():
    """
    Run the full cleaning pipeline and save the result to CSV.
    """
    file_name    = "social_media_productivity_6000.csv"
    output_name  = "cleaned_social_media_productivity.csv"

    df = load_dataset(file_name)
    if df.empty:
        print("Dataset is empty or failed to load. Exiting.")
        return

    print("--- DATASET OVERVIEW ---")
    print(df.head())
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    df = clean_dataset(df)

    print("\n--- CLEANED DATA PREVIEW ---")
    print(df.head())

    df.to_csv(output_name, index=False)
    print(f"\nCleaned dataset saved as '{output_name}'")


if __name__ == "__main__":
    main()
