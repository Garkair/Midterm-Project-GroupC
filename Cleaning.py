import pandas as pd


def load_dataset(file_name):
    """
    Load a CSV file safely.

    Args:
        file_name (str): CSV filename.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_name)
        print("Dataset loaded successfully.\n")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return pd.DataFrame()


def clean_dataset(df):
    """
    Perform data cleaning:
    - Remove duplicates
    - Handle missing values
    - Drop an unnecessary column
    - Remove invalid values

    Args:
        df (pandas.DataFrame): Raw dataset.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """
    print("Initial shape:", df.shape)

    # 1. Remove duplicates
    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)

    # 2. Show missing values
    print("\nMissing values BEFORE cleaning:")
    print(df.isnull().sum())

    strategy = input("\nChoose cleaning strategy (fill/drop): ").strip().lower()

    if strategy == "fill":
        df = df.ffill()
        print("Missing values filled using forward fill.")
    elif strategy == "drop":
        df = df.dropna()
        print("Rows with missing values dropped.")
    else:
        print("Invalid choice. No cleaning applied.")

    print("\nMissing values AFTER cleaning:")
    print(df.isnull().sum())

    # 3. Drop a column
    print("\nColumns in dataset:")
    print(list(df.columns))

    drop_col = input("\nEnter a column to drop (or press Enter to skip): ").strip()

    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])
        print(f"Column '{drop_col}' dropped.")
    elif drop_col != "":
        print("Column not found. Skipping.")

    # 4. Remove invalid negative values from numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df = df[df[col] >= 0]

    print("\nFinal shape after cleaning:", df.shape)

    return df


def main():
    file_name = "social_media_productivity_6000.csv"

    df = load_dataset(file_name)

    if df.empty:
        print("Dataset is empty. Exiting.")
        return

    print("\n--- DATASET OVERVIEW ---")
    print(df.head())

    print("\n--- DATASET INFO ---")
    df.info()

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # Clean the dataset
    df = clean_dataset(df)

    print("\n--- CLEANED DATA PREVIEW ---")
    print(df.head())

    # Save the cleaned dataset
    df.to_csv("cleaned_social_media_productivity.csv", index=False)
    print("\nCleaned dataset saved as 'cleaned_social_media_productivity.csv'")


if __name__ == "__main__":
    main()