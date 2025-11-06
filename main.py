import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# --------------------------
# Setup logging
# --------------------------
logging.basicConfig(
    filename='datacleaner.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------
# Load configuration
# --------------------------
with open('config.json', 'r') as f:
    config = json.load(f)

OUTPUT_FOLDER = config.get("output_folder", "outputs")
OUTLIER_THRESHOLD = config.get("outlier_threshold", 1.5)
MISSING_STRATEGY = config.get("missing_value_strategy", "auto")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------
# Helper functions
# --------------------------
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col].fillna(method='ffill', inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    logging.info("Missing values handled successfully.")
    return df


def remove_outliers(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - OUTLIER_THRESHOLD * IQR, Q3 + OUTLIER_THRESHOLD * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        if removed > 0:
            logging.info(f"Removed {removed} outliers from '{col}'")
    return df


def convert_date_columns(df):
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            logging.info(f"Converted '{col}' to datetime.")
        except (ValueError, TypeError):
            continue
    return df


def generate_summary(df, output_path):
    summary_lines = []
    summary_lines.append("===== DATA SUMMARY REPORT =====\n")
    summary_lines.append(f"Total Rows: {len(df)}\n")
    summary_lines.append(f"Total Columns: {len(df.columns)}\n\n")

    summary_lines.append("Column-wise Missing Values:\n")
    summary_lines.append(str(df.isnull().sum()) + "\n\n")

    summary_lines.append("Data Types:\n")
    summary_lines.append(str(df.dtypes) + "\n\n")

    summary_lines.append("Descriptive Statistics:\n")
    summary_lines.append(str(df.describe(include='all')) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    logging.info(f"Summary report saved at {output_path}")


# --------------------------
# Main cleaning function
# --------------------------
def clean_data(input_path):
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded dataset: {input_path}")
        print(f"Loaded dataset: {input_path}")
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # Extract base dataset name (e.g., raw_data.csv → raw_data)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Step 1: Remove Duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    logging.info(f"Removed {removed} duplicate rows.")

    # Step 2: Handle Missing Values
    df = handle_missing_values(df)

    # Step 3: Convert Date Columns
    df = convert_date_columns(df)

    # Step 4: Remove Outliers
    df = remove_outliers(df)

    # Step 5: Generate Summary Report
    summary_path = os.path.join(OUTPUT_FOLDER, f"summary_{base_name}.txt")
    generate_summary(df, summary_path)

    # Step 6: Save Cleaned Data
    output_path = os.path.join(OUTPUT_FOLDER, f"cleaned_data_{base_name}.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved at {output_path}")

    print("\nData cleaning completed successfully!")
    print(f"Cleaned data saved to: {output_path}")
    print(f"Summary report saved to: {summary_path}")


# --------------------------
# Run Script
# --------------------------
if __name__ == "__main__":
    INPUT_FILE = "data/raw_data.csv"
    clean_data(INPUT_FILE)

    print("\n Data cleaning script completed successfully!")