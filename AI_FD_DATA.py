"""
BA3.2 AI Solution
AI/FD Component 1 of 3: DATA

File:
    AI_FD_DATA.py

Purpose:
    Everything that happens to the data BEFORE a model sees it:
        - configuration (file locations, target column, random seed)
        - loading the CSV and exploring it
        - cleaning (repeated records, ID columns, missing labels)
        - splitting into training and test sets
        - testing every feature for fraud signal

The other two components import from this file:
    AI_FD_ML.py        classical machine learning, evaluation, prediction
    AI_FD_ADVANCED.py  deep learning and time-series analysis
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# CONFIGURATION

DATA_DIR = "DATA"
MODEL_DIR = "models"

DATASET_FILE = os.path.join(DATA_DIR, "Fraud_Detection_Dataset.csv")

# The saved file is now a whole Pipeline (preprocessing + model), so there
# is no separate scaler file any more.
MODEL_FILE = os.path.join(MODEL_DIR, "ai_fd_model.pkl")

# Column we're predicting
TARGET_COLUMN = "Fraudulent"

# Time column. In Fraud_Detection_Dataset.csv it is already a number (hour
# of day, 0-23). If a dataset has "HH:MM" text instead, convert_time_column
# turns it into minutes-since-midnight automatically.
TIME_COLUMN = "Time_of_Transaction"

# Identifier columns: row labels / user labels, not real features.
# Transaction_ID is also used to remove repeated records of the same
# transaction. User_ID is a number but means nothing as a number (user
# 4000 is not "more" than user 10), so it is dropped too.
TRANSACTION_ID_COLUMN = "Transaction_ID"
ID_COLUMNS = ["Transaction_ID", "User_ID"]

# Text columns with more unique values than this (IDs, free text, raw dates)
# would explode one-hot encoding, so they are dropped with a warning.
MAX_CATEGORIES = 50

RANDOM_STATE = 42


# DIRECTORY SETUP

def create_directories():
    """
    Create the data/model directories if they don't already exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# DATA LOADING

def load_dataset(file_path=DATASET_FILE):
    """
    Load the dataset from a CSV file.

    Returns:
        pandas.DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    return data


# DATA EXPLORATION

def explore_dataset(data):
    """
    Print basic information about the dataset.
    """
    print("\n========== DATASET INFORMATION ==========")

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nFirst five records:")
    print(data.head())

    print("\nDataset information:")
    print(data.info())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nStatistical summary:")
    print(data.describe())

    if TARGET_COLUMN in data.columns:
        print(f"\n{TARGET_COLUMN} class balance:")
        print(data[TARGET_COLUMN].value_counts())
        print(f"\n{TARGET_COLUMN} class balance (%):")
        print((data[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))


# TIME COLUMN CONVERSION

def convert_time_column(data, time_column=TIME_COLUMN, verbose=True):
    """
    Convert an "HH:MM" time column into minutes-since-midnight so it can
    be used as a numeric feature (e.g. "08:35" -> 515).

    If the column is already numeric (as in Fraud_Detection_Dataset.csv,
    where it is the hour of day) it is left alone.
    """
    data = data.copy()

    if time_column in data.columns and not pd.api.types.is_numeric_dtype(data[time_column]):
        parsed = pd.to_datetime(data[time_column], format="%H:%M", errors="coerce")
        data[time_column] = parsed.dt.hour * 60 + parsed.dt.minute

        if verbose:
            print(f"\nConverted '{time_column}' to minutes-since-midnight.")

    return data


# DATA PREPROCESSING

def preprocess_data(data, target_column=TARGET_COLUMN):
    """
    Clean the dataset (row-level cleaning only):
        1. remove repeated records of the same transaction (same
           Transaction_ID). This matters: if a copy lands in the training
           set and its twin in the test set, the model is "tested" on
           something it has already seen and the scores look better than
           they are.
        2. drop the ID columns (row/user labels, not features)
        3. drop any remaining identical rows
        4. drop rows with a missing target label
        5. convert the time column to a number (if it is text)
        6. drop text columns with too many unique values

    Missing FEATURE values are not filled here. That happens inside the
    model pipeline so the medians come from training data only.
    """
    data = data.copy()

    if TRANSACTION_ID_COLUMN in data.columns:
        rows_before = len(data)
        data = data.drop_duplicates(subset=[TRANSACTION_ID_COLUMN], keep="first")
        print(f"\nRepeated Transaction_IDs removed: {rows_before - len(data)}")

    data = data.drop(columns=[c for c in ID_COLUMNS if c in data.columns])

    rows_before = len(data)
    data = data.drop_duplicates()
    print(f"Other identical rows removed: {rows_before - len(data)}")

    # A missing label is unknown, not "not fraud". Never fill it in.
    if target_column in data.columns:
        rows_before = len(data)
        data = data.dropna(subset=[target_column])
        print(f"Rows removed for missing '{target_column}': {rows_before - len(data)}")

    data = convert_time_column(data)

    for column in data.select_dtypes(exclude=np.number).columns:
        if column == target_column:
            continue
        if data[column].nunique() > MAX_CATEGORIES:
            data = data.drop(columns=[column])
            print(f"Dropped '{column}' (more than {MAX_CATEGORIES} unique text values).")

    print("\nPreprocessing completed.")

    return data


# PREPARE MACHINE LEARNING DATA

def prepare_ml_data(data, target_column=TARGET_COLUMN, test_size=0.2):
    """
    Split the dataset into train/test features and targets.

    Features stay as a DataFrame (with column names). Imputing and
    encoding happen later, inside the pipeline.

    Args:
        data: pandas DataFrame (already preprocessed)
        target_column: column to predict (defaults to "Fraud")
        test_size: fraction of data held out for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.empty:
        raise ValueError("No feature columns were found.")

    # Fraud is a rare class here, so stratify to keep the same ratio in both
    # splits. Only stratify when the target looks like a class label.
    stratify_target = y if 1 < y.nunique() <= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_target
    )

    print("\nML data prepared.")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature names: {X.columns.tolist()}")

    return X_train, X_test, y_train, y_test


# FEATURE SIGNAL CHECK

def check_feature_signal(data, target_column=TARGET_COLUMN, alpha=0.05):
    """
    Test every feature on its own against the target, BEFORE modelling, to
    see whether any of them carries information about fraud.

        numeric features     -> single-feature ROC-AUC (0.5 = no signal) and
                                a Mann-Whitney U test
        categorical features -> chi-square test of independence on the
                                category vs fraud table (missing values are
                                treated as their own category)

    Because many features are tested at once, some will look "significant"
    by chance. The Bonferroni-adjusted threshold (alpha / number of tests)
    is shown so a lucky result isn't over-read.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

    y = data[target_column]
    features = [c for c in data.columns if c != target_column]
    results = []

    for column in features:
        if pd.api.types.is_numeric_dtype(data[column]):
            present = data[column].notna()
            values, labels = data.loc[present, column], y[present]
            auc = roc_auc_score(labels, values)
            p_value = stats.mannwhitneyu(
                values[labels == 1], values[labels == 0], alternative="two-sided"
            ).pvalue
            results.append((column, "numeric", f"AUC {auc:.3f}", p_value))
        else:
            table = pd.crosstab(data[column].fillna("Missing"), y)
            p_value = stats.chi2_contingency(table)[1]
            rates = (table[1] / table.sum(axis=1) * 100)
            results.append((column, "categorical",
                            f"fraud rate {rates.min():.2f}%-{rates.max():.2f}%", p_value))

    adjusted_alpha = alpha / len(results)

    print("\n========== DOES ANY FEATURE PREDICT FRAUD? ==========")
    print(f"Overall fraud rate: {y.mean() * 100:.2f}%")
    print(f"{'Feature':34} {'Type':12} {'Effect':26} {'p-value':>8}")
    for column, kind, effect, p_value in results:
        print(f"{column:34} {kind:12} {effect:26} {p_value:>8.3f}")

    print(f"\nSignificance threshold: {alpha} (adjusted for {len(results)} tests: {adjusted_alpha:.4f})")
    hits = [r[0] for r in results if r[3] < adjusted_alpha]
    maybe = [r[0] for r in results if adjusted_alpha <= r[3] < alpha]
    print("Clear signal       :", hits if hits else "none")
    print("Borderline (could be chance):", maybe if maybe else "none")

    return results
