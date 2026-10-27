"""
BA3.2 AI Solution
AI/FD Component

File:
    AI_FD.py

Purpose:
    Machine learning and time-series layer for the BA3.2 fraud detection
    solution. Trains a classifier on transaction data to flag likely
    fraud (target column: Fraud).

This module is independent from:
    - Ollama / Phi-3
    - faster-whisper
    - Text-to-Speech
    - GUI
These are separate modules in the project structure and can be integrated later.

What changed in this version:
    - Rows with a missing Fraud label are dropped (never filled in).
    - Transaction_ID is dropped BEFORE de-duplicating.
    - Imputing / encoding now lives inside a scikit-learn Pipeline, so it is
      fitted on training data only and is saved together with the model.
    - Categorical columns are one-hot encoded instead of silently ignored.
    - Predictions use fraud PROBABILITIES, not hard 0/1 labels.
    - Evaluation adds a confusion matrix, PR-AUC, ROC-AUC, a threshold
      trade-off table and cross-validation.
    - make_prediction takes named values, so column order can't go wrong.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

# statsmodels (ARIMA) is only needed for the time-series forecast, so it is
# imported inside forecast_time_series(). That way the fraud classifier runs
# even if statsmodels is not installed.


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


# PREPROCESSING PIPELINE

def build_preprocessor(X):
    """
    Build the feature transformer:
        numeric columns     -> fill missing with the median, and add a
                               "was missing" flag column
        non-numeric columns -> fill missing with the label "Missing", then
                               one-hot encode
    Keeping "missing" visible matters for fraud: a blank field can itself
    be a warning sign, and filling it with the most common value would
    hide that. Unknown categories seen at prediction time are ignored,
    not errors.
    """
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric_columns:
        transformers.append(
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric_columns)
        )

    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers)


def build_classifier_pipeline(X):
    """
    Preprocessing + Random Forest as ONE object. Saving this object saves
    everything needed to score a new transaction.

    class_weight="balanced" is used because fraud cases are a small
    minority of the total transactions. (Random Forests don't need feature
    scaling. If you switch to logistic regression or a neural net, add a
    StandardScaler step to the numeric transformer.)
    """
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])


# CLASSIFICATION MODEL

def train_classifier(X_train, y_train):
    """
    Train the fraud classifier pipeline on the training split.
    """
    model = build_classifier_pipeline(X_train)
    model.fit(X_train, y_train)

    print("\nClassification model trained.")

    return model


# CLASSIFICATION EVALUATION

def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the classifier. For fraud, judge it on precision and recall
    for class 1, PR-AUC and the confusion matrix. Accuracy alone can look
    great while missing most of the fraud.

    threshold: fraud probability at or above which a transaction is
    counted as fraud.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy * 100:.2f}%  (misleading on its own for fraud)")
    print(f"PR-AUC:   {pr_auc:.4f}  (main score for imbalanced data)")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("              Pred 0   Pred 1")
    print(f"Actual 0   {matrix[0][0]:8d} {matrix[0][1]:8d}   <- Pred 1 here = false alarms")
    print(f"Actual 1   {matrix[1][0]:8d} {matrix[1][1]:8d}   <- Pred 0 here = missed fraud")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "accuracy": accuracy,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix
    }


def show_threshold_tradeoffs(model, X_test, y_test,
                             thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """
    Show what happens to precision and recall at different fraud-score
    cut-offs. Lower threshold = catches more fraud but raises more false
    alarms. Use this to pick the review / block thresholds for
    make_ai_decision().
    """
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n========== THRESHOLD TRADE-OFFS ==========")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'Flagged':>8}")

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        print(f"{threshold:>10.1f} {precision:>10.2f} {recall:>8.2f} {int(predictions.sum()):>8d}")


def show_baseline_comparison(model, X_test, y_test):
    """
    Compare the model with a model that guesses at random. With rare fraud,
    a random guess scores a PR-AUC equal to the fraud rate, so the number
    only means something next to that baseline. If the model's PR-AUC is
    about the same as the fraud rate, it has found nothing to learn from.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    fraud_rate = float(np.mean(y_test))

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n========== MODEL VS RANDOM GUESSING ==========")
    print(f"{'':20} {'Model':>8} {'Random guess':>13}")
    print(f"{'PR-AUC':20} {pr_auc:>8.3f} {fraud_rate:>13.3f}")
    print(f"{'ROC-AUC':20} {roc_auc:>8.3f} {0.5:>13.3f}")

    if pr_auc < fraud_rate * 1.25 and abs(roc_auc - 0.5) < 0.05:
        print("\nThe model is no better than random guessing. The features "
              "do not separate fraud from legitimate transactions in this "
              "dataset. See check_feature_signal() for the evidence.")
    else:
        print(f"\nThe model beats random guessing: PR-AUC is "
              f"{pr_auc / fraud_rate:.1f}x the fraud rate.")


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


def cross_validate_classifier(data, target_column=TARGET_COLUMN, n_splits=5):
    """
    Stratified k-fold cross-validation. A single train/test split can be
    lucky or unlucky when there are few fraud cases; this shows how stable
    the scores are. The pipeline is refitted inside every fold, so nothing
    leaks between folds.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    pipeline = build_classifier_pipeline(X)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        pipeline, X, y,
        cv=folds,
        scoring={
            "pr_auc": "average_precision",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0)
        }
    )

    print(f"\n========== {n_splits}-FOLD CROSS-VALIDATION ==========")
    for name in ("pr_auc", "precision", "recall"):
        values = scores[f"test_{name}"]
        print(f"{name:>10}: {values.mean():.3f} +/- {values.std():.3f}")

    return scores


# DEEP LEARNING
#
# Two deep learning techniques, both built with scikit-learn's neural
# networks (no extra libraries needed):
#   1. A multi-layer neural network trained to classify transactions as
#      fraud / not fraud (supervised learning).
#   2. An autoencoder trained ONLY on legitimate transactions. It learns to
#      rebuild normal transactions through a narrow middle layer. Fraud is
#      unusual, so it should be rebuilt badly. The rebuild error becomes an
#      anomaly score (unsupervised anomaly detection).
# Both are judged with the same metrics as the Random Forest, so they can be
# compared fairly.

def build_neural_preprocessor(X):
    """
    Preprocessing for the neural networks. Same idea as build_preprocessor(),
    with two differences:
        - numeric columns are also STANDARDISED (centred and rescaled).
          Neural networks learn badly when one input is in the thousands
          (Transaction_Amount) and another is 0 or 1. The Random Forest
          doesn't care, so it didn't need this.
        - the output is always a plain dense array.
    """
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric_columns:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler())
            ]),
            numeric_columns
        ))

    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers, sparse_threshold=0)


def build_neural_network_pipeline(X, hidden_layers=(64, 32, 16)):
    """
    Preprocessing + a three-hidden-layer neural network as ONE object.
    Training stops early once the score on a held-back 10% of the training
    data stops improving, which guards against over-fitting.
    """
    return Pipeline([
        ("preprocess", build_neural_preprocessor(X)),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=200,
            random_state=RANDOM_STATE
        ))
    ])


def train_neural_network(X_train, y_train):
    """
    Train the neural network classifier. Fraud cases are given a higher
    weight so the rare class is not ignored (the neural-network equivalent
    of class_weight="balanced" in the Random Forest).
    """
    model = build_neural_network_pipeline(X_train)
    weights = compute_sample_weight("balanced", y_train)

    try:
        model.fit(X_train, y_train, model__sample_weight=weights)
        weighting = "fraud cases weighted up to offset the class imbalance"
    except (TypeError, ValueError):
        # Older scikit-learn versions cannot weight neural-network samples.
        model.fit(X_train, y_train)
        weighting = "no class weighting (not supported by this scikit-learn version)"

    network = model.named_steps["model"]
    print("\nNeural network trained.")
    print(f"Hidden layers: {network.hidden_layer_sizes}, stopped after {network.n_iter_} passes over the data.")
    print(f"Class imbalance: {weighting}.")

    return model


def train_autoencoder(X_train, y_train, hidden_layers=(16, 8, 16)):
    """
    Train an autoencoder on LEGITIMATE transactions only. The narrow middle
    layer (8 units) forces it to learn what a normal transaction looks like.

    Returns a dict holding the fitted preprocessing and the network.
    """
    legitimate = X_train[y_train == 0]

    preprocess = build_neural_preprocessor(legitimate)
    inputs = preprocess.fit_transform(legitimate)

    network = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        alpha=1e-3,
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=200,
        random_state=RANDOM_STATE
    )
    network.fit(inputs, inputs)  # the target is the input itself

    print("\nAutoencoder trained on legitimate transactions only.")
    print(f"Inputs: {inputs.shape[1]}, layers: {hidden_layers}, stopped after {network.n_iter_} passes over the data.")

    return {"preprocess": preprocess, "network": network}


def autoencoder_scores(autoencoder, X):
    """
    Anomaly score for each transaction: how badly the autoencoder rebuilds
    it (mean squared error). Higher = more unusual = more suspicious.
    """
    inputs = autoencoder["preprocess"].transform(X)
    rebuilt = autoencoder["network"].predict(inputs)
    return np.mean((inputs - rebuilt) ** 2, axis=1)


def compare_models(scores, y_test):
    """
    Put several models side by side, next to random guessing.

    scores: dict of {model name: array of fraud scores}, where a higher
    score means "more likely fraud" (probabilities or anomaly scores).
    """
    fraud_rate = float(np.mean(y_test))
    results = {}

    print("\n========== MODEL COMPARISON ==========")
    print(f"{'Model':30} {'PR-AUC':>8} {'ROC-AUC':>8}")

    for name, values in scores.items():
        pr_auc = average_precision_score(y_test, values)
        roc_auc = roc_auc_score(y_test, values)
        results[name] = {"pr_auc": pr_auc, "roc_auc": roc_auc}
        print(f"{name:30} {pr_auc:>8.3f} {roc_auc:>8.3f}")

    print(f"{'Random guessing':30} {fraud_rate:>8.3f} {0.5:>8.3f}")

    return results


# REGRESSION MODEL

def train_regressor(X_train, y_train):
    """
    Train a Random Forest regressor. Useful if predicting a numeric
    target instead of Fraud, e.g. Transaction_Amount.
    """
    model = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    print("\nRegression model trained.")

    return model


# REGRESSION EVALUATION

def evaluate_regressor(model, X_test, y_test):
    """
    Evaluate a regression model with MAE, MSE, RMSE and R².
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n========== REGRESSION EVALUATION ==========")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# PREDICTION

def make_prediction(model, input_data):
    """
    Score one transaction and return its FRAUD PROBABILITY (0.0 - 1.0).

    input_data is a dict of named values, so column order doesn't matter:

        make_prediction(model, {
            "Transaction_Amount": 950.0,
            "Transaction_Type": "Online Purchase",
            "Time_of_Transaction": 2,           # hour of day, 0-23
            "Device_Used": "Mobile",
            "Location": "Boston",
            "Previous_Fraudulent_Transactions": 1,
            "Account_Age": 40,
            "Number_of_Transactions_Last_24H": 9,
            "Payment_Method": "UPI",
        })

    Include every feature the model was trained on (not the ID columns).
    A blank/None value is fine: it is treated as missing.
    """
    row = pd.DataFrame([input_data])
    row = convert_time_column(row, verbose=False)

    return float(model.predict_proba(row)[0, 1])


# TIME-SERIES ANALYSIS

def prepare_time_series(data, date_column, value_column):
    """
    Prepare data for time-series analysis by indexing on a date column.

    Needs a real date column. Transaction_Time alone (HH:MM) is not enough.

    Example:
        Date        Fraud_Count
        2026-01-01  12
    """
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' does not exist.")

    if value_column not in data.columns:
        raise ValueError(f"Value column '{value_column}' does not exist.")

    ts_data = data.copy()
    ts_data[date_column] = pd.to_datetime(ts_data[date_column])
    ts_data = ts_data.sort_values(date_column)
    ts_data = ts_data.set_index(date_column)

    series = ts_data[value_column]

    return series


# TIME-SERIES STATISTICS

def analyze_time_series(series):
    """
    Print basic time-series statistics.
    """
    print("\n========== TIME-SERIES ANALYSIS ==========")
    print(f"Number of observations: {len(series)}")
    print(f"Mean: {series.mean():.2f}")
    print(f"Minimum: {series.min():.2f}")
    print(f"Maximum: {series.max():.2f}")
    print(f"Standard deviation: {series.std():.2f}")

    print("\nRecent observations:")
    print(series.tail())


# ARIMA TIME-SERIES FORECAST

def forecast_time_series(series, periods=7, order=(1, 1, 1)):
    """
    Forecast future values using ARIMA. Initial implementation - the
    final order should be tuned once we have real time-stamped data.
    """
    from statsmodels.tsa.arima.model import ARIMA  # imported here, see top of file

    print("\nTraining ARIMA model...")

    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=periods)

    print(f"\nForecast for next {periods} periods:")
    print(forecast)

    return forecast


# MODEL SAVING

def save_model(model, model_path=MODEL_FILE):
    """
    Save the trained pipeline (preprocessing + model) to disk.
    """
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


# MODEL LOADING

def load_model(model_path=MODEL_FILE):
    """
    Load a previously saved pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Saved model not found.")

    model = joblib.load(model_path)

    print("\nModel loaded successfully.")

    return model


# AI DECISION LAYER

def make_ai_decision(fraud_probability, review_threshold=0.5, block_threshold=0.9):
    """
    Convert a fraud probability into a business decision:

        probability >= block_threshold   -> BLOCK
        probability >= review_threshold  -> REVIEW
        otherwise                        -> ALLOW

    The default thresholds are placeholders. Set them using
    show_threshold_tradeoffs() and the business's actual tolerance for
    false alarms versus missed fraud.
    """
    if review_threshold > block_threshold:
        raise ValueError("review_threshold must not be higher than block_threshold.")

    if fraud_probability >= block_threshold:
        decision = "BLOCK"
    elif fraud_probability >= review_threshold:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    return {"fraud_probability": fraud_probability, "decision": decision}


# SYSTEM STATUS

def system_status():
    """
    Print a quick status check of each component.
    """
    print("\n========== SYSTEM STATUS ==========")
    print("Machine Learning      : READY")
    print("Model Evaluation      : READY")
    print("Time-Series Analysis  : READY")
    print("Prediction            : READY")
    print("AI Decision Layer     : READY")

    print("\nExternal components:")
    # Separate modules, not run from here yet
    print("faster-whisper        : SEPARATE MODULE")
    print("Ollama / Phi-3        : SEPARATE MODULE")
    print("Text-to-Speech        : SEPARATE MODULE")
    print("GUI                   : FUTURE MODULE")

    print("==========================================\n")


# MAIN

if __name__ == "__main__":

    create_directories()
    system_status()

    print("AI_FD.py is ready.")
    print("\nWaiting for the final industry, business problem and dataset.")

# Create a virtual environment first: python -m venv .venv
# Then install dependencies:
# pip install pandas scikit-learn statsmodels joblib
