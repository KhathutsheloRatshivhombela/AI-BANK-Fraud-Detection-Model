"""
BA3.2 AI Solution
AI/FD Component 3 of 3: ADVANCED TECHNIQUES

File:
    AI_FD_ADVANCED.py

Purpose:
    Deep learning and time-series analysis:
        - neural network fraud classifier (supervised)
        - autoencoder anomaly detector (learns what NORMAL looks like)
        - side-by-side comparison of models with random guessing
        - time-series preparation, statistics and ARIMA forecast
          (needs a real date column, which the current dataset does not have)

Needs AI_FD_DATA.py (configuration). It does not depend on AI_FD_ML.py.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

from AI_FD_DATA import RANDOM_STATE

# statsmodels (ARIMA) is only needed for the time-series forecast, so it is
# imported inside forecast_time_series(). That way everything else runs even
# if statsmodels is not installed.


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
