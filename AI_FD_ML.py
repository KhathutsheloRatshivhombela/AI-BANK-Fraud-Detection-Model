"""
BA3.2 AI Solution
AI/FD Component 2 of 3: MACHINE LEARNING

File:
    AI_FD_ML.py

Purpose:
    The classical machine-learning layer:
        - preprocessing + Random Forest fraud classifier (one Pipeline)
        - evaluation: PR-AUC, ROC-AUC, confusion matrix, thresholds,
          comparison with random guessing, cross-validation
        - regression model (for numeric targets such as Transaction_Amount)
        - scoring a single transaction and the ALLOW / REVIEW / BLOCK decision
        - saving and loading the trained model

Needs AI_FD_DATA.py (configuration and data helpers).
Deep learning and time-series analysis are in AI_FD_ADVANCED.py.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

from AI_FD_DATA import (
    MODEL_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    convert_time_column,
    create_directories
)


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

    print("AI_FD_ML.py is ready.")
    print("\nRun AI_FD_TEST.py to train, evaluate and save the model.")

# Create a virtual environment first: python -m venv .venv
# Then install dependencies:
# python -m pip install -r requirements.txt
