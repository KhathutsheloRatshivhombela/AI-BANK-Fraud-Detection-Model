# AI-BANK-Fraud-Detection-Model

Fraudulent transactions are a growing challenge for fintech companies. This project builds a fraud-detection pipeline, analyses patterns of fraud in financial transactions and tests how well the available data supports fraud prevention.

## Data Source
The primary dataset used for this project was sourced from Kaggle:
* **Dataset Link:** https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv

Download `Fraud_Detection_Dataset.csv` and place it in the `DATA/` folder.

## Project Structure
```
AI_FD.py          # all functions: loading, cleaning, training, evaluation, prediction
AI_FD_TEST.py     # runs the whole pipeline from start to finish
AI_FD_CHATBOT.py  # text assistant that scores and explains transactions
AI_FD_TIMESERIES_SAMPLE.py  # fraud rate by hour of day (time-series sample)
DATA/             # put Fraud_Detection_Dataset.csv here
models/           # the trained model is saved here (created automatically)
requirements.txt
```

## How to Run
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python AI_FD_TEST.py            # runs the full pipeline and saves the model
python AI_FD_CHATBOT.py         # chat with the fraud assistant
python AI_FD_TIMESERIES_SAMPLE.py --plot   # hourly fraud analysis, saves hourly_fraud_rate.png

```

## What the Pipeline Does
1. **Load and explore** the dataset (51,000 rows, 12 columns).
2. **Clean**: remove 1,000 repeated `Transaction_ID`s, drop ID columns, drop rows with no label. Missing feature values are filled inside the model pipeline, using training data only, with a "was missing" flag kept as a feature.
3. **Check for signal**: test every feature against fraud on its own, before modelling.
4. **Train** a Random Forest (`class_weight="balanced"`) inside a scikit-learn Pipeline.
5. **Evaluate** with PR-AUC, ROC-AUC, a confusion matrix, precision/recall and a threshold trade-off table, then compare against random guessing.
6. **Cross-validate** (5 stratified folds) to check the scores are stable.
7. **Deep learning**: train a neural network classifier and an autoencoder (see below) on the same split, and compare them with the Random Forest and with random guessing.
8. **Save** the whole pipeline (preprocessing + model) to one file.

## Deep Learning
Two techniques are used, both built with scikit-learn's neural networks, so no extra libraries are needed:
* **Neural network classifier (supervised):** three hidden layers (64, 32, 16 units, ReLU activation). Numeric inputs are standardised, fraud cases are weighted up to offset the class imbalance, and training stops early when the score on a held-back 10% stops improving, to avoid over-fitting.
* **Autoencoder (anomaly detection):** a network (16-8-16 units) trained only on legitimate transactions to rebuild them through a narrow middle layer. Fraud is unusual, so it should be rebuilt badly, and the rebuild error is used as an anomaly score.

Both are scored with the same PR-AUC and ROC-AUC as the Random Forest, so the comparison is fair.

## Chatbot
`AI_FD_CHATBOT.py` is an assistant for fraud analysts. They type plain requests, and the bot scores the transaction with the trained model and explains the result.
* **Score a transaction** in one line (`score amount=950 type=online purchase hour=2 device=mobile payment=UPI`) or through guided questions (type `score`).
* **Explain a score** (`why`): the fraud score compared with the usual fraud rate, the decision, and which values moved the score most compared with a typical value.
* **Explain the decisions** (ALLOW / REVIEW / BLOCK). The cut-offs are set by analyst capacity: REVIEW covers the riskiest 5% of transactions and BLOCK the riskiest 0.5%.
* **Answer honestly about accuracy**, including that the model performs at chance level on this dataset.
* The fraud score always comes from the machine-learning model. The bot only does the talking, so its answers can be traced and checked.
* **Optional language layer:** if a local Ollama server running Phi-3 is available and `FD_USE_OLLAMA=1` is set, answers are reworded in friendlier language, with the numbers and decision kept unchanged. If Ollama is not running, the bot uses its own text.

## Key Findings
* **The data is highly imbalanced**: 4.92% of transactions are fraud. A model that always says "not fraud" scores about 95% accuracy, so accuracy is not used to judge the model.
* **No feature predicts fraud in this dataset.** Single-feature ROC-AUCs are all between 0.495 and 0.508 (0.5 means no signal). Fraud rates barely change across transaction types, locations, payment methods, or the number of previous fraudulent transactions (4.7% with none, 4.8% with four).
* **The model performs at chance level**: PR-AUC 0.051 against a random-guess baseline of 0.049, and ROC-AUC 0.496. Gradient boosting and logistic regression give the same result, so this is a property of the data, not of the model chosen.
* **Deep learning does not change this**: the neural network scores a PR-AUC of 0.047 (ROC-AUC 0.477) and the autoencoder 0.051 (ROC-AUC 0.494), all within noise of random guessing (0.049).
* **One weak lead**: transactions with a missing `Device_Used` show a higher fraud rate (6.1% vs 4.9%). After adjusting for testing nine features at once, this is borderline and could be chance.
* **Fraud does not cluster by user** and does not depend on transaction order.
* **Fraud does not depend on the hour of day**: the hourly fraud rate stays between 4.27% and 5.78% (overall 4.90%), inside the range random variation would produce (chi-square p = 0.701).

## Limitations and Next Steps
* The dataset appears to be synthetic, with fraud labels unrelated to the other columns. Any accuracy figure from this data should be read with that in mind.
* The review/block thresholds in `make_ai_decision()` are placeholders. With this data, no threshold gives a useful balance of precision and recall.
* A dataset with real fraud patterns (and a date column, for the ARIMA time-series functions) would be needed for a production-quality model.

## License
MIT. See `LICENSE`.
