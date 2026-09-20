# Pulls in all the functions from AI_FD.py
from AI_FD import *

# Main script: loads the data, cleans it, trains a model, checks how well
# it performs, then saves it so it doesn't need retraining every run.


# STEP 1: Load data
create_directories()

data = load_dataset()

explore_dataset(data)  # optional, just to see what's in there

# STEP 2: Clean and prepare
# Removes repeated Transaction_IDs, drops the ID columns, and drops rows
# with no Fraudulent label. Missing feature values are filled inside the
# model pipeline (from training data only).
clean_data = preprocess_data(data)

# STEP 3: Split and train
# Column being predicted is "Fraudulent" (TARGET_COLUMN in AI_FD.py)
X_train, X_test, y_train, y_test = prepare_ml_data(
    clean_data,
    target_column=TARGET_COLUMN
)

model = train_classifier(X_train, y_train)

# STEP 4: Check results
# Look at PR-AUC, precision/recall for class 1 and the confusion matrix,
# not just accuracy.
results = evaluate_classifier(model, X_test, y_test)

# See how precision/recall change with the cut-off, to choose thresholds
show_threshold_tradeoffs(model, X_test, y_test)

# Optional but recommended: how stable are the scores across 5 splits?
cross_validate_classifier(clean_data, target_column=TARGET_COLUMN)

# STEP 5: Save it (one file: preprocessing + model together)
save_model(model)

# STEP 6: Example of scoring a single new transaction
# (uncomment and use values that match your dataset's columns)
# score = make_prediction(model, {
#     "Transaction_Amount": 950.0,
#     "Transaction_Type": "Online Purchase",
#     "Time_of_Transaction": 2,
#     "Device_Used": "Mobile",
#     "Location": "Boston",
#     "Previous_Fraudulent_Transactions": 1,
#     "Account_Age": 40,
#     "Number_of_Transactions_Last_24H": 9,
#     "Payment_Method": "UPI",
# })
# print(make_ai_decision(score, review_threshold=0.5, block_threshold=0.9))

print("\nAll done. Model is ready to use.")
