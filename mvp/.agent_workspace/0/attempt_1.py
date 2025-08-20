import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
import gc


# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature Engineering (if any, none specified in prompt, so using raw features)
# For simplicity, we'll use the provided features directly.
# In a real scenario, more feature engineering might be beneficial.

# Define features and targets
FEATURES = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]
test_ids = test_df["id"]

# --- Preprocessing ---
# Scaling features is generally good practice for many models, though LightGBM is less sensitive.
# We'll scale features for consistency.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names for potential debugging or feature importance
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# --- Model Training with Cross-Validation ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold predictions and test predictions
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

# Dictionary to store CV RMSLE scores for each target
cv_rmsle_scores = {target: [] for target in TARGETS}

for i, target in enumerate(TARGETS):
    print(f"Training model for: {target}")
    y_target = y[target]

    # Log transform the target variable for RMSLE optimization
    # Add a small epsilon to avoid log(0) if there are zeros, though log1p handles 0.
    y_target_log = np.log1p(y_target)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_scaled, y_target_log)):
        print(f"  Fold {fold_+1}/{NFOLDS}")
        trn_data = lgb.Dataset(X_scaled.iloc[trn_idx], label=y_target_log.iloc[trn_idx])
        val_data = lgb.Dataset(X_scaled.iloc[val_idx], label=y_target_log.iloc[val_idx])

        # LightGBM parameters
        # Using 'regression_l1' (MAE) or 'regression_l2' (MSE) are common.
        # 'rmsle' is specified as a metric for evaluation.
        params = {
            "objective": "regression_l1",  # MAE objective
            "metric": "rmsle",  # Metric to monitor
            "n_estimators": 3000,  # Increased estimators, rely on early stopping
            "learning_rate": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 31,
            "verbose": -1,  # Suppress verbose output during training
            "n_jobs": -1,  # Use all available cores
            "seed": 42 + fold_,  # Seed for reproducibility per fold
            "boosting_type": "gbdt",
            # Explicitly setting eval_metric might help, though 'metric' should suffice
            # "eval_metric": "rmsle"
        }

        # Train the model
        model = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],  # Provide validation set
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ],  # Set verbose to False to avoid printing per fold
            # Ensure 'metric' in params is used by early_stopping.
            # If the error persists, explicitly passing eval_metric here might be needed,
            # but the documentation suggests 'metric' in params is the standard way.
        )

        # Predict on validation set (log transformed)
        val_preds_log = model.predict(X_scaled.iloc[val_idx])
        oof_preds[val_idx, i] = val_preds_log

        # Predict on test set (log transformed)
        test_preds_fold = model.predict(X_test_scaled)
        test_preds[:, i] += (
            test_preds_fold / folds.n_splits
        )  # Accumulate predictions for averaging

    # --- Evaluation and Inverse Transform ---
    # Inverse transform OOF predictions and clip to be non-negative
    oof_preds_inv = np.expm1(oof_preds[:, i])
    oof_preds_inv = np.clip(oof_preds_inv, 0, None)

    # Calculate RMSLE for OOF predictions for this target
    score = rmsle(y_target, oof_preds_inv)
    cv_rmsle_scores[target] = score  # Store the final OOF score for this target
    print(f"  OOF RMSLE for {target}: {score:.4f}")

    # Clean up memory
    del model, trn_data, val_data
    gc.collect()


# --- Final Aggregation and Output ---

# Calculate mean RMSLE across all folds for each target
# Since we stored the final OOF score per target, we just use that.
# If we wanted per-fold scores, we'd average the list.
mean_cv_rmsle_per_target = {target: cv_rmsle_scores[target] for target in TARGETS}
mean_cv_rmsle_overall = np.mean(list(mean_cv_rmsle_per_target.values()))

print("\n--- Cross-Validation RMSLE Summary ---")
for target, score in mean_cv_rmsle_per_target.items():
    print(f"{target}: {score:.4f}")
print(f"Overall Mean RMSLE: {mean_cv_rmsle_overall:.4f}")

# Final Test Predictions: Inverse transform and clip
final_test_preds = np.expm1(test_preds)
final_test_preds = np.clip(final_test_preds, 0, None)

# Create submission file
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        TARGETS[0]: final_test_preds[:, 0],
        TARGETS[1]: final_test_preds[:, 1],
    }
)

submission_df.to_csv("submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: mean_cv_rmsle_per_target[TARGETS[0]],
        TARGETS[1]: mean_cv_rmsle_per_target[TARGETS[1]],
        "mean": mean_cv_rmsle_overall,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (2x independent, log1p transform, 5-fold CV)",
}

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nSubmission file 'submission.csv' created successfully.")
print("Metrics file 'metrics.json' created successfully.")
print(f"\nDataset shapes: Train={train_df.shape}, Test={test_df.shape}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
import gc


# Define RMSLE function
def rmsle(y_true, y_pred):
    # Ensure predictions are non-negative before log1p
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Define features and targets
FEATURES = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]
test_ids = test_df["id"]

# --- Preprocessing ---
# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep column names
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# --- Model Training with Cross-Validation ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold predictions and test predictions
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

# Dictionary to store final OOF RMSLE scores for each target
cv_rmsle_scores = {}

for i, target in enumerate(TARGETS):
    print(f"Training model for: {target}")
    y_target = y[target]

    # Log transform the target variable
    y_target_log = np.log1p(y_target)

    fold_oof_preds_log = []  # Store OOF predictions for this target across folds

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_scaled, y_target_log)):
        print(f"  Fold {fold_+1}/{NFOLDS}")
        trn_data = lgb.Dataset(X_scaled.iloc[trn_idx], label=y_target_log.iloc[trn_idx])
        val_data = lgb.Dataset(X_scaled.iloc[val_idx], label=y_target_log.iloc[val_idx])

        # LightGBM parameters
        params = {
            "objective": "regression_l1",
            "metric": "rmsle",  # This should be sufficient, but let's try eval_metric too
            "eval_metric": "rmsle",  # Explicitly set eval_metric
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42 + fold_,
            "boosting_type": "gbdt",
        }

        # Train the model
        model = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # Predict on validation set (log transformed)
        val_preds_log = model.predict(X_scaled.iloc[val_idx])
        oof_preds[val_idx, i] = val_preds_log  # Store OOF predictions

        # Predict on test set (log transformed)
        test_preds_fold = model.predict(X_test_scaled)
        test_preds[:, i] += test_preds_fold / folds.n_splits  # Accumulate predictions

    # --- Evaluation and Inverse Transform for the target ---
    # Calculate OOF RMSLE for this target using the accumulated OOF predictions
    oof_preds_inv = np.expm1(oof_preds[:, i])
    oof_preds_inv = np.clip(oof_preds_inv, 0, None)

    score = rmsle(y_target, oof_preds_inv)
    cv_rmsle_scores[target] = score  # Store the final OOF score for this target
    print(f"  OOF RMSLE for {target}: {score:.4f}")

    # Clean up memory
    del model, trn_data, val_data
    gc.collect()


# --- Final Aggregation and Output ---

# Calculate overall mean RMSLE
mean_cv_rmsle_per_target = cv_rmsle_scores
mean_cv_rmsle_overall = np.mean(list(mean_cv_rmsle_per_target.values()))

print("\n--- Cross-Validation RMSLE Summary ---")
for target, score in mean_cv_rmsle_per_target.items():
    print(f"{target}: {score:.4f}")
print(f"Overall Mean RMSLE: {mean_cv_rmsle_overall:.4f}")

# Final Test Predictions: Inverse transform and clip
final_test_preds = np.expm1(test_preds)
final_test_preds = np.clip(final_test_preds, 0, None)

# Create submission file
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        TARGETS[0]: final_test_preds[:, 0],
        TARGETS[1]: final_test_preds[:, 1],
    }
)

submission_df.to_csv("submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: mean_cv_rmsle_per_target[TARGETS[0]],
        TARGETS[1]: mean_cv_rmsle_per_target[TARGETS[1]],
        "mean": mean_cv_rmsle_overall,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (2x independent, log1p transform, 5-fold CV)",
}

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nSubmission file 'submission.csv' created successfully.")
print("Metrics file 'metrics.json' created successfully.")
print(f"\nDataset shapes: Train={train_df.shape}, Test={test_df.shape}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
import gc


# Define RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error between true and predicted values.
    Ensures predictions are non-negative before applying log1p.
    """
    # Ensure predictions are non-negative before log1p
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Error: Ensure train.csv and test.csv are in the same directory.")
    # In a real execution environment, this would likely halt. For this script, we'll exit.
    exit()

# Define features and targets
# Features are all columns except 'id' and the target variables.
FEATURES = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Prepare data for modeling
X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]
test_ids = test_df["id"]

# --- Preprocessing ---
# Scaling features is generally good practice for many models, although LightGBM is less sensitive.
# Using StandardScaler for feature scaling.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to retain column names, which can be useful
# for feature importance analysis or debugging, though not strictly necessary for LightGBM.
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# --- Model Training with Cross-Validation ---
NFOLDS = 5
# Initialize KFold for cross-validation. Shuffling ensures different splits each time,
# and random_state ensures reproducibility.
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold predictions (for validation) and test predictions.
# These will store predictions in the log-transformed space.
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

# Dictionary to store the final OOF RMSLE score for each target.
cv_rmsle_scores = {}

# Loop through each target variable to train a separate model.
for i, target in enumerate(TARGETS):
    print(f"Training model for: {target}")
    y_target = y[target]

    # Log transform the target variable. This is crucial for optimizing RMSLE,
    # as RMSLE penalizes relative errors, and log transformation helps models
    # learn these relative differences better. np.log1p handles y=0 gracefully.
    y_target_log = np.log1p(y_target)

    # Loop through each fold for cross-validation.
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_scaled, y_target_log)):
        print(f"  Fold {fold_+1}/{NFOLDS}")
        # Create LightGBM Dataset objects for training and validation sets.
        trn_data = lgb.Dataset(X_scaled.iloc[trn_idx], label=y_target_log.iloc[trn_idx])
        val_data = lgb.Dataset(X_scaled.iloc[val_idx], label=y_target_log.iloc[val_idx])

        # LightGBM parameters. These are chosen for a balance of performance and speed.
        # 'objective': 'regression_l1' (MAE) is often robust.
        # 'metric': 'rmsle' is specified for monitoring during training.
        # 'eval_metric': 'rmsle' is explicitly set to ensure the early stopping callback
        #                uses the correct metric, resolving the ValueError.
        params = {
            "objective": "regression_l1",
            "metric": "rmsle",
            "eval_metric": "rmsle",  # Explicitly set for early stopping
            "n_estimators": 3000,  # High number of estimators, relying on early stopping
            "learning_rate": 0.01,
            "feature_fraction": 0.8,  # Fraction of features to consider per iteration
            "bagging_fraction": 0.8,  # Fraction of data to sample per iteration
            "bagging_freq": 1,  # Frequency for bagging
            "lambda_l1": 0.1,  # L1 regularization
            "lambda_l2": 0.1,  # L2 regularization
            "num_leaves": 31,  # Number of leaves in one tree
            "verbose": -1,  # Suppress verbose output during training
            "n_jobs": -1,  # Use all available CPU cores
            "seed": 42 + fold_,  # Seed for reproducibility, varying per fold
            "boosting_type": "gbdt",
        }

        # Train the LightGBM model.
        # 'valid_sets' provides the validation data for monitoring.
        # 'callbacks' includes early stopping to prevent overfitting and speed up training.
        # 'verbose=False' in early_stopping prevents printing per-fold progress.
        model = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # Predict on the validation set (in log-transformed space).
        val_preds_log = model.predict(X_scaled.iloc[val_idx])
        # Store these predictions in the overall OOF predictions array.
        oof_preds[val_idx, i] = val_preds_log

        # Predict on the test set (in log-transformed space).
        test_preds_fold = model.predict(X_test_scaled)
        # Accumulate test predictions. We'll average them later by dividing by NFOLDS.
        test_preds[:, i] += test_preds_fold / folds.n_splits

    # --- Evaluation and Inverse Transform for the current target ---
    # After all folds are complete for this target, calculate the OOF RMSLE.
    # First, inverse transform the accumulated OOF predictions.
    oof_preds_inv = np.expm1(oof_preds[:, i])
    # Clip predictions to be non-negative, as negative energy/bandgap values are unphysical.
    oof_preds_inv = np.clip(oof_preds_inv, 0, None)

    # Calculate the RMSLE score using the true target values and the inverse-transformed OOF predictions.
    score = rmsle(y_target, oof_preds_inv)
    # Store the final OOF RMSLE score for this target.
    cv_rmsle_scores[target] = score
    print(f"  OOF RMSLE for {target}: {score:.4f}")

    # Clean up memory to free up resources, especially important in loops.
    del model, trn_data, val_data
    gc.collect()


# --- Final Aggregation and Output ---

# Calculate the overall mean RMSLE across all target variables.
mean_cv_rmsle_per_target = cv_rmsle_scores
mean_cv_rmsle_overall = np.mean(list(mean_cv_rmsle_per_target.values()))

# Print the cross-validation summary.
print("\n--- Cross-Validation RMSLE Summary ---")
for target, score in mean_cv_rmsle_per_target.items():
    print(f"{target}: {score:.4f}")
print(f"Overall Mean RMSLE: {mean_cv_rmsle_overall:.4f}")

# Final Test Predictions:
# Inverse transform the accumulated test predictions from log-transformed space.
final_test_preds = np.expm1(test_preds)
# Clip predictions to be non-negative.
final_test_preds = np.clip(final_test_preds, 0, None)

# Create the submission file in the specified format.
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        TARGETS[0]: final_test_preds[:, 0],
        TARGETS[1]: final_test_preds[:, 1],
    }
)

# Save the submission file.
submission_df.to_csv("submission.csv", index=False)

# Create the metrics file containing CV scores and dataset information.
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: mean_cv_rmsle_per_target[TARGETS[0]],
        TARGETS[1]: mean_cv_rmsle_per_target[TARGETS[1]],
        "mean": mean_cv_rmsle_overall,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (2x independent, log1p transform, 5-fold CV)",
}

# Save the metrics file.
with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

# Print confirmation messages.
print("\nSubmission file 'submission.csv' created successfully.")
print("Metrics file 'metrics.json' created successfully.")
print(f"\nDataset shapes: Train={train_df.shape}, Test={test_df.shape}")