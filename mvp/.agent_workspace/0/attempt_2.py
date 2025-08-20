import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import gc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error


# Define the RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature Engineering (if any, none specified so far, but good practice to have a placeholder)
# For this problem, the provided features seem sufficient.

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

# Scaling features - LightGBM is less sensitive to feature scaling, but it's good practice
# and can sometimes help with convergence or if other models were to be used.
# Using StandardScaler as it's common.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier indexing with iloc
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# Model Training with Cross-Validation
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Initialize arrays for out-of-fold predictions and test predictions
# oof_preds will store predictions for the training data, useful for local validation
# test_preds will accumulate predictions for the test data across folds
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

# Dictionary to store RMSLE scores for each target, per fold
cv_rmsle_scores_per_fold = {target: [] for target in TARGETS}

# LightGBM parameters
# Added 'eval_metric': 'rmsle' to address the ValueError
params = {
    "objective": "regression_l1",  # MAE objective often robust
    "metric": "rmsle",  # This is for reporting during training, but early_stopping needs explicit eval_metric
    "eval_metric": "rmsle",  # Explicitly set for early stopping
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,  # Base seed, fold-specific seeds will be added
    "boosting_type": "gbdt",
}

for i, target in enumerate(TARGETS):
    print(f"Training model for: {target}")
    y_target = y[target]

    # Log transform the target variable
    # Add a small epsilon to avoid log(0) if there are zeros, though unlikely for energy values
    y_target_log = np.log1p(y_target)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_scaled, y_target_log)):
        print(f"  Fold {fold_+1}/{NFOLDS}")
        trn_data = lgb.Dataset(X_scaled.iloc[trn_idx], label=y_target_log.iloc[trn_idx])
        val_data = lgb.Dataset(X_scaled.iloc[val_idx], label=y_target_log.iloc[val_idx])

        # Update seed for each fold for better randomness
        current_params = params.copy()
        current_params["seed"] = params["seed"] + fold_

        model = lgb.train(
            current_params,
            trn_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ],  # verbose=False is correct
        )

        # Predict on the validation set (in log-transformed space)
        val_preds_log = model.predict(X_scaled.iloc[val_idx])
        # Store these predictions in the overall OOF predictions array.
        oof_preds[val_idx, i] = val_preds_log

        # Predict on the test set (in log-transformed space)
        test_preds_fold = model.predict(X_test_scaled)
        # Accumulate test predictions. We'll average them later by dividing by NFOLDS.
        test_preds[:, i] += test_preds_fold / folds.n_splits

        # --- Calculate RMSLE for the current fold's validation set ---
        # Inverse transform predictions for this fold's validation set
        val_preds_inv = np.expm1(val_preds_log)
        # Clip predictions to be non-negative
        val_preds_inv = np.clip(val_preds_inv, 0, None)

        # Calculate RMSLE for this fold's validation set
        score = rmsle(y_target.iloc[val_idx], val_preds_inv)
        cv_rmsle_scores_per_fold[target].append(score)
        # print(f"    Fold {fold_+1} RMSLE for {target}: {score:.4f}") # Optional: print per-fold score

        # Clean up memory
        del model, trn_data, val_data
        gc.collect()

    # --- Final OOF calculation and reporting for the target ---
    # Calculate the mean RMSLE for this target across all folds
    mean_oof_rmsle_target = np.mean(cv_rmsle_scores_per_fold[target])
    print(
        f"  Mean OOF RMSLE for {target} across {NFOLDS} folds: {mean_oof_rmsle_target:.4f}"
    )


# --- Final Aggregation and Output ---

# Calculate the overall mean RMSLE across all target variables
mean_cv_rmsle_per_target = {
    target: np.mean(scores) for target, scores in cv_rmsle_scores_per_fold.items()
}
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