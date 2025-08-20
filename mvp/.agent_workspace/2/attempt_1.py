import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
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

# Feature and target columns
feature_cols = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X_train = train_df[feature_cols]
y_train = train_df[target_cols]
X_test = test_df[feature_cols]

# Identify categorical features
categorical_features = [
    "spacegroup"
]  # Assuming 'spacegroup' is the only categorical feature

# Convert categorical features to category type for LightGBM
for col in categorical_features:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Get indices for categorical features
categorical_features_indices = [
    X_train.columns.get_loc(col) for col in categorical_features
]

# LightGBM parameters
lgb_params = {
    "objective": "regression_l1",
    "metric": "rmsle",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "boosting_type": "gbdt",
}

# --- Retrain on full data and predict ---
final_models = {}
final_predictions = {}

print("Retraining models on full data...")

for target in target_cols:
    print(f"Training {target}...")
    y_train_target = np.log1p(y_train[target])

    model = lgb.LGBMRegressor(**lgb_params)
    # No early stopping needed for final training on full data
    model.fit(X_train, y_train_target, categorical_feature=categorical_features_indices)
    final_models[target] = model
    # Predict on test data and clip to be non-negative
    final_predictions[target] = np.clip(np.expm1(model.predict(X_test)), 0, None)

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"]})
submission_df["formation_energy_ev_natom"] = final_predictions[target_cols[0]]
submission_df["bandgap_energy_ev"] = final_predictions[target_cols[1]]
submission_df.to_csv("submission.csv", index=False)

# --- Calculate CV RMSLE on full training data for metrics.json ---
print("\nCalculating CV RMSLE on full training data...")
cv_rmsle_full_train = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for target in target_cols:
    print(f"Calculating CV for {target}...")
    y_train_target = np.log1p(y_train[target])
    fold_rmsles_full_train = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_target)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = (
            y_train_target.iloc[train_idx],
            y_train_target.iloc[val_idx],
        )

        # Re-initialize model for each fold
        model = lgb.LGBMRegressor(**lgb_params)

        # Fit with early stopping, ensuring eval_set and eval_metric are provided
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="rmsle",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)],
            categorical_feature=categorical_features_indices,  # Use indices
        )

        val_preds = model.predict(X_val_fold)
        # Clip predictions before calculating RMSLE
        val_preds_clipped = np.clip(np.expm1(val_preds), 0, None)
        y_val_fold_original = np.expm1(
            y_val_fold
        )  # Convert back to original scale for RMSLE calculation

        fold_rmsle = rmsle(y_val_fold_original, val_preds_clipped)
        fold_rmsles_full_train.append(fold_rmsle)
        # print(f"  Fold {fold+1} RMSLE: {fold_rmsle:.6f}") # Optional: print fold RMSLE

    cv_rmsle_full_train[target] = np.mean(fold_rmsles_full_train)

mean_cv_rmsle = np.mean(list(cv_rmsle_full_train.values()))

# Save metrics
metrics = {
    "cv_rmsle": {
        "formation_energy_ev_natom": cv_rmsle_full_train["formation_energy_ev_natom"],
        "bandgap_energy_ev": cv_rmsle_full_train["bandgap_energy_ev"],
        "mean": mean_cv_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (retrained on full data)",
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print results
print("\nDataset Shapes:")
print(f"Train data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print("\nCV RMSLE Scores (on full training data):")
print(f"  Formation Energy: {cv_rmsle_full_train['formation_energy_ev_natom']:.6f}")
print(f"  Bandgap Energy: {cv_rmsle_full_train['bandgap_energy_ev']:.6f}")
print(f"  Mean CV RMSLE: {mean_cv_rmsle:.6f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error


# Define the RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error between true and predicted values.
    Assumes y_true and y_pred are on the original scale (not log-transformed).
    """
    # Ensure predictions are non-negative before applying log1p
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature and target columns
# Exclude target columns and 'id' from features
feature_cols = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X_train = train_df[feature_cols]
y_train = train_df[target_cols]
X_test = test_df[feature_cols]

# Identify categorical features
categorical_features = [
    "spacegroup"
]  # Assuming 'spacegroup' is the only categorical feature

# Convert categorical features to category type for LightGBM
# This is a good practice for LightGBM to handle categorical features efficiently
for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
    if col in X_test.columns:
        X_test[col] = X_test[col].astype("category")

# Get indices for categorical features, as LightGBM's fit method can accept indices
# This is often more robust than passing column names directly, especially with callbacks.
categorical_features_indices = []
for col in categorical_features:
    if col in X_train.columns:
        categorical_features_indices.append(X_train.columns.get_loc(col))

# LightGBM parameters - these are generally good starting points
lgb_params = {
    "objective": "regression_l1",  # MAE objective, often robust
    "metric": "rmsle",  # Metric for evaluation during training (though we use custom RMSLE for CV)
    "n_estimators": 1000,  # A large number, relying on early stopping
    "learning_rate": 0.05,
    "feature_fraction": 0.8,  # Fraction of features to consider per iteration
    "bagging_fraction": 0.8,  # Fraction of data to sample per iteration
    "bagging_freq": 1,
    "lambda_l1": 0.1,  # L1 regularization
    "lambda_l2": 0.1,  # L2 regularization
    "num_leaves": 31,  # Default, good balance
    "verbose": -1,  # Suppress verbose output
    "n_jobs": -1,  # Use all available cores
    "seed": 42,  # For reproducibility
    "boosting_type": "gbdt",
}

# --- Retrain on full data and predict ---
final_models = {}
final_predictions = {}

print("Retraining models on full data...")

for target in target_cols:
    print(f"Training {target}...")
    # Log transform the target variable for training, as RMSLE is used
    y_train_target_log = np.log1p(y_train[target])

    model = lgb.LGBMRegressor(**lgb_params)
    # Fit on the entire training data. No early stopping needed here as we are training on full data.
    # Pass categorical feature indices.
    model.fit(
        X_train, y_train_target_log, categorical_feature=categorical_features_indices
    )
    final_models[target] = model

    # Predict on test data
    predictions_log = model.predict(X_test)
    # Convert predictions back to original scale and clip to be non-negative
    final_predictions[target] = np.clip(np.expm1(predictions_log), 0, None)

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"]})
submission_df["formation_energy_ev_natom"] = final_predictions[target_cols[0]]
submission_df["bandgap_energy_ev"] = final_predictions[target_cols[1]]
submission_df.to_csv("submission.csv", index=False)

# --- Calculate CV RMSLE on full training data for metrics.json ---
print("\nCalculating CV RMSLE on full training data...")
cv_rmsle_scores = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for target in target_cols:
    print(f"Calculating CV for {target}...")
    y_train_target_log = np.log1p(y_train[target])
    fold_rmsles = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_target_log)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold_log, y_val_fold_log = (
            y_train_target_log.iloc[train_idx],
            y_train_target_log.iloc[val_idx],
        )

        # Re-initialize model for each fold to ensure a fresh start
        model = lgb.LGBMRegressor(**lgb_params)

        # Fit with early stopping.
        # eval_set and eval_metric are crucial for early_stopping to work.
        # Pass categorical feature indices.
        model.fit(
            X_train_fold,
            y_train_fold_log,
            eval_set=[(X_val_fold, y_val_fold_log)],
            eval_metric="rmsle",  # LightGBM's internal RMSLE metric
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=-1)
            ],  # Stop if no improvement for 50 rounds
            categorical_feature=categorical_features_indices,
        )

        # Predict on the validation set
        val_preds_log = model.predict(X_val_fold)

        # Convert predictions and true values back to original scale for RMSLE calculation
        # The rmsle function expects original scale values.
        val_preds_original_scale = np.expm1(val_preds_log)
        y_val_fold_original_scale = np.expm1(y_val_fold_log)

        # Calculate RMSLE for the fold
        fold_rmsle = rmsle(y_val_fold_original_scale, val_preds_original_scale)
        fold_rmsles.append(fold_rmsle)
        # print(f"  Fold {fold+1} RMSLE: {fold_rmsle:.6f}") # Optional: print fold RMSLE for debugging

    cv_rmsle_scores[target] = np.mean(fold_rmsles)

mean_cv_rmsle = np.mean(list(cv_rmsle_scores.values()))

# Save metrics to metrics.json
metrics = {
    "cv_rmsle": {
        "formation_energy_ev_natom": cv_rmsle_scores["formation_energy_ev_natom"],
        "bandgap_energy_ev": cv_rmsle_scores["bandgap_energy_ev"],
        "mean": mean_cv_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (retrained on full data)",
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print dataset shapes and CV RMSLE summary
print("\nDataset Shapes:")
print(f"Train data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print("\nCV RMSLE Scores (on full training data):")
print(f"  Formation Energy: {cv_rmsle_scores['formation_energy_ev_natom']:.6f}")
print(f"  Bandgap Energy: {cv_rmsle_scores['bandgap_energy_ev']:.6f}")
print(f"  Mean CV RMSLE: {mean_cv_rmsle:.6f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error


# Define the RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error between true and predicted values.
    Assumes y_true and y_pred are on the original scale (not log-transformed).
    """
    # Ensure predictions are non-negative before applying log1p
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature and target columns
# Exclude target columns and 'id' from features
feature_cols = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X_train = train_df[feature_cols]
y_train = train_df[target_cols]
X_test = test_df[feature_cols]

# Identify categorical features
categorical_features = [
    "spacegroup"
]  # Assuming 'spacegroup' is the only categorical feature

# Convert categorical features to category type for LightGBM
# This is a good practice for LightGBM to handle categorical features efficiently
for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
    if col in X_test.columns:
        X_test[col] = X_test[col].astype("category")

# Get indices for categorical features, as LightGBM's fit method can accept indices
# This is often more robust than passing column names directly, especially with callbacks.
categorical_features_indices = []
for col in categorical_features:
    if col in X_train.columns:
        categorical_features_indices.append(X_train.columns.get_loc(col))

# LightGBM parameters - these are generally good starting points
lgb_params = {
    "objective": "regression_l1",  # MAE objective, often robust
    "metric": "rmsle",  # Metric for evaluation during training (though we use custom RMSLE for CV)
    "n_estimators": 1000,  # A large number, relying on early stopping
    "learning_rate": 0.05,
    "feature_fraction": 0.8,  # Fraction of features to consider per iteration
    "bagging_fraction": 0.8,  # Fraction of data to sample per iteration
    "bagging_freq": 1,
    "lambda_l1": 0.1,  # L1 regularization
    "lambda_l2": 0.1,  # L2 regularization
    "num_leaves": 31,  # Default, good balance
    "verbose": -1,  # Suppress verbose output
    "n_jobs": -1,  # Use all available cores
    "seed": 42,  # For reproducibility
    "boosting_type": "gbdt",
}

# --- Retrain on full data and predict ---
final_models = {}
final_predictions = {}

print("Retraining models on full data...")

for target in target_cols:
    print(f"Training {target}...")
    # Log transform the target variable for training, as RMSLE is used
    y_train_target_log = np.log1p(y_train[target])

    model = lgb.LGBMRegressor(**lgb_params)
    # Fit on the entire training data. No early stopping needed here as we are training on full data.
    # Pass categorical feature indices.
    model.fit(
        X_train, y_train_target_log, categorical_feature=categorical_features_indices
    )
    final_models[target] = model

    # Predict on test data
    predictions_log = model.predict(X_test)
    # Convert predictions back to original scale and clip to be non-negative
    final_predictions[target] = np.clip(np.expm1(predictions_log), 0, None)

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"]})
submission_df["formation_energy_ev_natom"] = final_predictions[target_cols[0]]
submission_df["bandgap_energy_ev"] = final_predictions[target_cols[1]]
submission_df.to_csv("submission.csv", index=False)

# --- Calculate CV RMSLE on full training data for metrics.json ---
print("\nCalculating CV RMSLE on full training data...")
cv_rmsle_scores = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for target in target_cols:
    print(f"Calculating CV for {target}...")
    y_train_target_log = np.log1p(y_train[target])
    fold_rmsles = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_target_log)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold_log, y_val_fold_log = (
            y_train_target_log.iloc[train_idx],
            y_train_target_log.iloc[val_idx],
        )

        # Re-initialize model for each fold to ensure a fresh start
        model = lgb.LGBMRegressor(**lgb_params)

        # Fit with early stopping.
        # eval_set and eval_metric are crucial for early_stopping to work.
        # Pass categorical feature indices.
        model.fit(
            X_train_fold,
            y_train_fold_log,
            eval_set=[(X_val_fold, y_val_fold_log)],
            eval_metric="rmsle",  # LightGBM's internal RMSLE metric
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=-1)
            ],  # Stop if no improvement for 50 rounds
            categorical_feature=categorical_features_indices,
        )

        # Predict on the validation set
        val_preds_log = model.predict(X_val_fold)

        # Convert predictions and true values back to original scale for RMSLE calculation
        # The rmsle function expects original scale values.
        val_preds_original_scale = np.expm1(val_preds_log)
        y_val_fold_original_scale = np.expm1(y_val_fold_log)

        # Calculate RMSLE for the fold
        fold_rmsle = rmsle(y_val_fold_original_scale, val_preds_original_scale)
        fold_rmsles.append(fold_rmsle)
        # print(f"  Fold {fold+1} RMSLE: {fold_rmsle:.6f}") # Optional: print fold RMSLE for debugging

    cv_rmsle_scores[target] = np.mean(fold_rmsles)

mean_cv_rmsle = np.mean(list(cv_rmsle_scores.values()))

# Save metrics to metrics.json
metrics = {
    "cv_rmsle": {
        "formation_energy_ev_natom": cv_rmsle_scores["formation_energy_ev_natom"],
        "bandgap_energy_ev": cv_rmsle_scores["bandgap_energy_ev"],
        "mean": mean_cv_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (retrained on full data)",
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print dataset shapes and CV RMSLE summary
print("\nDataset Shapes:")
print(f"Train data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print("\nCV RMSLE Scores (on full training data):")
print(f"  Formation Energy: {cv_rmsle_scores['formation_energy_ev_natom']:.6f}")
print(f"  Bandgap Energy: {cv_rmsle_scores['bandgap_energy_ev']:.6f}")
print(f"  Mean CV RMSLE: {mean_cv_rmsle:.6f}")