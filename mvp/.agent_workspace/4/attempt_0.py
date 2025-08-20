import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature and target columns
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
FEATURES = [col for col in train_df.columns if col not in ["id"] + TARGETS]

X_train = train_df[FEATURES]
y_train = train_df[TARGETS]
X_test = test_df[FEATURES]
test_ids = test_df["id"]

# Ensure all feature columns are numeric, coercing errors to NaN
for col in FEATURES:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Simple imputation for any NaNs that might have been created
# In a real scenario, more sophisticated imputation would be used.
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())


# --- Model Training and Evaluation ---

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train), len(TARGETS)))
test_preds = np.zeros((len(X_test), len(TARGETS)))

cv_rmsle_scores = []

for i, target in enumerate(TARGETS):
    print(f"Training and evaluating for target: {target}")
    y_train_target = y_train[target]

    # Apply log1p transformation
    y_train_transformed = np.log1p(y_train_target)

    fold_rmsle_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_transformed)):
        X_train_fold, y_train_fold = (
            X_train.iloc[train_idx],
            y_train_transformed.iloc[train_idx],
        )
        X_val_fold, y_val_fold = (
            X_train.iloc[val_idx],
            y_train_transformed.iloc[val_idx],
        )

        # Gradient Boosting Regressor
        gbr = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        gbr.fit(X_train_fold, y_train_fold)

        # Predict on validation set
        val_preds_transformed = gbr.predict(X_val_fold)

        # Inverse transform and clip
        val_preds = np.expm1(val_preds_transformed)
        val_preds = np.clip(val_preds, 0, None)

        # Calculate RMSLE for the fold
        fold_rmsle = np.sqrt(
            mean_squared_error(y_train_target.iloc[val_idx], val_preds)
        )
        fold_rmsle_scores.append(fold_rmsle)

        # Store OOF predictions
        oof_preds[val_idx, i] = val_preds

        # Predict on test set for this fold and accumulate
        test_preds_fold_transformed = gbr.predict(X_test)
        test_preds_fold = np.expm1(test_preds_fold_transformed)
        test_preds_fold = np.clip(test_preds_fold, 0, None)
        test_preds[:, i] += test_preds_fold / NFOLDS

    print(f"  Fold RMSLE scores: {fold_rmsle_scores}")
    cv_rmsle_scores.append(np.mean(fold_rmsle_scores))
    print(f"  Average CV RMSLE for {target}: {cv_rmsle_scores[-1]}")

# Calculate overall CV RMSLE
mean_cv_rmsle = np.mean(cv_rmsle_scores)
print(f"\nOverall CV RMSLE: {mean_cv_rmsle}")

# --- Create Submission File ---
submission_df = pd.DataFrame({"id": test_ids})
submission_df["formation_energy_ev_natom"] = test_preds[:, 0]
submission_df["bandgap_energy_ev"] = test_preds[:, 1]

# Ensure directory exists
output_dir = "./4"
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

# --- Create Metrics File ---
metrics = {
    "cv_rmsle": {
        TARGETS[0]: cv_rmsle_scores[0],
        TARGETS[1]: cv_rmsle_scores[1],
        "mean": mean_cv_rmsle,
    },
    "n_train": len(X_train),
    "n_test": len(X_test),
    "model": "GradientBoostingRegressor (5-fold CV)",
}

metrics_path = os.path.join(output_dir, "metrics.json")
import json

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics file saved to {metrics_path}")

print("\nDataset shapes:")
print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print("\nCV RMSLE Summary:")
print(f"  Formation Energy: {cv_rmsle_scores[0]:.4f}")
print(f"  Bandgap Energy: {cv_rmsle_scores[1]:.4f}")
print(f"  Mean CV RMSLE: {mean_cv_rmsle:.4f}")