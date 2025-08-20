import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import os


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Prepare data
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
FEATURES = [col for col in train_df.columns if col not in ["id"] + TARGETS]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]

# Log transform targets
y_log = np.log1p(y)

# Model Training and Prediction
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

for i, target in enumerate(TARGETS):
    print(f"Training model for {target}...")
    fold_rmsle_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log[target])):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log[target].iloc[train_idx], y_log[target].iloc[val_idx]

        lgb_params = {
            "objective": "regression_l1",  # MAE is often robust
            "metric": "rmsle",
            "n_estimators": 1000,  # Increased estimators, but no early stopping
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "boosting_type": "gbdt",
        }

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)  # No early stopping

        val_preds = model.predict(X_val)
        oof_preds[val_idx, i] = val_preds

        # Evaluate on validation set
        fold_rmsle = rmsle(np.expm1(y_val), np.maximum(0, np.expm1(val_preds)))
        fold_rmsle_scores.append(fold_rmsle)
        print(f"Fold {fold+1} RMSLE for {target}: {fold_rmsle}")

    print(f"Average RMSLE for {target} across folds: {np.mean(fold_rmsle_scores)}")

    # Predict on test set
    test_preds[:, i] = model.predict(X_test)

# Inverse transform and clip predictions
oof_preds_inv = np.maximum(0, np.expm1(oof_preds))
test_preds_inv = np.maximum(0, np.expm1(test_preds))

# Calculate overall OOF RMSLE
oof_rmsle_formation = rmsle(train_df[TARGETS[0]], oof_preds_inv[:, 0])
oof_rmsle_bandgap = rmsle(train_df[TARGETS[1]], oof_preds_inv[:, 1])
mean_oof_rmsle = (oof_rmsle_formation + oof_rmsle_bandgap) / 2

print("\n--- Overall OOF RMSLE ---")
print(f"Formation Energy RMSLE: {oof_rmsle_formation}")
print(f"Bandgap Energy RMSLE: {oof_rmsle_bandgap}")
print(f"Mean OOF RMSLE: {mean_oof_rmsle}")

# Create submission file
submission_df = pd.DataFrame(
    {
        "id": test_df["id"],
        TARGETS[0]: test_preds_inv[:, 0],
        TARGETS[1]: test_preds_inv[:, 1],
    }
)

# Ensure output directory exists
os.makedirs("./0", exist_ok=True)
submission_df.to_csv("./0/submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: oof_rmsle_formation,
        TARGETS[1]: oof_rmsle_bandgap,
        "mean": mean_oof_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM Regressor (5-fold CV, log1p transform)",
}

import json

with open("./0/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nSubmission file created: ./0/submission.csv")
print("Metrics file created: ./0/metrics.json")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")