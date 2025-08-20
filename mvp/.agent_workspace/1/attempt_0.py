import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor
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

# Cross-validation setup
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Model training
oof_preds = np.zeros((len(X), len(TARGETS)))
test_preds = np.zeros((len(X_test), len(TARGETS)))

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y_log)):
    X_train, y_train = X.iloc[train_idx], y_log.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y_log.iloc[valid_idx]

    # XGBoost Regressor for each target
    model_formation = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,  # Increased estimators, but no early stopping
        learning_rate=0.05,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    model_bandgap = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,  # Increased estimators, but no early stopping
        learning_rate=0.05,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )

    # Train models
    model_formation.fit(X_train, y_train[TARGETS[0]])
    model_bandgap.fit(X_train, y_train[TARGETS[1]])

    # Predict on validation and test sets
    oof_preds[valid_idx, 0] = model_formation.predict(X_valid)
    oof_preds[valid_idx, 1] = model_bandgap.predict(X_valid)

    test_preds[:, 0] += model_formation.predict(X_test) / NFOLDS
    test_preds[:, 1] += model_bandgap.predict(X_test) / NFOLDS

# Inverse transform predictions and clip
oof_preds = np.expm1(oof_preds)
test_preds = np.expm1(test_preds)

oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

# Calculate CV RMSLE
cv_rmsle_formation = rmsle(np.expm1(y_log[TARGETS[0]]), oof_preds[:, 0])
cv_rmsle_bandgap = rmsle(np.expm1(y_log[TARGETS[1]]), oof_preds[:, 1])
cv_rmsle_mean = (cv_rmsle_formation + cv_rmsle_bandgap) / 2

print(f"CV RMSLE - formation_energy_ev_natom: {cv_rmsle_formation:.5f}")
print(f"CV RMSLE - bandgap_energy_ev: {cv_rmsle_bandgap:.5f}")
print(f"CV RMSLE - Mean: {cv_rmsle_mean:.5f}")

# Create submission file
submission_df = pd.DataFrame(
    {"id": test_df["id"], TARGETS[0]: test_preds[:, 0], TARGETS[1]: test_preds[:, 1]}
)

# Create output directory if it doesn't exist
os.makedirs("./1", exist_ok=True)
submission_df.to_csv("./1/submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: cv_rmsle_formation,
        TARGETS[1]: cv_rmsle_bandgap,
        "mean": cv_rmsle_mean,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "XGBoost (independent models per target)",
}
import json

with open("./1/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print("Submission file created at ./1/submission.csv")
print("Metrics file created at ./1/metrics.json")