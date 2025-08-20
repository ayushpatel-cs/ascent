import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor


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

# Ensure all feature columns are numeric, coercing errors to NaN
for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Simple imputation for any NaNs that might have been created
for col in FEATURES:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X_test[col].median())

# Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train RandomForestRegressor for each target
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    multi_output_model = MultiOutputRegressor(model)

    # Log transform targets
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    multi_output_model.fit(X_train, y_train_log)

    # Predict on validation set
    val_preds_log = multi_output_model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0  # Clip predictions to be non-negative

    oof_preds[val_idx] = val_preds

    # Predict on test set
    test_preds_log = multi_output_model.predict(X_test)
    test_preds_fold = np.expm1(test_preds_log)
    test_preds_fold[test_preds_fold < 0] = 0  # Clip predictions to be non-negative
    test_preds += test_preds_fold / kf.n_splits

# Calculate OOF RMSLE
oof_rmsle_scores = {}
for i, target in enumerate(TARGETS):
    oof_rmsle_scores[target] = rmsle(y[target], oof_preds[:, i])
oof_rmsle_scores["mean"] = np.mean(list(oof_rmsle_scores.values()))

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"]})
for i, target in enumerate(TARGETS):
    submission_df[target] = test_preds[:, i]

submission_df.to_csv("./3/submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": oof_rmsle_scores,
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "RandomForestRegressor (MultiOutputRegressor)",
}
import json

with open("./3/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

# Print summary
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print("\nCross-validation RMSLE scores:")
for target, score in oof_rmsle_scores.items():
    print(f"- {target}: {score:.4f}")