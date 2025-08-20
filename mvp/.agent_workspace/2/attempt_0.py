import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
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
test_ids = test_df["id"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Cross-validation setup
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Model training and prediction
oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

for i, target in enumerate(TARGETS):
    print(f"Training model for {target}...")
    y_target = y[target].values

    # Log transform target
    y_target_log = np.log1p(y_target)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_target_log)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_target_log[train_idx], y_target_log[val_idx]

        model = Ridge(random_state=42)
        model.fit(X_train, y_train)

        val_preds_log = model.predict(X_val)
        oof_preds[val_idx, i] = val_preds_log

        test_fold_preds_log = model.predict(X_test_scaled)
        test_preds[:, i] += test_fold_preds_log / NFOLDS

    print(f"Fold {fold+1} completed for {target}")

# Inverse transform predictions and clip
oof_preds_exp = np.expm1(oof_preds)
oof_preds_clipped = np.clip(oof_preds_exp, 0, None)

test_preds_exp = np.expm1(test_preds)
test_preds_clipped = np.clip(test_preds_exp, 0, None)

# Calculate OOF RMSLE
oof_rmsle_scores = {}
for i, target in enumerate(TARGETS):
    # Ensure true values are non-negative before log1p for RMSLE calculation
    y_true_clipped = np.clip(y[target].values, 0, None)
    oof_rmsle_scores[target] = rmsle(y_true_clipped, oof_preds_clipped[:, i])
    print(f"OOF RMSLE for {target}: {oof_rmsle_scores[target]:.4f}")

mean_oof_rmsle = np.mean(list(oof_rmsle_scores.values()))
print(f"\nMean OOF RMSLE: {mean_oof_rmsle:.4f}")

# Create submission file
submission_df = pd.DataFrame({"id": test_ids})
submission_df[TARGETS[0]] = test_preds_clipped[:, 0]
submission_df[TARGETS[1]] = test_preds_clipped[:, 1]

# Ensure output directory exists
os.makedirs("./2", exist_ok=True)
submission_df.to_csv("./2/submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: oof_rmsle_scores[TARGETS[0]],
        TARGETS[1]: oof_rmsle_scores[TARGETS[1]],
        "mean": mean_oof_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "Ridge (5-fold CV)",
}
import json

with open("./2/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nSubmission file created: ./2/submission.csv")
print("Metrics file created: ./2/metrics.json")
print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")