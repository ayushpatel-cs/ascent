import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import json


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature Engineering (example: interaction terms, polynomial features could be added here)
# For simplicity, we'll use the provided features directly.
# Consider adding more complex features if performance is not satisfactory.

# Define features (X) and targets (y)
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
FEATURES = [col for col in train_df.columns if col not in ["id"] + TARGETS]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]
test_ids = test_df["id"]

# Preprocessing: Scaling features
# Scaling can help some models, though LightGBM is less sensitive to it.
# We'll scale features for consistency and potential future model changes.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# Model Training with Cross-Validation
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), len(TARGETS)))
test_preds = np.zeros((len(test_df), len(TARGETS)))

cv_rmsle_scores = {"formation_energy_ev_natom": [], "bandgap_energy_ev": []}

for i, target in enumerate(TARGETS):
    print(f"Training model for: {target}")
    y_target = y[target]

    # Log transform the target variable
    y_target_log = np.log1p(y_target)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_scaled, y_target_log)):
        print(f"Fold {fold_+1}/{NFOLDS}")
        trn_data = lgb.Dataset(X_scaled.iloc[trn_idx], label=y_target_log.iloc[trn_idx])
        val_data = lgb.Dataset(X_scaled.iloc[val_idx], label=y_target_log.iloc[val_idx])

        # LightGBM parameters - tuned for reasonable performance
        # These can be further optimized using hyperparameter tuning libraries
        params = {
            "objective": "regression_l1",  # MAE objective often robust
            "metric": "rmsle",
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
            "seed": 42 + fold_,
            "boosting_type": "gbdt",
        }

        model = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=-1)],
        )

        val_preds_log = model.predict(X_scaled.iloc[val_idx])
        oof_preds[val_idx, i] = val_preds_log

        test_preds_fold = model.predict(X_test_scaled)
        test_preds[:, i] += test_preds_fold / folds.n_splits

    # Inverse transform predictions and clip to be non-negative
    oof_preds_inv = np.expm1(oof_preds[:, i])
    oof_preds_inv = np.clip(oof_preds_inv, 0, None)

    # Calculate RMSLE for OOF predictions
    score = rmsle(y_target, oof_preds_inv)
    cv_rmsle_scores[target].append(score)
    print(f"RMSLE for {target}: {score}")

# Calculate mean RMSLE across folds for each target
mean_cv_rmsle_per_target = {
    target: np.mean(scores) for target, scores in cv_rmsle_scores.items()
}
mean_cv_rmsle_overall = np.mean(list(mean_cv_rmsle_per_target.values()))

print("\n--- Cross-Validation RMSLE Summary ---")
for target, score in mean_cv_rmsle_per_target.items():
    print(f"{target}: {score:.4f}")
print(f"Overall Mean RMSLE: {mean_cv_rmsle_overall:.4f}")

# Final Test Predictions
# Inverse transform test predictions and clip to be non-negative
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
        TARGETS[0]: np.mean(cv_rmsle_scores[TARGETS[0]]),
        TARGETS[1]: np.mean(cv_rmsle_scores[TARGETS[1]]),
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