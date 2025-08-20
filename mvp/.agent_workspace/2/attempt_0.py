import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import json


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()


# Feature Engineering (example, can be expanded)
def feature_engineer(df):
    df["lattice_volume"] = (
        df["lattice_vector_1_ang"]
        * df["lattice_vector_2_ang"]
        * df["lattice_vector_3_ang"]
        * np.sqrt(
            1
            - np.cos(np.radians(df["lattice_angle_alpha_degree"])) ** 2
            - np.cos(np.radians(df["lattice_angle_beta_degree"])) ** 2
            - np.cos(np.radians(df["lattice_angle_gamma_degree"])) ** 2
            + 2
            * np.cos(np.radians(df["lattice_angle_alpha_degree"]))
            * np.cos(np.radians(df["lattice_angle_beta_degree"]))
            * np.cos(np.radians(df["lattice_angle_gamma_degree"]))
        )
    )
    df["density"] = df["number_of_total_atoms"] / df["lattice_volume"]
    df["spacegroup"] = df["spacegroup"].astype("category")
    return df


train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

# Define features and targets
feature_cols = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X_train = train_df[feature_cols]
y_train = train_df[target_cols]
X_test = test_df[feature_cols]

# Handle categorical features for LightGBM
categorical_features = ["spacegroup"]
for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
    if col in X_test.columns:
        X_test[col] = X_test[col].astype("category")

# Model Training (LightGBM)
models = {}
predictions = {}
cv_rmsle_scores = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for target in target_cols:
    print(f"Training model for {target}...")
    y_train_target = np.log1p(y_train[target])

    fold_rmsles = []
    fold_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_target)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = (
            y_train_target.iloc[train_idx],
            y_train_target.iloc[val_idx],
        )

        # LGBM parameters (tuned for speed and reasonable performance)
        lgb_params = {
            "objective": "regression_l1",  # MAE objective often robust
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

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="rmsle",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)],
            categorical_feature=categorical_features,
        )

        val_preds = model.predict(X_val_fold)
        fold_rmsle = rmsle(np.expm1(y_val_fold), np.expm1(val_preds))
        fold_rmsles.append(fold_rmsle)

        fold_preds += model.predict(X_test) / kf.n_splits

    cv_rmsle_scores[target] = np.mean(fold_rmsles)
    predictions[target] = np.clip(
        np.expm1(fold_preds), 0, None
    )  # Clip predictions to be non-negative

# Retrain on full data
final_models = {}
final_predictions = {}

for target in target_cols:
    print(f"Retraining {target} on full data...")
    y_train_target = np.log1p(y_train[target])

    lgb_params = {
        "objective": "regression_l1",
        "metric": "rmsle",
        "n_estimators": 1000,  # Use the number of estimators from early stopping or a slightly higher value
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

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train_target, categorical_feature=categorical_features)
    final_models[target] = model
    final_predictions[target] = np.clip(np.expm1(model.predict(X_test)), 0, None)

# Create submission file
submission_df = pd.DataFrame({"id": test_df["id"]})
submission_df["formation_energy_ev_natom"] = final_predictions[target_cols[0]]
submission_df["bandgap_energy_ev"] = final_predictions[target_cols[1]]
submission_df.to_csv("submission.csv", index=False)

# Calculate CV RMSLE on full training data for metrics.json
cv_rmsle_full_train = {}
for target in target_cols:
    y_true_target = np.log1p(y_train[target])

    fold_rmsles_full_train = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_target)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = (
            y_train_target.iloc[train_idx],
            y_train_target.iloc[val_idx],
        )

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

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="rmsle",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)],
            categorical_feature=categorical_features,
        )

        val_preds = model.predict(X_val_fold)
        fold_rmsle = rmsle(np.expm1(y_val_fold), np.expm1(val_preds))
        fold_rmsles_full_train.append(fold_rmsle)

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

print("Dataset Shapes:")
print(f"Train data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print("\nCV RMSLE Scores (on full training data):")
print(f"  Formation Energy: {cv_rmsle_full_train['formation_energy_ev_natom']:.6f}")
print(f"  Bandgap Energy: {cv_rmsle_full_train['bandgap_energy_ev']:.6f}")
print(f"  Mean CV RMSLE: {mean_cv_rmsle:.6f}")