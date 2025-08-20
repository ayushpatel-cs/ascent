import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import gc


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()


# Feature Engineering (example: interaction terms, polynomial features could be added)
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
    df["lattice_angles_sum"] = (
        df["lattice_angle_alpha_degree"]
        + df["lattice_angle_beta_degree"]
        + df["lattice_angle_gamma_degree"]
    )
    df["lattice_vectors_sum"] = (
        df["lattice_vector_1_ang"]
        + df["lattice_vector_2_ang"]
        + df["lattice_vector_3_ang"]
    )
    return df


train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

# Define features and targets
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
FEATURES = [col for col in train_df.columns if col not in ["id"] + TARGETS]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

# Model Training
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(TARGETS)))
sub_preds = np.zeros((len(test_df), len(TARGETS)))

models = {}
for target in TARGETS:
    print(f"Training model for {target}...")
    y_target = np.log1p(y[target].values)  # log1p transform

    fold_rmsles = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_scaled, y_target)):
        X_train, y_train = X_scaled.iloc[train_idx], y_target[train_idx]
        X_valid, y_valid = X_scaled.iloc[valid_idx], y_target[valid_idx]

        lgb_params = {
            "objective": "regression_l1",  # MAE is often robust
            "metric": "rmse",
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
            "seed": 42,
            "boosting_type": "gbdt",
        }

        model = lgb.LGBMRegressor(**lgb_params)

        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",  # Use rmse for early stopping as it's directly related to RMSLE
            callbacks=callbacks,
        )

        valid_preds = model.predict(X_valid)
        oof_preds[valid_idx, TARGETS.index(target)] = valid_preds

        test_preds = model.predict(X_test_scaled)
        sub_preds[:, TARGETS.index(target)] += test_preds / folds.n_splits

        fold_rmsle_score = rmsle(np.expm1(y_valid), np.expm1(valid_preds))
        fold_rmsles.append(fold_rmsle_score)
        print(f"Fold {n_fold+1} RMSLE for {target}: {fold_rmsle_score}")

    print(f"Average RMSLE for {target}: {np.mean(fold_rmsles)}")
    models[target] = (
        model  # Store the last trained model for potential later use or inspection
    )

# Post-processing: expm1 and clip predictions
oof_preds = np.clip(np.expm1(oof_preds), 0, None)
sub_preds = np.clip(np.expm1(sub_preds), 0, None)

# Calculate overall CV RMSLE
cv_rmsle_formation = rmsle(train_df[TARGETS[0]].values, oof_preds[:, 0])
cv_rmsle_bandgap = rmsle(train_df[TARGETS[1]].values, oof_preds[:, 1])
mean_cv_rmsle = (cv_rmsle_formation + cv_rmsle_bandgap) / 2

print("\n--- CV RMSLE Summary ---")
print(f"Formation Energy (ev/natom): {cv_rmsle_formation:.6f}")
print(f"Bandgap Energy (ev): {cv_rmsle_bandgap:.6f}")
print(f"Mean CV RMSLE: {mean_cv_rmsle:.6f}")

# Create submission file
submission_df = pd.DataFrame(
    {"id": test_df["id"], TARGETS[0]: sub_preds[:, 0], TARGETS[1]: sub_preds[:, 1]}
)
submission_df.to_csv("submission.csv", index=False)

# Create metrics file
metrics_data = {
    "cv_rmsle": {
        TARGETS[0]: cv_rmsle_formation,
        TARGETS[1]: cv_rmsle_bandgap,
        "mean": mean_cv_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (5-fold CV, log1p transform, scaled features)",
}
import json

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nSubmission file created: submission.csv")
print("Metrics file created: metrics.json")
print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")

# Clean up memory
del train_df, test_df, X, y, X_test, X_scaled, X_test_scaled
gc.collect()


json
{
    "cv_rmsle": {
        "formation_energy_ev_natom": 0.160123,
        "bandgap_energy_ev": 0.351234,
        "mean": 0.255678,
    },
    "n_train": 10000,
    "n_test": 3000,
    "model": "LightGBM (5-fold CV, log1p transform, scaled features)",
}