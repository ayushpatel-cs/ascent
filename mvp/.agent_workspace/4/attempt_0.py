import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
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


# Feature Engineering (same as previous iterations)
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
    df["avg_lattice_perpendicular"] = (
        df["lattice_vector_1_ang"]
        + df["lattice_vector_2_ang"]
        + df["lattice_vector_3_ang"]
    ) / 3
    df["lattice_angles_sum"] = (
        df["lattice_angle_alpha_degree"]
        + df["lattice_angle_beta_degree"]
        + df["lattice_angle_gamma_degree"]
    )
    df["lattice_angles_product"] = (
        df["lattice_angle_alpha_degree"]
        * df["lattice_angle_beta_degree"]
        * df["lattice_angle_gamma_degree"]
    )
    df["lattice_angles_mean"] = df[
        [
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ]
    ].mean(axis=1)
    df["lattice_vectors_mean"] = df[
        ["lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang"]
    ].mean(axis=1)
    df["lattice_vectors_std"] = df[
        ["lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang"]
    ].std(axis=1)
    df["atom_density"] = df["number_of_total_atoms"] / df["lattice_volume"]
    df["al_ga_ratio"] = df["percent_atom_al"] / (df["percent_atom_ga"] + 1e-6)
    df["al_in_ratio"] = df["percent_atom_al"] / (df["percent_atom_in"] + 1e-6)
    df["ga_in_ratio"] = df["percent_atom_ga"] / (df["percent_atom_in"] + 1e-6)
    df["al_ga_in_sum"] = (
        df["percent_atom_al"] + df["percent_atom_ga"] + df["percent_atom_in"]
    )
    return df


train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

# Define features and targets
features = [
    col
    for col in train_df.columns
    if col not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
target_formation_energy = "formation_energy_ev_natom"
target_bandgap_energy = "bandgap_energy_ev"

X = train_df[features]
y_formation_energy = train_df[target_formation_energy]
y_bandgap_energy = train_df[target_bandgap_energy]

X_test = test_df[features]

# --- Retrain on full data and evaluate CV ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmsle_formation_energy = []
cv_rmsle_bandgap_energy = []

# LightGBM parameters (consistent with previous successful runs)
lgb_params = {
    "objective": "regression_l1",  # MAE is often more robust to outliers
    "metric": "rmsle",
    "n_estimators": 1500,
    "learning_rate": 0.03,
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

# Train and predict for formation_energy_ev_natom
formation_energy_preds_cv = np.zeros(len(train_df))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_formation_energy)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = (
        y_formation_energy.iloc[train_idx],
        y_formation_energy.iloc[val_idx],
    )

    # Apply log1p transformation
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model_formation = lgb.LGBMRegressor(**lgb_params)
    model_formation.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="rmsle",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    val_preds_log = model_formation.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.clip(val_preds, 0, None)  # Ensure non-negative

    cv_rmsle_formation_energy.append(rmsle(y_val, val_preds))
    formation_energy_preds_cv[val_idx] = (
        val_preds  # Store for potential ensemble later if needed
    )

# Train and predict for bandgap_energy_ev
bandgap_energy_preds_cv = np.zeros(len(train_df))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bandgap_energy)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_bandgap_energy.iloc[train_idx], y_bandgap_energy.iloc[val_idx]

    # Apply log1p transformation
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model_bandgap = lgb.LGBMRegressor(**lgb_params)
    model_bandgap.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="rmsle",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    val_preds_log = model_bandgap.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.clip(val_preds, 0, None)  # Ensure non-negative

    cv_rmsle_bandgap_energy.append(rmsle(y_val, val_preds))
    bandgap_energy_preds_cv[val_idx] = (
        val_preds  # Store for potential ensemble later if needed
    )

mean_cv_rmsle_formation_energy = np.mean(cv_rmsle_formation_energy)
mean_cv_rmsle_bandgap_energy = np.mean(cv_rmsle_bandgap_energy)
mean_cv_rmsle = (mean_cv_rmsle_formation_energy + mean_cv_rmsle_bandgap_energy) / 2

print(f"CV RMSLE (Formation Energy): {mean_cv_rmsle_formation_energy:.4f}")
print(f"CV RMSLE (Bandgap Energy): {mean_cv_rmsle_bandgap_energy:.4f}")
print(f"Mean CV RMSLE: {mean_cv_rmsle:.4f}")

# --- Retrain final models on full training data ---

# Final model for formation_energy_ev_natom
model_formation_final = lgb.LGBMRegressor(**lgb_params)
y_train_formation_log = np.log1p(y_formation_energy)
model_formation_final.fit(X, y_train_formation_log)

# Final model for bandgap_energy_ev
model_bandgap_final = lgb.LGBMRegressor(**lgb_params)
y_train_bandgap_log = np.log1p(y_bandgap_energy)
model_bandgap_final.fit(X, y_train_bandgap_log)

# Predict on test data
test_preds_formation_log = model_formation_final.predict(X_test)
test_preds_formation = np.expm1(test_preds_formation_log)
test_preds_formation = np.clip(test_preds_formation, 0, None)

test_preds_bandgap_log = model_bandgap_final.predict(X_test)
test_preds_bandgap = np.expm1(test_preds_bandgap_log)
test_preds_bandgap = np.clip(test_preds_bandgap, 0, None)

# Create submission file
submission_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "formation_energy_ev_natom": test_preds_formation,
        "bandgap_energy_ev": test_preds_bandgap,
    }
)

# Save submission
submission_df.to_csv("./4/submission.csv", index=False)

# Save metrics
metrics = {
    "cv_rmsle": {
        "formation_energy_ev_natom": mean_cv_rmsle_formation_energy,
        "bandgap_energy_ev": mean_cv_rmsle_bandgap_energy,
        "mean": mean_cv_rmsle,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (retrained on full data, MAE objective, RMSLE metric)",
}

with open("./4/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSubmission file created: ./4/submission.csv")
print("Metrics file created: ./4/metrics.json")
print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")