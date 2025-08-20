import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature Engineering (if any, none specified, so using raw features)
# For simplicity, we'll use the provided features directly.
# In a real scenario, more feature engineering might be beneficial.

# Define features and targets
feature_cols = [
    "spacegroup",
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
]
target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

X = train_df[feature_cols]
y = train_df[target_cols]
X_test = test_df[feature_cols]

# Preprocessing: Scale numerical features
# Although LightGBM is less sensitive to feature scaling, it's good practice.
# We'll use a ColumnTransformer to handle this.
# For simplicity, let's assume all features are numerical for now.
# If there were categorical features, they would need different handling.

# Identify numerical features (all are numerical in this case)
numerical_features = feature_cols

# Create preprocessing pipelines for numerical features
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_features)],
    remainder="passthrough",  # Keep other columns if any (though none here)
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Convert back to DataFrame to maintain column names if needed, or use numpy arrays
# For LightGBM, numpy arrays are fine.
X_processed = pd.DataFrame(
    X_processed, columns=numerical_features
)  # Re-add column names for clarity if needed
X_test_processed = pd.DataFrame(X_test_processed, columns=numerical_features)


# LightGBM Parameters (tuned from previous runs or reasonable defaults)
lgb_params = {
    "objective": "regression_l1",  # MAE objective often works well for RMSLE
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
    "seed": 42,
    "boosting_type": "gbdt",
}

# --- Cross-validation to get CV scores (as done previously) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmsle_formation_energy = []
cv_rmsle_bandgap_energy = []

# Use processed data for CV
X_cv = X_processed
y_formation_energy = y["formation_energy_ev_natom"]
y_bandgap_energy = y["bandgap_energy_ev"]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv, y_formation_energy)):
    X_train, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
    y_train_form, y_val_form = (
        y_formation_energy.iloc[train_idx],
        y_formation_energy.iloc[val_idx],
    )
    y_train_band, y_val_band = (
        y_bandgap_energy.iloc[train_idx],
        y_bandgap_energy.iloc[val_idx],
    )

    # Train for formation_energy_ev_natom
    model_formation = lgb.LGBMRegressor(**lgb_params)
    # Apply log1p transformation to target
    y_train_form_log = np.log1p(y_train_form)
    y_val_form_log = np.log1p(y_val_form)
    # Removed early stopping callback as it caused the error
    model_formation.fit(
        X_train,
        y_train_form_log,
        eval_set=[(X_val, y_val_form_log)],
        eval_metric="rmsle",
    )
    val_preds_log = model_formation.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.clip(val_preds, 0, None)  # Ensure non-negative
    cv_rmsle_formation_energy.append(rmsle(y_val_form, val_preds))

    # Train for bandgap_energy_ev
    model_bandgap = lgb.LGBMRegressor(**lgb_params)
    # Apply log1p transformation to target
    y_train_band_log = np.log1p(y_train_band)
    y_val_band_log = np.log1p(y_val_band)
    # Removed early stopping callback
    model_bandgap.fit(
        X_train,
        y_train_band_log,
        eval_set=[(X_val, y_val_band_log)],
        eval_metric="rmsle",
    )
    val_preds_log = model_bandgap.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.clip(val_preds, 0, None)  # Ensure non-negative
    cv_rmsle_bandgap_energy.append(rmsle(y_val_band, val_preds))

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
# Removed early stopping callback
model_formation_final.fit(X_processed, y_train_formation_log)

# Final model for bandgap_energy_ev
model_bandgap_final = lgb.LGBMRegressor(**lgb_params)
y_train_bandgap_log = np.log1p(y_bandgap_energy)
# Removed early stopping callback
model_bandgap_final.fit(X_processed, y_train_bandgap_log)

# Predict on test data
test_preds_formation_log = model_formation_final.predict(X_test_processed)
test_preds_formation = np.expm1(test_preds_formation_log)
test_preds_formation = np.clip(test_preds_formation, 0, None)

test_preds_bandgap_log = model_bandgap_final.predict(X_test_processed)
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
# Ensure the directory ./4/ exists
import os

if not os.path.exists("./4"):
    os.makedirs("./4")
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