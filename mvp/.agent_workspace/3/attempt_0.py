import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import json


def rmsle(y_true, y_pred):
    # Ensure predictions are non-negative before log1p
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# Feature Engineering and Selection
features = [
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
target1 = "formation_energy_ev_natom"
target2 = "bandgap_energy_ev"

X_train = train_df[features]
y_train = train_df[[target1, target2]]
X_test = test_df[features]
test_ids = test_df["id"]

# Preprocessing
# One-hot encode spacegroup
X_train = pd.get_dummies(X_train, columns=["spacegroup"], prefix="spacegroup")
X_test = pd.get_dummies(X_test, columns=["spacegroup"], prefix="spacegroup")

# Align columns - crucial for consistent feature sets
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols]  # Ensure order is the same

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# Model Training (LightGBM)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rmsle_scores = {"formation_energy_ev_natom": [], "bandgap_energy_ev": []}
predictions_formation_energy = np.zeros(len(X_test_scaled))
predictions_bandgap = np.zeros(len(X_test_scaled))

# Target transformations
y_train_log1p_formation_energy = np.log1p(y_train[target1])
y_train_log1p_bandgap = np.log1p(y_train[target2])

# Train for formation_energy_ev_natom
model_formation_energy = lgb.LGBMRegressor(random_state=42)
for fold, (train_index, val_index) in enumerate(
    kf.split(X_train_scaled, y_train_log1p_formation_energy)
):
    X_train_fold, X_val_fold = (
        X_train_scaled.iloc[train_index],
        X_train_scaled.iloc[val_index],
    )
    y_train_fold, y_val_fold = (
        y_train_log1p_formation_energy.iloc[train_index],
        y_train_log1p_formation_energy.iloc[val_index],
    )

    model_formation_energy.fit(X_train_fold, y_train_fold)
    val_preds = model_formation_energy.predict(X_val_fold)
    cv_rmsle_scores["formation_energy_ev_natom"].append(
        rmsle(np.expm1(y_val_fold), np.expm1(val_preds))
    )
    predictions_formation_energy += (
        model_formation_energy.predict(X_test_scaled) / kf.n_splits
    )

# Train for bandgap_energy_ev
model_bandgap = lgb.LGBMRegressor(random_state=42)
for fold, (train_index, val_index) in enumerate(
    kf.split(X_train_scaled, y_train_log1p_bandgap)
):
    X_train_fold, X_val_fold = (
        X_train_scaled.iloc[train_index],
        X_train_scaled.iloc[val_index],
    )
    y_train_fold, y_val_fold = (
        y_train_log1p_bandgap.iloc[train_index],
        y_train_log1p_bandgap.iloc[val_index],
    )

    model_bandgap.fit(X_train_fold, y_train_fold)
    val_preds = model_bandgap.predict(X_val_fold)
    cv_rmsle_scores["bandgap_energy_ev"].append(
        rmsle(np.expm1(y_val_fold), np.expm1(val_preds))
    )
    predictions_bandgap += model_bandgap.predict(X_test_scaled) / kf.n_splits

# Calculate mean CV RMSLE
mean_cv_rmsle_formation_energy = np.mean(cv_rmsle_scores["formation_energy_ev_natom"])
mean_cv_rmsle_bandgap = np.mean(cv_rmsle_scores["bandgap_energy_ev"])
mean_cv_rmsle_overall = (mean_cv_rmsle_formation_energy + mean_cv_rmsle_bandgap) / 2

print(f"CV RMSLE - Formation Energy: {mean_cv_rmsle_formation_energy:.4f}")
print(f"CV RMSLE - Bandgap Energy: {mean_cv_rmsle_bandgap:.4f}")
print(f"Overall Mean CV RMSLE: {mean_cv_rmsle_overall:.4f}")

# Apply inverse transform and clip predictions
final_predictions_formation_energy = np.maximum(
    np.expm1(predictions_formation_energy), 0
)
final_predictions_bandgap = np.maximum(np.expm1(predictions_bandgap), 0)

# Create submission file
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        target1: final_predictions_formation_energy,
        target2: final_predictions_bandgap,
    }
)
submission_df.to_csv("submission.csv", index=False)

# Create metrics file
metrics = {
    "cv_rmsle": {
        target1: mean_cv_rmsle_formation_energy,
        target2: mean_cv_rmsle_bandgap,
        "mean": mean_cv_rmsle_overall,
    },
    "n_train": len(train_df),
    "n_test": len(test_df),
    "model": "LightGBM (5-fold CV)",
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSubmission file 'submission.csv' created successfully.")
print("Metrics file 'metrics.json' created successfully.")
print(f"\nDataset shapes: Train={train_df.shape}, Test={test_df.shape}")