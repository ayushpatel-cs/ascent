# --- Fixed competition description (short, stable) ---
PROBLEM_PROMPT = """\
Kaggle: Predicting Transparent Conductors — two-target regression.

Objective:
Predict for each material id in test.csv:
- formation_energy_ev_natom
- bandgap_energy_ev

Evaluation:
Column-wise RMSLE with log1p:
RMSLE = sqrt( (1/n) * Σ (log(1+p_i) - log(1+a_i))^2 ); final score = mean over the two targets.

Local files:
- ./train.csv  (features + both targets)
- ./test.csv   (features only)

Feature columns in test.csv:
id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in,
lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang,
lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

Submission (required):
Write ./submission.csv with EXACT header:
id,formation_energy_ev_natom,bandgap_energy_ev

Environment constraints:
CPU-only; deterministic; no internet or package installs.
Use Python 3.10+, numpy, pandas, scikit-learn (optionally lightgbm/xgboost if already available).

Artifacts for iteration (requested):
Write ./metrics.json with CV RMSLE per target and mean; also print dataset shapes and CV summary.
"""

# --- Dev agent starter context (include verbatim in orchestrator prompts) ---
DEV_CONTEXT_STARTER = """\
You are the Dev Agent.

Allowed files:
- ./train.csv, ./test.csv

Feature columns in test.csv:
id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in,
lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang,
lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

Objective:
Predict for each material id in test.csv:
- formation_energy_ev_natom
- bandgap_energy_ev

Evaluation:
Column-wise RMSLE with log1p:
RMSLE = sqrt( (1/n) * Σ (log(1+p_i) - log(1+a_i))^2 ); final score = mean over the two targets.

Environment:
- Python 3.10+, CPU-only, deterministic; no internet or package installs.
- Available libraries: numpy, pandas, scikit-learn, lightgbm, xgboost, statsmodels, scipy.
- Return ONLY a single Python fenced block with self-contained code.

IO contract (always do):
1) Write ./{id}/submission.csv with EXACT header:
   id,formation_energy_ev_natom,bandgap_energy_ev
2) Write ./{id}/metrics.json with:
   {
     "cv_rmsle": {
       "formation_energy_ev_natom": <float>,
       "bandgap_energy_ev": <float>,
       "mean": <float>
     },
     "n_train": <int>,
     "n_test": <int>,
     "model": "<brief description>"
   }
3) Print dataset shapes and a concise CV RMSLE summary.

Modeling guidance (optional, keep fast <3 min CPU):
- 5-fold KFold(shuffle=True, random_state=42).
- Train on log1p(y); predict with expm1; clip to >= 0.
- Fit two regressors or a MultiOutputRegressor.
- DO NOT USE EARLY STOPPING for training your model. This will cause bugs and avoid it all possible costs.
"""

# --- Orchestrator output format: one self-contained prompt to the Dev Agent ---
ORCHESTRATOR_TASK_TEMPLATE = """\
You are the Orchestrator.

Using the materials below, produce ONE concise, self-contained prompt addressed to the Dev Agent.
Requirements for your output:
- List the usable local files the Dev Agent may read.
- Include the Dev Context verbatim so that the agent has context on the problem.
- Choose exactly ONE concrete next task that moves forward given the Blackboard
  (avoid repeating prior attempts; if the last run failed, first instruct a targeted fix).
- Specify deliverables and constraints explicitly:
  * must write ./submission.csv with the exact header;
  * should write ./metrics.json with CV RMSLE per-target and mean;
  * CPU-only, deterministic, no network or installs; use numpy/pandas/sklearn (LGBM/XGB only if present).
- Keep it ≤ 500 words. No meta-commentary about your reasoning.
- Output ONLY the final prompt text (no JSON, no code fences).

=== Problem ===
{problem_description}

=== Blackboard (latest summary) ===
{blackboard}

=== Dev Context (starter) ===
{dev_description}
"""