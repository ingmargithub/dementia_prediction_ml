from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve().parents[0]  # Adjust the parent index if needed
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# TO DO: simplify this to __init__ maybe
from dementia_prediction.config import INPUT_FILE, RESULTS_DIR, SHEET_NAME
from dementia_prediction.evaluation import evaluate_models
from dementia_prediction.features import build_feature_sets, prepare_dataframe
from dementia_prediction.inference import compare_final_models
from dementia_prediction.reporting import save_results
from dementia_prediction.missingness import check_missingness
from dementia_prediction.diagnose_features import diagnose_features
print("Project functions imported.")

# --- RUN ANALYSIS --- #

# read
print("Read data file (memory issues fixed with calamine engine!)")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input data file not found: {INPUT_FILE}")
raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="calamine")
df = prepare_dataframe(raw)
print(f"Raw observations: {len(raw)}")
print(f"Analysis observations: {len(df)}")
print(f"Dementia events: {int(df['event'].sum())}")

# features
print("Get pre-specified features")
feature_sets = build_feature_sets(df)
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} predictors")
   
# check for NaNs and Infs in df and deal with them
missingness = check_missingness(df, feature_sets,)
#df = median_impute(df, feature_sets) # let the CV imputer do this

# check if features are collinear etc
diagnose_features(df, feature_sets)

# evaluate models with feature sets
print("\nCross-validated survival prediction")
metrics, oof_predictions, fold_results = evaluate_models(df, feature_sets)
print(metrics.round(4).to_string())

# TO DO: fix comparison between base & base + regional DAWM ratings
print("\n TO DO: Nested Cox-model comparison")
comparison = compare_final_models(df, feature_sets["base"], feature_sets["regional_dawm"])
print(f"Likelihood-ratio statistic: {comparison.likelihood_ratio:.4f}")
print(f"Degrees of freedom: {comparison.degrees_of_freedom}")
print(f"Likelihood-ratio p-value: {comparison.p_value:.6g}")
comparison_df = pd.DataFrame([comparison.__dict__])

# save the results!
print("Writing results")
save_results(RESULTS_DIR, metrics, oof_predictions, fold_results, comparison_df)
print(f"Results written to {RESULTS_DIR}")
