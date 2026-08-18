from __future__ import annotations
import sys
import pandas as pd
from pathlib import Path

# local paths
PROJECT_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# local functions
import dementia_prediction as dem_pred

# --- RUN ANALYSIS --- #

# (PIP) INSTALL CALAMINE ENGINE FOR OPENXLSX IF NEEDED
print("Read data file")
if not dem_pred.INPUT_FILE.exists():
    raise FileNotFoundError(f"Input data file not found: {dem_pred.INPUT_FILE}")
raw = pd.read_excel(dem_pred.INPUT_FILE, sheet_name=dem_pred.SHEET_NAME, engine="calamine")
df = dem_pred.prepare_dataframe(raw)
print(f"Raw observations: {len(raw)}")
print(f"Analysis observations: {len(df)}")
print(f"Dementia events: {int(df['event'].sum())}")

# features
feature_sets = dem_pred.build_feature_sets(df)
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} predictors")
   
# evaluate models with feature sets
print("\nCross-validated survival prediction")
metrics, oof_predictions, fold_results = dem_pred.evaluate_models(df, feature_sets)
print(metrics.round(4).to_string())

# TO DO: fix comparison between base & base + regional DAWM ratings
print("\n TO DO: Nested Cox-model comparison")
comparison = dem_pred.compare_final_models(df, feature_sets["base"], feature_sets["regional_dawm"])
print(f"Likelihood-ratio: {comparison.likelihood_ratio:.4f}")
print(f"Likelihood-ratio p-value: {comparison.p_value:.6g}")
comparison_df = pd.DataFrame([comparison.__dict__])

# save the results!
print("Writing results")
dem_pred.save_results(dem_pred.RESULTS_DIR, metrics, oof_predictions, fold_results, comparison_df)
print(f"Results written to {dem_pred.RESULTS_DIR}")
