from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dementia_prediction.config import INPUT_FILE, RESULTS_DIR, SHEET_NAME
from dementia_prediction.evaluation import evaluate_models
from dementia_prediction.features import build_feature_sets, prepare_dataframe
from dementia_prediction.inference import compare_final_models
from dementia_prediction.reporting import save_results


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input data file not found: {INPUT_FILE}")
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    df = prepare_dataframe(raw)
    print(f"Raw observations: {len(raw)}")
    print(f"Analysis observations: {len(df)}")
    print(f"Dementia events: {int(df['event'].sum())}")
    feature_sets = build_feature_sets(df)
    for name, features in feature_sets.items():
        print(f"{name}: {len(features)} predictors")
    metrics, oof_predictions, fold_results = evaluate_models(df, feature_sets)
    print("\nCross-validated survival prediction")
    print(metrics.round(4).to_string())
    comparison = compare_final_models(df, feature_sets["base"], feature_sets["regional_dawm"])
    print("\nNested Cox-model comparison")
    print(f"Likelihood-ratio statistic: {comparison.likelihood_ratio:.4f}")
    print(f"Degrees of freedom: {comparison.degrees_of_freedom}")
    print(f"Likelihood-ratio p-value: {comparison.p_value:.6g}")
    comparison_df = pd.DataFrame([comparison.__dict__])
    save_results(RESULTS_DIR, metrics, oof_predictions, fold_results, comparison_df)
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
