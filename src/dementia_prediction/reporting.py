from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_results(results_dir: Path, metrics: pd.DataFrame, predictions, fold_results, comparison: pd.DataFrame) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "cross_validated_metrics.csv")
    comparison.to_csv(results_dir / "cox_nested_model_comparison.csv", index=False)
    for model_name, risk in predictions.items():
        pd.DataFrame({"oof_risk_score": risk}).to_csv(results_dir / f"oof_risk_{model_name}.csv", index=False)
    for model_name, folds in fold_results.items():
        pd.DataFrame(folds).to_csv(results_dir / f"folds_{model_name}.csv", index=False)
