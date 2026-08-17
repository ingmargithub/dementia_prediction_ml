from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from .config import N_OUTER, RANDOM_STATE
from .model import make_cox_pipeline, make_survival_target


def _folds(df: pd.DataFrame):
    splitter = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    return splitter.split(df, df["event"].astype(int))


def oof_predictions(df: pd.DataFrame, features: list[str]):
    """Generate out-of-fold Cox relative-risk scores; no time horizon is used."""
    X = df[features]
    y = make_survival_target(df)
    oof_risk = np.full(len(df), np.nan)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(_folds(df), start=1):
        model = make_cox_pipeline(X.iloc[train_idx])
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_risk[test_idx] = model.predict(X.iloc[test_idx])
        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "events_train": int(df["event"].iloc[train_idx].sum()),
            "events_test": int(df["event"].iloc[test_idx].sum()),
        })
    return oof_risk, fold_results


def evaluate_oof(df: pd.DataFrame, oof_risk: np.ndarray) -> dict[str, float]:
    """Evaluate survival discrimination over the observed follow-up."""
    y = make_survival_target(df)
    return {
        "cindex": float(concordance_index_censored(y["event"], y["time"], oof_risk)[0]),
        "ipcw_cindex": float(concordance_index_ipcw(y, y, oof_risk)[0]),
    }


def evaluate_models(df: pd.DataFrame, feature_sets: dict[str, list[str]]):
    metrics, predictions, fold_results = [], {}, {}
    for name, features in feature_sets.items():
        risk, folds = oof_predictions(df, features)
        result = evaluate_oof(df, risk)
        result.update(model=name, n_features=len(features))
        metrics.append(result)
        predictions[name] = risk
        fold_results[name] = folds
    metrics_df = pd.DataFrame(metrics).set_index("model")
    base, dawm = metrics_df.loc["base"], metrics_df.loc["regional_dawm"]
    delta = {
        "model": "delta_regional_dawm_minus_base",
        "cindex": dawm["cindex"] - base["cindex"],
        "ipcw_cindex": dawm["ipcw_cindex"] - base["ipcw_cindex"],
        "n_features": len(feature_sets["regional_dawm"]) - len(feature_sets["base"]),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([delta]).set_index("model")])
    return metrics_df, predictions, fold_results
