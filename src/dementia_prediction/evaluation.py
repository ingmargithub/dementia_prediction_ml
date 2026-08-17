from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
)

from .config import N_OUTER, RANDOM_STATE
from .model import make_survival_target


def _folds(df: pd.DataFrame):
    """generate stratified outer CV folds based on dementia event status"""

    splitter = StratifiedKFold(
        n_splits=N_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return splitter.split(
        df,
        df["event"].astype(int),
    )


def make_cox_pipeline() -> Pipeline:
    """
    Create the preprocessing + Cox PH pipeline. Imputation and scaling are 
    fitted separately within each fold 
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "cox",
            CoxPHSurvivalAnalysis(
                alpha=0.0,
            ),
        ),
    ])


def oof_predictions(
    df: pd.DataFrame,
    features: list[str],
):
    """
    Generate out-of-fold Cox relative-risk scores. Missing values are 
    median-imputed using only the training observations in each fold. Scaling 
    is also fitted per fold
    """

    X = df[features].copy()
    y = make_survival_target(df)

    print(f"X = {len(X)}, y = {len(y)}")

    # Initialize OOF predictions.
    oof_risk = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        _folds(df),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"events_train={int(y_train['event'].sum())}, "
            f"events_test={int(y_test['event'].sum())}"
        )
       
        # make and fit model
        model = make_cox_pipeline()
        model.fit(
            X_train,
            y_train,
        )

        # Prediction on the completely held-out fold.
        risk = model.predict(X_test)
        oof_risk[test_idx] = risk
        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "events_train": int(
                y_train["event"].sum()
            ),
            "events_test": int(
                y_test["event"].sum()
            ),
        })

    if np.isnan(oof_risk).any():
        raise RuntimeError(
            "Some observations do not have an out-of-fold prediction."
        )

    return oof_risk, fold_results


def evaluate_oof(
    df: pd.DataFrame,
    oof_risk: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate out-of-fold survival discrimination over the complete follow-up period.
    """

    y = make_survival_target(df)

    return {
        "cindex": float(
            concordance_index_censored(
                y["event"],
                y["time"],
                oof_risk,
            )[0]
        ),
        "ipcw_cindex": float(
            concordance_index_ipcw(
                y,
                y,
                oof_risk,
            )[0]
        ),
    }


def evaluate_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
):
    """
    Compare the base and regional-DAWM Cox models using identical outer CV folds.
    """

    metrics = []
    predictions = {}
    fold_results = {}

    for name, features in feature_sets.items():

        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        risk, folds = oof_predictions(
            df,
            features,
        )
        result = evaluate_oof(
            df,
            risk,
        )

        result.update({
            "model": name,
            "n_features": len(features),
        })

        metrics.append(result)
        predictions[name] = risk
        fold_results[name] = folds

    metrics_df = (
        pd.DataFrame(metrics)
        .set_index("model")
    )

    base = metrics_df.loc["base"]
    dawm = metrics_df.loc["regional_dawm"]

    delta = {
        "model": "delta_regional_dawm_minus_base",
        "cindex": (
            dawm["cindex"]
            - base["cindex"]
        ),
        "ipcw_cindex": (
            dawm["ipcw_cindex"]
            - base["ipcw_cindex"]
        ),
        "n_features": (
            len(feature_sets["regional_dawm"])
            - len(feature_sets["base"])
        ),
    }

    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([delta])
        .set_index("model"),
    ])

    return (
        metrics_df,
        predictions,
        fold_results,
    )