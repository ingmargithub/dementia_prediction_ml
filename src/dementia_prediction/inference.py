from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2

from .model import make_cox_pipeline, make_survival_target


@dataclass(frozen=True)
class CoxComparison:
    log_likelihood_base: float
    log_likelihood_dawm: float
    likelihood_ratio: float
    degrees_of_freedom: int
    p_value: float


def _partial_log_likelihood(
    model,
    X: pd.DataFrame,
    y,
) -> float:
    """Breslow-style Cox partial log-likelihood for the fitted linear predictor."""

    risk_score = model.predict(X)

    time = y["time"]
    event = y["event"]

    order = np.argsort(time)

    time = time[order]
    event = event[order]
    risk_score = risk_score[order]

    log_likelihood = 0.0

    for event_time in np.unique(time[event]):

        event_mask = (
            (time == event_time)
            & event
        )

        risk_set = time >= event_time

        n_events = int(event_mask.sum())

        log_likelihood += risk_score[event_mask].sum()

        # use logsumexp rather than exp to avoid overflow
        log_risk_set = logsumexp(
            risk_score[risk_set]
        )

        log_likelihood -= (
            n_events * log_risk_set
        )

    return float(log_likelihood)


def compare_final_models(
    df: pd.DataFrame,
    base_features: list[str],
    dawm_features: list[str],
) -> CoxComparison:
    """compare the final base and base + DAWM models on LL and LR"""

    y = make_survival_target(df)

    # make and fit the two final models
    base_model = make_cox_pipeline(
        df[base_features]
    )

    dawm_model = make_cox_pipeline(
        df[dawm_features]
    )

    base_model.fit(
        df[base_features],
        y,
    )

    dawm_model.fit(
        df[dawm_features],
        y,
    )

    # calculate partial log-likelihood
    ll_base = _partial_log_likelihood(
        base_model,
        df[base_features],
        y,
    )

    ll_dawm = _partial_log_likelihood(
        dawm_model,
        df[dawm_features],
        y,
    )

    # likelihood ratio
    lr = 2.0 * (
        ll_dawm - ll_base
    )

    # number of fitted coefficients
    base_n_params = len(
        base_model
        .named_steps["cox"]
        .coef_
    )

    dawm_n_params = len(
        dawm_model
        .named_steps["cox"]
        .coef_
    )

    df_diff = (
        dawm_n_params
        - base_n_params
    )

    # test the actual difference using chi-square
    pval = chi2.sf(
        lr,
        df_diff,
    )

    # return the data class
    return CoxComparison(
        ll_base,
        ll_dawm,
        lr,
        df_diff,
        pval,
    )