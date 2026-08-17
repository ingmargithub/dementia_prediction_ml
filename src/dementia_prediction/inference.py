from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2
from sksurv.util import Surv

from .model import make_cox_pipeline


@dataclass(frozen=True)
class CoxComparison:
    log_likelihood_base: float
    log_likelihood_dawm: float
    likelihood_ratio: float
    degrees_of_freedom: int
    p_value: float


def _partial_log_likelihood(model, X: pd.DataFrame, y) -> float:
    """Breslow-style Cox partial log-likelihood for the fitted linear predictor."""
    risk_score = model.predict(X)
    time = y["time"]
    event = y["event"]
    order = time.argsort()
    time, event, risk_score = time[order], event[order], risk_score[order]
    log_likelihood = 0.0
    for event_time in pd.unique(time[event]):
        event_mask = (time == event_time) & event
        risk_set = time >= event_time
        n_events = int(event_mask.sum())
        log_likelihood += risk_score[event_mask].sum()
        log_likelihood -= n_events * __import__("numpy").log(
            __import__("numpy").exp(risk_score[risk_set]).sum()
        )
    return float(log_likelihood)


def compare_final_models(df: pd.DataFrame, base_features: list[str], dawm_features: list[str]) -> CoxComparison:
    y = Surv.from_arrays(df["event"].to_numpy(bool), df["DEM_TIME_IN_DAYS"].to_numpy(float))
    base_model = make_cox_pipeline(df[base_features])
    dawm_model = make_cox_pipeline(df[dawm_features])
    base_model.fit(df[base_features], y)
    dawm_model.fit(df[dawm_features], y)
    ll_base = _partial_log_likelihood(base_model, df[base_features], y)
    ll_dawm = _partial_log_likelihood(dawm_model, df[dawm_features], y)
    lr = 2.0 * (ll_dawm - ll_base)
    df_diff = len(dawm_features) - len(base_features)
    return CoxComparison(ll_base, ll_dawm, lr, df_diff, float(chi2.sf(lr, df_diff)))
