from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from .features import CATEGORICAL_FEATURES


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    
    # check type
    categorical = [c for c in X.columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in X.columns if c not in categorical]
       
    # which pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    # apply pipeline to the columns
    return ColumnTransformer([ 
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")


def make_survival_target(df: pd.DataFrame) -> np.ndarray:
    """Sets the target (y) which is the 1/0 dementia incidence + its time 
    in days to its occurence"""
    return Surv.from_arrays(
        event=df["event"].to_numpy(dtype=bool),
        time=df["DEM_TIME_IN_DAYS"].to_numpy(dtype=float),
    )


def make_cox_pipeline(X: pd.DataFrame) -> Pipeline:
    """Primary unpenalized Cox proportional-hazards model."""
    
    # convert inf to nan first
    np.where(np.isinf(X), np.nan, X)
    
    return Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("cox", CoxPHSurvivalAnalysis(alpha=0.0)),
    ])