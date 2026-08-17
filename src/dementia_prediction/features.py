from __future__ import annotations

import numpy as np
import pandas as pd

DEMOGRAPHICS = ["AGE", "SEX", "EDUCATION", "WEIGHT", "HEIGHT"] # if we add weight & height then BMI is not needed
VASCULAR = ["HTN", "DM2", "CAD_01"] # removed afib/flutter features
LIFESTYLE = ["SMOKINGSTATUS", "SMOKING_YEARS", "ALCOHOLGWEEK"] # removed phys and smoking_ever
SVD_MARKERS = ["MICROBLD", "SUBCORTYN", "VRSYN", "INFCORTYN", "INFCERYN"] #removed hematoma
WMH_LOBAR = ["OccipitalP", "ParietalP", "TemporalP", "FrontalP"] # only use periventricular WMH..?
BASELINE_COGNITION = ["SA1MEMORY2", "SA1WORKING2", "SA1SPEED2"]
REGIONAL_DAWM = ["FrontalDAWM", "ParietalDAWM", "OccipitalDAWM"] # removed "TemporalDAWM" for now because of very low count

LEAKAGE_PREFIXES = (
    "DEM_", "DEMENTIA", "COGSTATUS", "SA2", "DAYS_AGES_AGESII",
)

CATEGORICAL_FEATURES = {
    "SEX", "HTN", "DM2", "CAD_01", "SMOKINGSTATUS", "MICROBLD", "SUBCORTYN", 
    "VRSYN", "INFCORTYN", "INFCERYN", "EDUCATION",
}


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply prespecified exclusions and construct baseline-derived variables."""
    df = df.copy()

    if "ExcludeDAWM" in df:
        df = df.loc[df["ExcludeDAWM"] != 1].copy()
    if "COGSTATUS" in df:
        df = df.loc[df["COGSTATUS"].notna()].copy()
        df = df.loc[df["COGSTATUS"] != "dem"].copy()

    pairs = {
        "FrontalDAWM": ("LFrontalDAWM", "RFrontalDAWM"),
        "ParietalDAWM": ("LParietalDAWM", "RParietalDAWM"),
        "OccipitalDAWM": ("LOccipitalDAWM", "ROccipitalDAWM"),
        #"TemporalDAWM": ("LTemporalDAWM", "RTemporalDAWM"),
    }
    
    # make DAWM bilateral
    for new, (left, right) in pairs.items():
        if left in df and right in df:
            bilateral = pd.to_numeric(df[left], errors="coerce") + pd.to_numeric(df[right], errors="coerce")
            df[new] = np.rint(bilateral / 2.0).replace(4, 3)

    # continuous smoking years is better than categorical smoking status
    if {"C_YEAR", "P_YEAR", "CH_YEAR"}.issubset(df.columns):
        df["SMOKING_YEARS"] = df[["C_YEAR", "P_YEAR", "CH_YEAR"]].fillna(0).sum(axis=1)

    # ICV is a normalizing factor in linear models, perhaps helps to adjust this model too
    if "baselineICV" not in df and {"BPP01GM_VOL", "BPP01WM_VOL", "BPP01CSF_VOL"}.issubset(df.columns):
        df["baselineICV"] = df[["BPP01GM_VOL", "BPP01WM_VOL", "BPP01CSF_VOL"]].sum(axis=1)

    # do log-transforms as usual in log-linear models
    for column in WMH_LOBAR:
        if column in df:
            df[f"{column}_log"] = np.log1p(pd.to_numeric(df[column], errors="coerce"))

    # needed for Cox PH setup
    required = ["DEM_DEMENTIA", "DEM_TIME_IN_DAYS"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required outcome columns: {missing}")
    df["DEM_DEMENTIA"] = pd.to_numeric(df["DEM_DEMENTIA"], errors="coerce")
    df["DEM_TIME_IN_DAYS"] = pd.to_numeric(df["DEM_TIME_IN_DAYS"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df.loc[df["DEM_TIME_IN_DAYS"] > 0].copy()
    df["event"] = df["DEM_DEMENTIA"].astype(bool)
    return df


def _available(columns: set[str], candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in columns]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build the one prespecified base model and its regional-DAWM extension."""
    columns = set(df.columns)
    wmh = _available(columns, [f"{c}_log" for c in WMH_LOBAR])
    if not wmh:
        wmh = _available(columns, ["BPP01WML_VOL", "A2_B_BPP01WML_VOL"])

    base = (
        _available(columns, DEMOGRAPHICS)
        + _available(columns, VASCULAR)
        + _available(columns, LIFESTYLE)
        + _available(columns, SVD_MARKERS)
        + wmh
        + _available(columns, ["baselineICV"])
        + _available(columns, BASELINE_COGNITION)
    )
    base = list(dict.fromkeys(base))
    base = [
        c for c in base
        if c not in {"event", "DEM_DEMENTIA", "DEM_TIME_IN_DAYS"}
        and not any(c.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
    ]

    dawm = _available(columns, REGIONAL_DAWM)
    if set(dawm) != set(REGIONAL_DAWM):
        raise ValueError(f"Regional DAWM model requires all four variables; missing: {sorted(set(REGIONAL_DAWM) - set(dawm))}")

    return {"base": base, "regional_dawm": base + dawm}
