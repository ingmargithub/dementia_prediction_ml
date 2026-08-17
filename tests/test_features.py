import pandas as pd
from dementia_prediction.features import REGIONAL_DAWM, build_feature_sets, prepare_dataframe


def test_regional_dawm_construction():
    df = pd.DataFrame({
        "ExcludeDAWM": [0, 0], "COGSTATUS": ["normal", "normal"],
        "LFrontalDAWM": [1, 4], "RFrontalDAWM": [1, 4],
        "LTemporalDAWM": [2, 2], "RTemporalDAWM": [2, 2],
        "LParietalDAWM": [0, 1], "RParietalDAWM": [0, 1],
        "LOccipitalDAWM": [3, 3], "ROccipitalDAWM": [3, 3],
        "DEM_DEMENTIA": [0, 1], "DEM_TIME_IN_DAYS": [1000, 1500],
    })
    out = prepare_dataframe(df)
    assert out["FrontalDAWM"].tolist() == [1, 3]
    assert out["TemporalDAWM"].tolist() == [2, 2]
    assert out["ParietalDAWM"].tolist() == [0, 1]
    assert out["OccipitalDAWM"].tolist() == [3, 3]
    


def test_only_regional_dawm_is_added():
    df = pd.DataFrame({
        "AGE": [70, 72], "SEX": [0, 1], "EDUCATION": [2, 3], "BMI": [25, 27],
        "DEM_DEMENTIA": [0, 1], "DEM_TIME_IN_DAYS": [1000, 1500],
        **{column: [1, 2] for column in REGIONAL_DAWM},
    })
    features = build_feature_sets(df)
    assert set(features["regional_dawm"]) - set(features["base"]) == set(REGIONAL_DAWM)
    print(features)