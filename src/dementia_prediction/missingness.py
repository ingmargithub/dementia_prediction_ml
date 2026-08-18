import numpy as np
import pandas as pd
from .model import make_preprocessor

def check_missingness(df: pd.DataFrame, features: list[str],
                      ) -> dict[str, pd.DataFrame]:
    """
    Report missing values, infinities, and zero prevalence for each
    feature in each model feature set. 
    """

    X = df[features].copy()

    # Replace Inf with NaN so they are handled consistently.
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = X[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    summary = pd.DataFrame(index=features)
    summary["dtype"] = X.dtypes
    summary["n"] = len(X)

    # Missingness
    summary["n_missing"] = X.isna().sum()
    summary["pct_missing"] = (
        X.isna().mean() * 100
    )

    # Unique values
    summary["n_unique"] = X.nunique(dropna=True)

    # Zero prevalence for numeric variables
    summary["n_zero"] = np.nan
    summary["pct_zero"] = np.nan
    for col in numeric_cols:
        summary.loc[col, "n_zero"] = (X[col] == 0).sum()
        summary.loc[col, "pct_zero"] = (
            (X[col] == 0).mean() * 100
        )

    print(f"Observations: {len(X)}")
    print(f"Features:     {len(features)}")
    print(f"Total missing cells: "
          f"{int(X.isna().sum().sum())}"
    )

    print("\nFeatures with missing values:")
    missing_features = summary[
        summary["n_missing"] > 0
    ]
    if len(missing_features) == 0:
        print("None")
    else:
        print(
            missing_features[
                [
                    "n_missing",
                    "pct_missing",
                    "n_unique",
                    "n_zero",
                    "pct_zero",
                ]
            ].to_string()
        )

    print("\nFeatures with >75% zeros:")
    zero_heavy = summary[
        summary["pct_zero"] > 75
    ]
    if len(zero_heavy) == 0:
        print("None")
    else:
        print(
            zero_heavy[
                [
                    "n_zero",
                    "pct_zero",
                    "n_unique",
                ]
            ].to_string()
        )

    print("\nFeatures with only one observed value:")
    constant = summary[
        summary["n_unique"] <= 1
    ]
    if len(constant) == 0:
        print("None")
    else:
        print(
            constant[
                [
                    "n_unique",
                    "n_missing",
                ]
            ].to_string()
        )

def diagnose_features(df, features):
    """Further diagnostics"""

    X = df[features].copy()

    numeric = X.select_dtypes(include=np.number)
    corr = numeric.corr()

    pairs = []
    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            r = corr.iloc[i, j]

            if abs(r) >= 0.90:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": r,
                })

    print("\nCorrelations >= |0.90|:")

    if pairs:
        print(
            pd.DataFrame(pairs)
            .sort_values(
                "correlation",
                key=lambda x: abs(x),
                ascending=False,
            )
            .to_string(index=False)
        )
    else:
        print("None")

    # re-use encoding etc.
    X_numeric = numeric.copy()
    preprocessor = make_preprocessor(X_numeric)
    X_numeric = preprocessor.fit_transform(X_numeric)
    
    # check for rank deficiency
    matrix = np.asarray(X_numeric, dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    n_columns = matrix.shape[1]

    print("\nMatrix:")
    print("  rows:    ", matrix.shape[0])
    print("  columns: ", n_columns)
    print("  rank:    ", rank)

    if rank < n_columns:
        print(            "WARNING: matrix is rank deficient."        )
    else:
        print(            "Matrix is full rank."        )