import numpy as np
import pandas as pd


def check_missingness(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    Report missing values, infinities, and zero prevalence for each
    feature in each model feature set. Returns a missingness summary df
    """

    summaries = {}

    for name, features in feature_sets.items():

        X = df[features].copy()

        # Replace infinities with NaN so they are handled consistently.
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

        summaries[name] = summary.sort_values(
            "pct_missing",
            ascending=False,
        )

        print(f"\n{'=' * 80}")
        print(name)
        print(f"{'=' * 80}")

        print(f"Observations: {len(X)}")
        print(f"Features:     {len(features)}")
        print(
            f"Total missing cells: "
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

    return summaries


def median_impute(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Replace +/- infinity with NaN first and median-impute the 
    numeric predictors that were NaNs.
    """

    df = df.copy()

    all_features = sorted(
        set(
            feature
            for features in feature_sets.values()
            for feature in features
        )
    )

    numeric_features = df[all_features].select_dtypes(
        include=np.number
    ).columns

    # treat infinities as missing
    df[numeric_features] = df[numeric_features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print("\nMedian imputation")
    print("=" * 80)

    for col in numeric_features:

        n_missing = df[col].isna().sum()

        if n_missing == 0:
            continue

        median = df[col].median()

        print(
            f"{col}: "
            f"{n_missing} missing "
            f"({100 * n_missing / len(df):.2f}%) "
            f"-> median = {median:.4g}"
        )

        df[col] = df[col].fillna(median)

    return df