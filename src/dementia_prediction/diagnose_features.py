import numpy as np
import pandas as pd


def diagnose_features(df, feature_sets):

    for name, features in feature_sets.items():

        X = df[features].copy()

        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        # ---------------------------------------------------------
        # 1. Missing / infinite values
        # ---------------------------------------------------------

        numeric = X.select_dtypes(include=np.number)

        n_nan = numeric.isna().sum()
        n_inf = np.isinf(numeric).sum()

        print("\nMissing values:")
        print(n_nan[n_nan > 0].sort_values(ascending=False))

        print("\nInfinite values:")
        print(n_inf[n_inf > 0].sort_values(ascending=False))

        # ---------------------------------------------------------
        # 2. Number of unique values
        # ---------------------------------------------------------

        summary = pd.DataFrame({
            "n_unique": X.nunique(dropna=True),
            "n_missing": X.isna().sum(),
        })

        summary["pct_zero"] = np.nan

        for col in numeric.columns:
            summary.loc[col, "pct_zero"] = (
                (numeric[col] == 0).mean() * 100
            )

        print("\nLowest-variation variables:")
        print(
            summary
            .sort_values("n_unique")
            .head(20)
            .to_string()
        )

        # ---------------------------------------------------------
        # 3. Near-zero variance
        # ---------------------------------------------------------

        print("\nVariables >95% zero:")

        zero_heavy = summary[
            summary["pct_zero"] > 95
        ]

        if len(zero_heavy):
            print(zero_heavy.to_string())
        else:
            print("None")

        # ---------------------------------------------------------
        # 4. Correlations
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # 5. Matrix rank
        # ---------------------------------------------------------

        X_numeric = numeric.copy()

        X_numeric = X_numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_numeric = X_numeric.fillna(
            X_numeric.median()
        )

        matrix = X_numeric.to_numpy(dtype=float)

        rank = np.linalg.matrix_rank(matrix)
        n_columns = matrix.shape[1]

        print("\nMatrix:")
        print("  rows:    ", matrix.shape[0])
        print("  columns: ", n_columns)
        print("  rank:    ", rank)

        if rank < n_columns:
            print(
                "WARNING: matrix is rank deficient."
            )
        else:
            print(
                "Matrix is full rank."
            )