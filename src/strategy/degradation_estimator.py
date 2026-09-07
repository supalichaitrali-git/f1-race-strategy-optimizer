"""
Empirical F1 tire degradation estimator.

Method:
1. Build a race/lap baseline independent of tire compound.
2. Calculate lap-time residuals relative to that baseline.
3. Estimate compound-specific degradation from tire age.
4. Apply isotonic regression to enforce a physically meaningful
   non-decreasing degradation curve.
5. Validate on completely unseen races.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score


DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/model_degradation_curve.csv"
)

COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]

MAX_TIRE_AGE = 40

TEST_RACE_FRACTION = 0.20


def load_data() -> pd.DataFrame:
    """Load clean dry-race data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df = df[
        df["Compound"].isin(COMPOUNDS)
    ].copy()

    df = df[
        df["TrackStatus"] == 1
    ].copy()

    df = df[
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= MAX_TIRE_AGE)
    ].copy()

    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    required_columns = [
        "Season",
        "GrandPrix",
        "Driver",
        "LapNumber",
        "LapTimeSec",
        "Compound",
        "TyreLife",
        "TrackTemp",
        "AirTemp",
        "Humidity",
    ]

    df = df.dropna(
        subset=required_columns
    ).copy()

    df["RaceKey"] = (
        df["Season"].astype(str)
        + "_"
        + df["GrandPrix"].astype(str)
    )

    return df


def create_race_lap_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate normal lap performance for each race/lap.

    The baseline deliberately does NOT condition on compound,
    because doing so would remove much of the tire-age signal.
    """

    result = df.copy()

    baseline = (
        result.groupby(
            ["RaceKey", "LapNumber"]
        )["LapTimeSec"]
        .median()
        .rename("RaceLapBaseline")
    )

    result = result.merge(
        baseline,
        on=["RaceKey", "LapNumber"],
        how="left",
    )

    result["Residual"] = (
        result["LapTimeSec"]
        - result["RaceLapBaseline"]
    )

    return result


def split_races(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by complete races so the test set contains
    races never seen during training.
    """

    races = (
        df["RaceKey"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(races) < 10:
        raise ValueError(
            "Not enough races for a reliable race-level split."
        )

    test_count = max(
        1,
        int(len(races) * TEST_RACE_FRACTION),
    )

    train_races = races[:-test_count]
    test_races = races[-test_count:]

    train = df[
        df["RaceKey"].isin(train_races)
    ].copy()

    test = df[
        df["RaceKey"].isin(test_races)
    ].copy()

    return train, test


def build_curve(
    train: pd.DataFrame,
    compound: str,
) -> pd.DataFrame:
    """
    Build a monotonic compound-specific degradation curve.

    The curve is based on the median residual at each tire age.
    Isotonic regression forces degradation to be non-decreasing.
    """

    compound_df = train[
        train["Compound"] == compound
    ].copy()

    if compound_df.empty:
        raise ValueError(
            f"No training data available for {compound}."
        )

    age_summary = (
        compound_df.groupby("TyreLife")[
            "Residual"
        ]
        .agg(
            median="median",
            count="count",
        )
        .reset_index()
    )

    age_summary = age_summary[
        age_summary["count"] >= 20
    ].copy()

    if len(age_summary) < 5:
        raise ValueError(
            f"Not enough tire-age observations for {compound}."
        )

    x = age_summary["TyreLife"].to_numpy()
    y = age_summary["median"].to_numpy()

    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )

    fitted = isotonic.fit_transform(
        x,
        y,
    )

    # Normalize the curve so the minimum observed degradation
    # is zero.
    fitted = fitted - fitted.min()

    curve_ages = np.arange(
        1,
        MAX_TIRE_AGE + 1,
    )

    curve_values = isotonic.predict(
        curve_ages
    )

    curve_values = (
        curve_values
        - curve_values.min()
    )

    curve = pd.DataFrame(
        {
            "Compound": compound,
            "TyreLife": curve_ages,
            "DegradationSec": curve_values,
        }
    )

    return curve


def predict_residual(
    df: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Predict tire degradation residuals."""

    predictions = np.zeros(
        len(df),
        dtype=float,
    )

    for compound in COMPOUNDS:

        mask = (
            df["Compound"] == compound
        )

        if not mask.any():
            continue

        curve = curves[compound]

        age_to_value = dict(
            zip(
                curve["TyreLife"],
                curve["DegradationSec"],
            )
        )

        predictions[mask] = (
            df.loc[mask, "TyreLife"]
            .map(age_to_value)
            .fillna(0.0)
            .to_numpy()
        )

    return predictions


def evaluate(
    test: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
) -> None:
    """Evaluate degradation estimates on unseen races."""

    actual = test["Residual"].to_numpy()

    predicted = predict_residual(
        test,
        curves,
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    print()
    print("=" * 70)
    print("UNSEEN-RACE DEGRADATION PERFORMANCE")
    print("=" * 70)

    print(
        f"R²  : {r2:.4f}"
    )

    print(
        f"MAE : {mae:.4f} sec"
    )

    print(
        f"Test laps: {len(test):,}"
    )

    print(
        f"Test races: "
        f"{test['RaceKey'].nunique():,}"
    )


def print_curves(
    curves: dict[str, pd.DataFrame],
) -> None:
    """Print estimated degradation curves."""

    print()
    print("=" * 70)
    print("EMPIRICAL TIRE DEGRADATION CURVES")
    print("=" * 70)

    selected_ages = [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ]

    for compound in COMPOUNDS:

        print()
        print("-" * 70)
        print(compound)
        print("-" * 70)

        curve = curves[compound]

        for age in selected_ages:

            value = curve.loc[
                curve["TyreLife"] == age,
                "DegradationSec",
            ]

            if value.empty:
                continue

            print(
                f"Age {age:>2}: "
                f"{value.iloc[0]:+.4f} sec"
            )


def save_curves(
    curves: dict[str, pd.DataFrame],
) -> None:
    """Save degradation curves."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = pd.concat(
        curves.values(),
        ignore_index=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved degradation curves: "
        f"{OUTPUT_FILE}"
    )


def main() -> None:
    """Run empirical degradation estimation."""

    print()
    print("=" * 70)
    print("F1 EMPIRICAL TIRE DEGRADATION ESTIMATOR")
    print("=" * 70)

    print()
    print("Loading data...")

    df = load_data()

    print(
        f"Usable laps: "
        f"{len(df):,}"
    )

    print(
        f"Races: "
        f"{df['RaceKey'].nunique():,}"
    )

    print()
    print(
        "Creating race/lap baseline..."
    )

    df = create_race_lap_baseline(
        df
    )

    print(
        "Splitting by complete races..."
    )

    train, test = split_races(
        df
    )

    print(
        f"Training laps: "
        f"{len(train):,}"
    )

    print(
        f"Testing laps: "
        f"{len(test):,}"
    )

    print(
        f"Training races: "
        f"{train['RaceKey'].nunique():,}"
    )

    print(
        f"Testing races: "
        f"{test['RaceKey'].nunique():,}"
    )

    curves = {}

    print()
    print(
        "Building monotonic degradation curves..."
    )

    for compound in COMPOUNDS:

        curves[compound] = build_curve(
            train,
            compound,
        )

    print_curves(
        curves
    )

    evaluate(
        test,
        curves,
    )

    save_curves(
        curves
    )

    print()
    print("=" * 70)
    print("DEGRADATION ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()