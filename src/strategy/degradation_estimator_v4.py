"""
F1 tire degradation estimator - V4.

Uses stint-relative lap time to reduce the influence of:
- driver pace
- circuit characteristics
- overall race pace
- fuel load progression

The degradation signal is estimated from the change in lap
performance relative to the driver's own stint baseline.

Validation is performed at race level.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


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


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and clean dry-race lap data."""

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
        & (df["TyreLife"] <= 40)
    ].copy()

    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    required = [
        "LapTimeSec",
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    df = df.dropna(
        subset=required
    )

    df = df.sort_values(
        [
            "Season",
            "GrandPrix",
            "Driver",
            "LapNumber",
        ]
    ).reset_index(drop=True)

    return df


# -------------------------------------------------------------------
# Identify stints
# -------------------------------------------------------------------

def create_stint_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a unique stint identifier.

    A new stint starts when:
    - the tire compound changes, or
    - tire age resets.
    """

    result = df.copy()

    result["NewStint"] = (
        (
            result["Compound"]
            != result.groupby(
                ["Season", "GrandPrix", "Driver"]
            )["Compound"].shift()
        )
        |
        (
            result["TyreLife"]
            < result.groupby(
                ["Season", "GrandPrix", "Driver"]
            )["TyreLife"].shift()
        )
    )

    result["NewStint"] = (
        result["NewStint"]
        .fillna(True)
    )

    result["StintNumber"] = (
        result.groupby(
            [
                "Season",
                "GrandPrix",
                "Driver",
            ]
        )["NewStint"]
        .cumsum()
    )

    result["StintID"] = (
        result["Season"].astype(str)
        + "_"
        + result["GrandPrix"].astype(str)
        + "_"
        + result["Driver"].astype(str)
        + "_"
        + result["StintNumber"].astype(str)
    )

    return result


# -------------------------------------------------------------------
# Create relative lap time
# -------------------------------------------------------------------

def create_relative_lap_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize lap times within each driver/race/stint.

    The median lap time of the stint is used as the baseline.
    """

    result = df.copy()

    stint_median = (
        result.groupby(
            "StintID"
        )["LapTimeSec"]
        .transform("median")
    )

    result["RelativeLapTime"] = (
        result["LapTimeSec"]
        - stint_median
    )

    return result


# -------------------------------------------------------------------
# Prepare data
# -------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create stint-normalized modeling data."""

    df = create_stint_ids(
        df
    )

    df = create_relative_lap_time(
        df
    )

    # Remove very short stints.
    stint_sizes = (
        df.groupby(
            "StintID"
        ).size()
    )

    valid_stints = stint_sizes[
        stint_sizes >= 5
    ].index

    df = df[
        df["StintID"].isin(
            valid_stints
        )
    ].copy()

    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# Race-level split
# -------------------------------------------------------------------

def split_by_race(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete races into train and test sets."""

    races = (
        df[
            ["Season", "GrandPrix"]
        ]
        .drop_duplicates()
        .sort_values(
            ["Season", "GrandPrix"]
        )
        .reset_index(drop=True)
    )

    test_count = max(
        1,
        int(len(races) * 0.20),
    )

    test_races = races.tail(
        test_count
    )

    test_keys = set(
        zip(
            test_races["Season"],
            test_races["GrandPrix"],
        )
    )

    keys = list(
        zip(
            df["Season"],
            df["GrandPrix"],
        )
    )

    test_mask = pd.Series(
        [
            key in test_keys
            for key in keys
        ],
        index=df.index,
    )

    train_df = df[
        ~test_mask
    ].copy()

    test_df = df[
        test_mask
    ].copy()

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# -------------------------------------------------------------------
# Train one compound model
# -------------------------------------------------------------------

def train_compound_model(
    train_df: pd.DataFrame,
    compound: str,
):
    """Train a degradation model for one compound."""

    data = train_df[
        train_df["Compound"] == compound
    ].copy()

    X = data[
        [
            "TyreLife",
            "LapNumber",
            "TrackTemp",
        ]
    ]

    y = data[
        "RelativeLapTime"
    ]

    polynomial = PolynomialFeatures(
        degree=2,
        include_bias=False,
    )

    X_poly = polynomial.fit_transform(
        X
    )

    model = Ridge(
        alpha=10.0
    )

    model.fit(
        X_poly,
        y
    )

    return (
        polynomial,
        model,
    )


# -------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------

def evaluate_models(
    models,
    test_df: pd.DataFrame,
) -> None:
    """Evaluate models on completely unseen races."""

    print()
    print("=" * 70)
    print("UNSEEN-RACE PERFORMANCE")
    print("=" * 70)

    for compound in COMPOUNDS:

        data = test_df[
            test_df["Compound"] == compound
        ].copy()

        if data.empty:
            continue

        polynomial, model = models[
            compound
        ]

        X = data[
            [
                "TyreLife",
                "LapNumber",
                "TrackTemp",
            ]
        ]

        y = data[
            "RelativeLapTime"
        ]

        predictions = model.predict(
            polynomial.transform(
                X
            )
        )

        r2 = r2_score(
            y,
            predictions,
        )

        mae = mean_absolute_error(
            y,
            predictions,
        )

        print()
        print(
            f"{compound}:"
        )

        print(
            f"  R²  : {r2:.4f}"
        )

        print(
            f"  MAE : {mae:.4f} sec"
        )

        print(
            f"  Laps: {len(data):,}"
        )


# -------------------------------------------------------------------
# Estimate curves
# -------------------------------------------------------------------

def estimate_curves(
    models,
    train_df: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Estimate degradation from tire age 1 to 40.

    Lap number and track temperature are held constant.
    """

    median_lap = train_df[
        "LapNumber"
    ].median()

    median_temp = train_df[
        "TrackTemp"
    ].median()

    curves = {}

    for compound in COMPOUNDS:

        polynomial, model = models[
            compound
        ]

        rows = []

        for age in range(1, 41):

            rows.append(
                {
                    "TyreLife": age,
                    "LapNumber": median_lap,
                    "TrackTemp": median_temp,
                }
            )

        curve_df = pd.DataFrame(
            rows
        )

        predictions = model.predict(
            polynomial.transform(
                curve_df[
                    [
                        "TyreLife",
                        "LapNumber",
                        "TrackTemp",
                    ]
                ]
            )
        )

        baseline = predictions[0]

        degradation = (
            predictions - baseline
        )

        # Enforce physically meaningful behavior.
        degradation = np.maximum.accumulate(
            degradation
        )

        curves[compound] = (
            degradation.tolist()
        )

    return curves


# -------------------------------------------------------------------
# Print curves
# -------------------------------------------------------------------

def print_curves(
    curves: dict[str, list[float]],
) -> None:
    """Print degradation curves."""

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

    print()
    print("=" * 70)
    print("STINT-RELATIVE TIRE DEGRADATION")
    print("=" * 70)

    for compound in COMPOUNDS:

        print()
        print("-" * 70)
        print(compound)
        print("-" * 70)

        print(
            f"{'Tire Age':<12}"
            f"{'Degradation (sec)':>25}"
        )

        for age in selected_ages:

            value = curves[
                compound
            ][age - 1]

            print(
                f"{age:<12}"
                f"{value:>25.4f}"
            )


# -------------------------------------------------------------------
# Save curves
# -------------------------------------------------------------------

def save_curves(
    curves: dict[str, list[float]],
) -> None:
    """Save degradation curves."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for compound in COMPOUNDS:

        for age in range(1, 41):

            rows.append(
                {
                    "Compound": compound,
                    "TyreLife": age,
                    "EstimatedDegradationSec": (
                        curves[
                            compound
                        ][age - 1]
                    ),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved degradation curves: "
        f"{OUTPUT_FILE}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run V4 degradation analysis."""

    print()
    print("=" * 70)
    print("F1 TIRE DEGRADATION ESTIMATOR - V4")
    print("=" * 70)

    print()
    print("Loading data...")

    df = load_data()

    print(
        f"Raw usable laps: {len(df):,}"
    )

    print()
    print("Creating stint-relative lap times...")

    df = prepare_data(
        df
    )

    print(
        f"Usable stint-normalized laps: "
        f"{len(df):,}"
    )

    print(
        f"Unique stints: "
        f"{df['StintID'].nunique():,}"
    )

    print()
    print("Creating race-level train/test split...")

    train_df, test_df = split_by_race(
        df
    )

    print(
        f"Training laps: {len(train_df):,}"
    )

    print(
        f"Testing laps:  {len(test_df):,}"
    )

    print()
    print("Training compound-specific models...")

    models = {}

    for compound in COMPOUNDS:

        polynomial, model = (
            train_compound_model(
                train_df,
                compound,
            )
        )

        models[compound] = (
            polynomial,
            model,
        )

        count = len(
            train_df[
                train_df["Compound"]
                == compound
            ]
        )

        print(
            f"  {compound}: "
            f"{count:,} training laps"
        )

    evaluate_models(
        models,
        test_df,
    )

    print()
    print("Estimating controlled degradation curves...")

    curves = estimate_curves(
        models,
        train_df,
    )

    print_curves(
        curves
    )

    save_curves(
        curves
    )

    print()
    print("=" * 70)
    print("V4 DEGRADATION ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()