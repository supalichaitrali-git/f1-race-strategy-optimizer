"""
F1 tire degradation estimator - V3.

Estimates compound-specific tire degradation using
interpretable regression models and race-level validation.

Goal:
    Estimate how lap time changes as a tire gets older.

The model controls for:
- tire age
- race progression
- track temperature
- driver
- circuit
- season

SOFT, MEDIUM and HARD are modeled separately so that
each compound can have its own degradation rate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

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

TARGET = "LapTimeSec"


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and clean dry-tire race data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # Keep only dry compounds.
    df = df[
        df["Compound"].isin(COMPOUNDS)
    ].copy()

    # Keep green-flag racing laps.
    df = df[
        df["TrackStatus"] == 1
    ].copy()

    # Restrict tire age to a reasonable modeling range.
    df = df[
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= 40)
    ].copy()

    # Remove clearly abnormal lap times.
    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    required_columns = [
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
        subset=required_columns
    )

    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# Race-level train/test split
# -------------------------------------------------------------------

def split_by_race(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split complete races into training and testing sets.

    No laps from a test race are allowed into training.
    """

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

    race_keys = list(
        zip(
            df["Season"],
            df["GrandPrix"],
        )
    )

    test_mask = pd.Series(
        [
            race in test_keys
            for race in race_keys
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
# Feature preparation
# -------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create interpretable tire-age features.

    TyreLifeSquared allows a curved degradation relationship.
    """

    result = df.copy()

    result["TyreLifeSquared"] = (
        result["TyreLife"] ** 2
    )

    return result


# -------------------------------------------------------------------
# Build one compound model
# -------------------------------------------------------------------

def build_compound_model() -> Pipeline:
    """
    Build a regression model for one tire compound.

    Numerical:
        TyreLife
        TyreLifeSquared
        LapNumber
        TrackTemp

    Categorical:
        Driver
        GrandPrix
        Season
    """

    numerical_features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "TrackTemp",
    ]

    categorical_features = [
        "Driver",
        "GrandPrix",
        "Season",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                Ridge(alpha=10.0),
            ),
        ]
    )

    return model


# -------------------------------------------------------------------
# Train compound models
# -------------------------------------------------------------------

def train_models(
    train_df: pd.DataFrame,
) -> dict[str, Pipeline]:
    """Train a separate model for each tire compound."""

    models = {}

    train_df = prepare_features(
        train_df
    )

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    for compound in COMPOUNDS:

        compound_df = train_df[
            train_df["Compound"] == compound
        ].copy()

        X = compound_df[
            features
        ]

        y = compound_df[
            TARGET
        ]

        model = build_compound_model()

        model.fit(
            X,
            y,
        )

        models[compound] = model

        print(
            f"  {compound}: "
            f"{len(compound_df):,} training laps"
        )

    return models


# -------------------------------------------------------------------
# Evaluate compound models
# -------------------------------------------------------------------

def evaluate_models(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
) -> None:
    """Evaluate each compound model."""

    df = prepare_features(
        df
    )

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    print()
    print("=" * 70)
    print("UNSEEN-RACE MODEL PERFORMANCE")
    print("=" * 70)

    for compound in COMPOUNDS:

        compound_df = df[
            df["Compound"] == compound
        ].copy()

        model = models[
            compound
        ]

        X = compound_df[
            features
        ]

        y = compound_df[
            TARGET
        ]

        predictions = model.predict(
            X
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
            f"  Laps: {len(compound_df):,}"
        )


# -------------------------------------------------------------------
# Estimate degradation curves
# -------------------------------------------------------------------

def estimate_curves(
    models: dict[str, Pipeline],
    train_df: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Estimate tire degradation from age 1 to 40.

    Conditions such as track temperature, driver,
    circuit and season are held constant.

    Degradation is measured relative to age 1.
    """

    train_df = prepare_features(
        train_df
    )

    base_row = {
        "LapNumber": train_df["LapNumber"].median(),
        "TrackTemp": train_df["TrackTemp"].median(),
        "Driver": train_df["Driver"].mode()[0],
        "GrandPrix": train_df["GrandPrix"].mode()[0],
        "Season": int(
            train_df["Season"].median()
        ),
    }

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    curves = {}

    for compound in COMPOUNDS:

        rows = []

        for age in range(1, 41):

            row = base_row.copy()

            row["TyreLife"] = age

            row["TyreLifeSquared"] = (
                age ** 2
            )

            rows.append(row)

        prediction_df = pd.DataFrame(
            rows
        )

        predictions = models[
            compound
        ].predict(
            prediction_df[
                features
            ]
        )

        baseline = predictions[0]

        degradation = (
            predictions - baseline
        )

        # Make the curve non-decreasing.
        #
        # This prevents a statistical model from
        # predicting that an older tire is faster
        # than a younger tire.
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
    """Print selected points from degradation curves."""

    print()
    print("=" * 70)
    print("ESTIMATED TIRE DEGRADATION CURVES")
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
    """Save degradation curves as CSV."""

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
                        curves[compound][age - 1]
                    ),
                }
            )

    curve_df = pd.DataFrame(
        rows
    )

    curve_df.to_csv(
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
    """Run the complete V3 degradation analysis."""

    print()
    print("=" * 70)
    print("F1 TIRE DEGRADATION ESTIMATOR - V3")
    print("=" * 70)

    print()
    print("Loading data...")

    df = load_data()

    print(
        f"Usable laps: {len(df):,}"
    )

    print(
        "Races: "
        f"{df[['Season', 'GrandPrix']].drop_duplicates().shape[0]}"
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
    print("Training separate compound models...")

    models = train_models(
        train_df
    )

    evaluate_models(
        models,
        test_df,
    )

    print()
    print("=" * 70)
    print("ESTIMATING CONTROLLED DEGRADATION")
    print("=" * 70)

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
    print("V3 DEGRADATION ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()