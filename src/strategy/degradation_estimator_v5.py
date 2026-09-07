"""
F1 tire degradation estimator - V5.

Models lap time using:
- tire age
- race progression
- track temperature
- air temperature
- humidity
- driver
- circuit
- season
- compound-specific tire-age interactions

The model is validated using complete unseen races.

V5 uses early-stint laps as a fresh-tire reference when
estimating degradation curves.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


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

    required_columns = [
        "LapTimeSec",
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    df = df.dropna(
        subset=required_columns
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
# Create stint IDs
# -------------------------------------------------------------------

def create_stint_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify tire stints for each driver.

    A new stint starts when:
    - compound changes
    - tire age decreases
    """

    result = df.copy()

    group_columns = [
        "Season",
        "GrandPrix",
        "Driver",
    ]

    previous_compound = (
        result.groupby(group_columns)[
            "Compound"
        ].shift()
    )

    previous_age = (
        result.groupby(group_columns)[
            "TyreLife"
        ].shift()
    )

    result["NewStint"] = (
        result["Compound"]
        != previous_compound
    ) | (
        result["TyreLife"]
        < previous_age
    )

    result["NewStint"] = (
        result["NewStint"]
        .fillna(True)
    )

    result["StintNumber"] = (
        result.groupby(
            group_columns
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
# Feature engineering
# -------------------------------------------------------------------

def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create nonlinear and compound interaction features."""

    result = df.copy()

    result["TyreLifeSquared"] = (
        result["TyreLife"] ** 2
    )

    result["LapNumberSquared"] = (
        result["LapNumber"] ** 2
    )

    # Explicit compound × tire-age interactions.
    for compound in COMPOUNDS:

        result[
            f"TyreLife_{compound}"
        ] = (
            result["TyreLife"]
            * (
                result["Compound"]
                == compound
            ).astype(int)
        )

        result[
            f"TyreLifeSquared_{compound}"
        ] = (
            result["TyreLifeSquared"]
            * (
                result["Compound"]
                == compound
            ).astype(int)
        )

    return result


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
# Build model
# -------------------------------------------------------------------

def build_model() -> Pipeline:
    """Build the V5 regression model."""

    numerical_features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "LapNumberSquared",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
        "TyreLifeSquared_SOFT",
        "TyreLifeSquared_MEDIUM",
        "TyreLifeSquared_HARD",
    ]

    categorical_features = [
        "Compound",
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
                Ridge(
                    alpha=20.0
                ),
            ),
        ]
    )

    return model


# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the V5 model."""

    data = create_features(
        train_df
    )

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "LapNumberSquared",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Compound",
        "Driver",
        "GrandPrix",
        "Season",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
        "TyreLifeSquared_SOFT",
        "TyreLifeSquared_MEDIUM",
        "TyreLifeSquared_HARD",
    ]

    X = data[
        features
    ]

    y = data[
        TARGET
    ]

    model = build_model()

    model.fit(
        X,
        y
    )

    return model


# -------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    label: str,
) -> None:
    """Evaluate model performance."""

    data = create_features(
        df
    )

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "LapNumberSquared",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Compound",
        "Driver",
        "GrandPrix",
        "Season",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
        "TyreLifeSquared_SOFT",
        "TyreLifeSquared_MEDIUM",
        "TyreLifeSquared_HARD",
    ]

    X = data[
        features
    ]

    y = data[
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
        f"{label}:"
    )

    print(
        f"  R²  : {r2:.4f}"
    )

    print(
        f"  MAE : {mae:.4f} sec"
    )


# -------------------------------------------------------------------
# Estimate degradation curves
# -------------------------------------------------------------------

def estimate_curves(
    model: Pipeline,
    train_df: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Estimate compound-specific degradation.

    We compare every tire age against age 1 while holding
    race conditions constant.
    """

    train_df = create_features(
        train_df
    )

    representative = {
        "LapNumber": train_df[
            "LapNumber"
        ].median(),

        "LapNumberSquared": (
            train_df[
                "LapNumber"
            ].median()
            ** 2
        ),

        "TrackTemp": train_df[
            "TrackTemp"
        ].median(),

        "AirTemp": train_df[
            "AirTemp"
        ].median(),

        "Humidity": train_df[
            "Humidity"
        ].median(),

        "Driver": train_df[
            "Driver"
        ].mode()[0],

        "GrandPrix": train_df[
            "GrandPrix"
        ].mode()[0],

        "Season": int(
            train_df[
                "Season"
            ].median()
        ),
    }

    features = [
        "TyreLife",
        "TyreLifeSquared",
        "LapNumber",
        "LapNumberSquared",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Compound",
        "Driver",
        "GrandPrix",
        "Season",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
        "TyreLifeSquared_SOFT",
        "TyreLifeSquared_MEDIUM",
        "TyreLifeSquared_HARD",
    ]

    curves = {}

    for compound in COMPOUNDS:

        rows = []

        for age in range(1, 41):

            row = representative.copy()

            row["TyreLife"] = age
            row["TyreLifeSquared"] = (
                age ** 2
            )

            row["Compound"] = compound

            for current_compound in COMPOUNDS:

                row[
                    f"TyreLife_{current_compound}"
                ] = (
                    age
                    if compound
                    == current_compound
                    else 0
                )

                row[
                    f"TyreLifeSquared_{current_compound}"
                ] = (
                    age ** 2
                    if compound
                    == current_compound
                    else 0
                )

            rows.append(
                row
            )

        curve_df = pd.DataFrame(
            rows
        )

        predictions = model.predict(
            curve_df[
                features
            ]
        )

        baseline = predictions[0]

        degradation = (
            predictions - baseline
        )

        # Prevent the statistical model from
        # producing physically impossible decreases.
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

    ages = [
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
    print("V5 ESTIMATED TIRE DEGRADATION")
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

        for age in ages:

            value = curves[
                compound
            ][age - 1]

            print(
                f"{age:<12}"
                f"{value:>25.4f}"
            )


# -------------------------------------------------------------------
# Save
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
    """Run V5."""

    print()
    print("=" * 70)
    print("F1 TIRE DEGRADATION ESTIMATOR - V5")
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
    print("Creating race-level split...")

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
    print("Training V5 model...")

    model = train_model(
        train_df
    )

    evaluate_model(
        model,
        train_df,
        "Training performance",
    )

    evaluate_model(
        model,
        test_df,
        "Unseen-race performance",
    )

    print()
    print(
        "Estimating compound-specific curves..."
    )

    curves = estimate_curves(
        model,
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
    print("V5 DEGRADATION ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()