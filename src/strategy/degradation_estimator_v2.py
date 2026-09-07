"""
F1 tire degradation estimator.

Estimates compound-specific tire degradation while controlling for:
- race progression
- driver
- circuit
- season
- track temperature
- tire compound

Uses race-level validation to test generalization.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

DRY_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]

TARGET = "LapTimeSec"


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and clean F1 dry-tire lap data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df = df[
        df["Compound"].isin(DRY_COMPOUNDS)
    ].copy()

    df = df[
        df["TrackStatus"] == 1
    ]

    df = df[
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= 40)
    ]

    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ]

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
# Create compound-age interaction features
# -------------------------------------------------------------------

def add_interaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create explicit compound × tire-age features.

    These allow each tire compound to have its own
    degradation relationship.
    """

    result = df.copy()

    for compound in DRY_COMPOUNDS:

        column_name = (
            f"TyreLife_{compound}"
        )

        result[column_name] = (
            result["TyreLife"]
            * (
                result["Compound"] == compound
            ).astype(int)
        )

    return result


# -------------------------------------------------------------------
# Create race-level split
# -------------------------------------------------------------------

def split_by_race(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split complete races into training and testing sets.
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
# Build model
# -------------------------------------------------------------------

def build_model() -> Pipeline:
    """
    Build the regression model.

    Numerical features include explicit compound-age
    interaction terms.
    """

    numerical_features = [
        "TyreLife",
        "LapNumber",
        "TrackTemp",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
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
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
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
# Train model
# -------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the compound-aware degradation model."""

    train_data = add_interaction_features(
        train_df
    )

    features = [
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
    ]

    X = train_data[
        features
    ]

    y = train_data[
        TARGET
    ]

    model = build_model()

    model.fit(
        X,
        y,
    )

    return model


# -------------------------------------------------------------------
# Evaluate model
# -------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> tuple[float, float]:
    """Evaluate model using R² and MAE."""

    data = add_interaction_features(
        df
    )

    features = [
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
        "TyreLife_SOFT",
        "TyreLife_MEDIUM",
        "TyreLife_HARD",
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

    return (
        float(r2),
        float(mae),
    )


# -------------------------------------------------------------------
# Estimate controlled degradation curves
# -------------------------------------------------------------------

def estimate_degradation(
    model: Pipeline,
    train_df: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Estimate degradation for each compound.

    Lap number, temperature, driver, circuit and season
    are held constant while tire age changes.
    """

    results = {}

    base_row = {
        "LapNumber": train_df["LapNumber"].median(),
        "TrackTemp": train_df["TrackTemp"].median(),
        "Driver": train_df["Driver"].mode()[0],
        "GrandPrix": train_df["GrandPrix"].mode()[0],
        "Season": int(train_df["Season"].median()),
    }

    for compound in DRY_COMPOUNDS:

        rows = []

        for age in range(1, 41):

            row = base_row.copy()

            row["TyreLife"] = age
            row["Compound"] = compound

            row["TyreLife_SOFT"] = (
                age
                if compound == "SOFT"
                else 0
            )

            row["TyreLife_MEDIUM"] = (
                age
                if compound == "MEDIUM"
                else 0
            )

            row["TyreLife_HARD"] = (
                age
                if compound == "HARD"
                else 0
            )

            rows.append(row)

        prediction_df = pd.DataFrame(
            rows
        )

        predictions = model.predict(
            prediction_df[
                [
                    "TyreLife",
                    "LapNumber",
                    "Compound",
                    "TrackTemp",
                    "Driver",
                    "GrandPrix",
                    "Season",
                    "TyreLife_SOFT",
                    "TyreLife_MEDIUM",
                    "TyreLife_HARD",
                ]
            ]
        )

        baseline = predictions[0]

        degradation = (
            predictions - baseline
        )

        results[compound] = (
            degradation.tolist()
        )

    return results


# -------------------------------------------------------------------
# Print curves
# -------------------------------------------------------------------

def print_degradation_curves(
    curves: dict[str, list[float]],
) -> None:
    """Print estimated compound-specific degradation curves."""

    print()
    print("=" * 70)
    print("COMPOUND-SPECIFIC TIRE DEGRADATION")
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

    for compound in DRY_COMPOUNDS:

        print()
        print("-" * 70)
        print(compound)
        print("-" * 70)

        print(
            f"{'Age':<10}"
            f"{'Estimated degradation (sec)':>30}"
        )

        for age in selected_ages:

            degradation = curves[
                compound
            ][age - 1]

            print(
                f"{age:<10}"
                f"{degradation:>30.4f}"
            )


# -------------------------------------------------------------------
# Save curves
# -------------------------------------------------------------------

def save_curves(
    curves: dict[str, list[float]],
) -> None:
    """Save compound-specific degradation curves."""

    output_file = Path(
        "data/processed/model_degradation_curve.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for compound in DRY_COMPOUNDS:

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
        output_file,
        index=False,
    )

    print()
    print(
        f"Saved model degradation curve: "
        f"{output_file}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run the complete degradation estimation."""

    print()
    print("=" * 70)
    print("COMPOUND-AWARE F1 TIRE DEGRADATION ESTIMATOR")
    print("=" * 70)

    print()
    print("Loading F1 data...")

    df = load_data()

    print(
        f"Usable laps: {len(df):,}"
    )

    print(
        f"Races: "
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
    print(
        "Training compound-aware nonlinear model..."
    )

    model = train_model(
        train_df
    )

    print()
    print("Evaluating training performance...")

    train_r2, train_mae = evaluate_model(
        model,
        train_df,
    )

    print(
        f"Training R²: {train_r2:.4f}"
    )

    print(
        f"Training MAE: {train_mae:.4f} sec"
    )

    print()
    print("Evaluating on unseen races...")

    test_r2, test_mae = evaluate_model(
        model,
        test_df,
    )

    print(
        f"Unseen-race R²: {test_r2:.4f}"
    )

    print(
        f"Unseen-race MAE: {test_mae:.4f} sec"
    )

    print()
    print(
        "Estimating compound-specific degradation curves..."
    )

    curves = estimate_degradation(
        model,
        train_df,
    )

    print_degradation_curves(
        curves
    )

    save_curves(
        curves
    )

    print()
    print("=" * 70)
    print("DEGRADATION ESTIMATION COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()