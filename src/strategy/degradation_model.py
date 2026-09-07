"""
Race-level validated F1 tire degradation model.

Trains the model on complete F1 races and evaluates it on
completely unseen races to measure real generalization.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

FEATURES = [
    "TyreLife",
    "LapNumber",
    "Compound",
    "TrackTemp",
    "Driver",
    "GrandPrix",
    "Season",
]

TARGET = "LapTimeSec"


# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """Load and prepare clean dry-tire F1 lap data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # Keep dry compounds only
    df = df[
        df["Compound"].isin(DRY_COMPOUNDS)
    ].copy()

    # Normal racing conditions
    df = df[
        df["TrackStatus"] == 1
    ].copy()

    # Reasonable tire ages
    df = df[
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= 40)
    ].copy()

    # Remove extreme lap times
    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    # Required columns
    required_columns = FEATURES + [TARGET]

    df = df.dropna(
        subset=required_columns
    )

    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# Build model
# -------------------------------------------------------------------

def build_model() -> Pipeline:
    """Build the regression pipeline."""

    numerical_features = [
        "TyreLife",
        "LapNumber",
        "TrackTemp",
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
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    return model


# -------------------------------------------------------------------
# Split by race
# -------------------------------------------------------------------

def create_race_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by complete races.

    The test set contains races that the model never sees
    during training.
    """

    race_keys = (
        df[
            ["Season", "GrandPrix"]
        ]
        .drop_duplicates()
        .sort_values(
            ["Season", "GrandPrix"]
        )
        .reset_index(drop=True)
    )

    # Use the final 20% of races as unseen test races.
    test_count = max(
        1,
        int(len(race_keys) * 0.20),
    )

    test_races = race_keys.tail(
        test_count
    )

    test_keys = set(
        zip(
            test_races["Season"],
            test_races["GrandPrix"],
        )
    )

    race_pairs = list(
        zip(
            df["Season"],
            df["GrandPrix"],
        )
    )

    is_test = [
        pair in test_keys
        for pair in race_pairs
    ]

    train_df = df[
        ~pd.Series(
            is_test,
            index=df.index,
        )
    ].copy()

    test_df = df[
        pd.Series(
            is_test,
            index=df.index,
        )
    ].copy()

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# -------------------------------------------------------------------
# Train model
# -------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the regression model."""

    model = build_model()

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    model.fit(
        X_train,
        y_train,
    )

    return model


# -------------------------------------------------------------------
# Evaluate model
# -------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Evaluate model using R² and mean absolute error.

    Returns
    -------
    tuple[float, float]
        R² score and MAE in seconds.
    """

    X = df[
        FEATURES
    ]

    y = df[
        TARGET
    ]

    predictions = model.predict(
        X
    )

    r_squared = r2_score(
        y,
        predictions,
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    return (
        float(r_squared),
        float(mae),
    )


# -------------------------------------------------------------------
# Extract tire degradation coefficient
# -------------------------------------------------------------------

def get_tire_age_coefficient(
    model: Pipeline,
) -> float:
    """
    Extract the TyreLife coefficient from the trained model.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    regressor = model.named_steps[
        "regressor"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = pd.Series(
        regressor.coef_,
        index=feature_names,
    )

    tire_age_features = [
        feature
        for feature in coefficients.index
        if feature.endswith(
            "TyreLife"
        )
    ]

    if len(tire_age_features) != 1:
        raise RuntimeError(
            "Could not identify TyreLife coefficient."
        )

    return float(
        coefficients[
            tire_age_features[0]
        ]
    )


# -------------------------------------------------------------------
# Estimate degradation by compound
# -------------------------------------------------------------------

def estimate_compound_degradation(
    train_df: pd.DataFrame,
) -> dict[str, float]:
    """Estimate degradation rates using training races only."""

    rates = {}

    for compound in DRY_COMPOUNDS:

        compound_data = train_df[
            train_df["Compound"] == compound
        ].copy()

        model = train_model(
            compound_data
        )

        rates[compound] = (
            get_tire_age_coefficient(
                model
            )
        )

    return rates


# -------------------------------------------------------------------
# Print test race information
# -------------------------------------------------------------------

def print_test_races(
    test_df: pd.DataFrame,
) -> None:
    """Display races reserved for validation."""

    test_races = (
        test_df[
            ["Season", "GrandPrix"]
        ]
        .drop_duplicates()
        .sort_values(
            ["Season", "GrandPrix"]
        )
    )

    print()
    print("-" * 70)
    print("UNSEEN TEST RACES")
    print("-" * 70)

    for _, race in test_races.iterrows():

        print(
            f"{int(race['Season'])} "
            f"- {race['GrandPrix']}"
        )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run race-level validated degradation analysis."""

    print()
    print("=" * 70)
    print("RACE-LEVEL VALIDATED F1 TIRE DEGRADATION MODEL")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print()
    print("Loading F1 dataset...")

    df = load_training_data()

    print(
        f"Total usable laps: {len(df):,}"
    )

    print(
        f"Total races: "
        f"{df[['Season', 'GrandPrix']].drop_duplicates().shape[0]}"
    )

    # ---------------------------------------------------------------
    # Create race-level split
    # ---------------------------------------------------------------

    print()
    print("Creating race-level train/test split...")

    train_df, test_df = create_race_split(
        df
    )

    print(
        f"Training laps: {len(train_df):,}"
    )

    print(
        f"Testing laps:  {len(test_df):,}"
    )

    print_test_races(
        test_df
    )

    # ---------------------------------------------------------------
    # Train model
    # ---------------------------------------------------------------

    print()
    print("Training model on training races...")

    model = train_model(
        train_df
    )

    # ---------------------------------------------------------------
    # Training evaluation
    # ---------------------------------------------------------------

    train_r2, train_mae = evaluate_model(
        model,
        train_df,
    )

    # ---------------------------------------------------------------
    # Unseen race evaluation
    # ---------------------------------------------------------------

    print()
    print("Evaluating on completely unseen races...")

    test_r2, test_mae = evaluate_model(
        model,
        test_df,
    )

    # ---------------------------------------------------------------
    # Degradation rates
    # ---------------------------------------------------------------

    print()
    print(
        "Estimating compound-specific degradation..."
    )

    rates = estimate_compound_degradation(
        train_df
    )

    print()
    print("-" * 70)
    print("ESTIMATED DEGRADATION RATES")
    print("-" * 70)

    for compound in DRY_COMPOUNDS:

        print(
            f"{compound:<10}"
            f"{rates[compound]:+.4f} sec/lap"
        )

    # ---------------------------------------------------------------
    # Model performance
    # ---------------------------------------------------------------

    print()
    print("-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)

    print(
        f"Training R²: "
        f"{train_r2:.4f}"
    )

    print(
        f"Training MAE: "
        f"{train_mae:.4f} sec"
    )

    print()

    print(
        f"Unseen-race R²: "
        f"{test_r2:.4f}"
    )

    print(
        f"Unseen-race MAE: "
        f"{test_mae:.4f} sec"
    )

    # ---------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------

    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if test_r2 > 0:
        print(
            "The model explains some variation in "
            "lap time on unseen races."
        )
    else:
        print(
            "The model does not generalize well to "
            "the selected unseen races."
        )

    print(
        "Degradation rates should be validated further "
        "before being used by the race strategy simulator."
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()