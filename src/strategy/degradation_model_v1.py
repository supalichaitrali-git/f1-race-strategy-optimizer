"""
Data-driven F1 tire degradation model.

Estimates tire degradation from real F1 lap data while
controlling for race progression, driver, circuit, season,
track temperature, and tire compound.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
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


# -------------------------------------------------------------------
# Load training data
# -------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """Load and prepare F1 lap data for degradation modeling."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    # ---------------------------------------------------------------
    # Keep dry tires
    # ---------------------------------------------------------------

    df = df[
        df["Compound"].isin(
            DRY_COMPOUNDS
        )
    ].copy()

    # ---------------------------------------------------------------
    # Normal racing conditions
    # ---------------------------------------------------------------

    df = df[
        df["TrackStatus"] == 1
    ].copy()

    # ---------------------------------------------------------------
    # Reasonable tire ages
    # ---------------------------------------------------------------

    df = df[
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= 40)
    ].copy()

    # ---------------------------------------------------------------
    # Remove extreme lap times
    # ---------------------------------------------------------------

    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    # ---------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------

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

    return df.reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# Build regression model
# -------------------------------------------------------------------

def build_model() -> Pipeline:
    """
    Build the regression pipeline.

    Numerical features:
        TyreLife
        LapNumber
        TrackTemp

    Categorical features:
        Compound
        Driver
        GrandPrix
        Season
    """

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
# Train overall model
# -------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
) -> Pipeline:
    """Train the regression model."""

    features = [
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    X = df[features]

    y = df["LapTimeSec"]

    model = build_model()

    model.fit(
        X,
        y,
    )

    return model


# -------------------------------------------------------------------
# Extract tire degradation coefficient
# -------------------------------------------------------------------

def get_tire_age_coefficient(
    model: Pipeline,
) -> float:
    """
    Extract the TyreLife coefficient.

    A positive coefficient means lap time increases
    as tire age increases.
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
# Train compound-specific models
# -------------------------------------------------------------------

def estimate_compound_degradation(
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Train one model per dry compound.

    This allows SOFT, MEDIUM, and HARD to have
    different degradation rates.
    """

    rates = {}

    for compound in DRY_COMPOUNDS:

        compound_data = df[
            df["Compound"] == compound
        ].copy()

        model = train_model(
            compound_data
        )

        rate = get_tire_age_coefficient(
            model
        )

        rates[compound] = rate

    return rates


# -------------------------------------------------------------------
# Evaluate model
# -------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> float:
    """Calculate training R²."""

    features = [
        "TyreLife",
        "LapNumber",
        "Compound",
        "TrackTemp",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    X = df[features]

    y = df["LapTimeSec"]

    return float(
        model.score(
            X,
            y,
        )
    )


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def main() -> None:
    """Run the data-driven degradation analysis."""

    print()
    print("=" * 70)
    print("DATA-DRIVEN F1 TIRE DEGRADATION MODEL")
    print("=" * 70)

    print()
    print("Loading training data...")

    df = load_training_data()

    print(
        f"Training rows: {len(df):,}"
    )

    print()
    print("Estimating compound-specific degradation...")

    rates = estimate_compound_degradation(
        df
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

    print()
    print(
        "Training overall model..."
    )

    overall_model = train_model(
        df
    )

    r_squared = evaluate_model(
        overall_model,
        df
    )

    print()
    print(
        f"Overall training R²: "
        f"{r_squared:.4f}"
    )

    print()
    print("=" * 70)
    print("MODEL COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()