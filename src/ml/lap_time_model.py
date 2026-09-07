"""
Production lap-time prediction model for the F1 Race Strategy Optimizer.

The model predicts lap time using only information available before
the lap is driven. Sector times and other post-lap information are
intentionally excluded to prevent data leakage.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

MODEL_FILE = Path(
    "models/lap_time_model.joblib"
)

COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


NUMERIC_FEATURES = [
    "TyreLife",
    "LapNumber",
    "RaceProgress",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
]


CATEGORICAL_FEATURES = [
    "Compound",
    "Driver",
    "GrandPrix",
    "Season",
]


TARGET = "LapTimeSec"


def load_data() -> pd.DataFrame:
    """Load and prepare clean race data."""

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
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    df = df.dropna(
        subset=[
            TARGET,
            "TyreLife",
            "LapNumber",
            "Compound",
            "Driver",
            "GrandPrix",
            "Season",
        ]
    )

    df["RaceKey"] = (
        df["Season"].astype(str)
        + "_"
        + df["GrandPrix"].astype(str)
    )

    df["RaceProgress"] = (
        df["LapNumber"]
        / df.groupby("RaceKey")["LapNumber"]
        .transform("max")
    )

    return df


def split_by_race(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by complete races.

    This prevents laps from the same race appearing in both
    training and testing data.
    """

    races = (
        df["RaceKey"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    test_count = max(
        1,
        int(len(races) * test_fraction),
    )

    test_races = set(
        races[-test_count:]
    )

    train_df = df[
        ~df["RaceKey"].isin(test_races)
    ].copy()

    test_df = df[
        df["RaceKey"].isin(test_races)
    ].copy()

    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Build the machine-learning pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Ridge(
        alpha=10.0
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def train_model(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the lap-time prediction model."""

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        TARGET
    ]

    pipeline = build_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
) -> None:
    """Evaluate predictions on completely unseen races."""

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_test = test_df[
        features
    ]

    y_test = test_df[
        TARGET
    ]

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("UNSEEN-RACE MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"R²  : {r2:.4f}"
    )

    print(
        f"MAE : {mae:.4f} sec"
    )

    print(
        f"Test laps : {len(test_df):,}"
    )

    print(
        f"Test races: "
        f"{test_df['RaceKey'].nunique():,}"
    )


def save_model(
    model: Pipeline,
) -> None:
    """Save trained model to disk."""

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()
    print(
        f"Saved model: {MODEL_FILE}"
    )


def main() -> None:
    """Train and evaluate the production lap-time model."""

    print()
    print("=" * 70)
    print("F1 LAP-TIME PREDICTION MODEL")
    print("=" * 70)

    print()
    print("Loading data...")

    df = load_data()

    print(
        f"Usable laps: {len(df):,}"
    )

    print(
        f"Races: "
        f"{df['RaceKey'].nunique():,}"
    )

    print()
    print("Splitting by complete races...")

    train_df, test_df = split_by_race(
        df
    )

    print(
        f"Training laps: {len(train_df):,}"
    )

    print(
        f"Testing laps : {len(test_df):,}"
    )

    print(
        f"Training races: "
        f"{train_df['RaceKey'].nunique():,}"
    )

    print(
        f"Testing races : "
        f"{test_df['RaceKey'].nunique():,}"
    )

    print()
    print("Training model...")

    model = train_model(
        train_df
    )

    print(
        "Model training complete."
    )

    evaluate_model(
        model,
        test_df,
    )

    save_model(
        model
    )

    print()
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()