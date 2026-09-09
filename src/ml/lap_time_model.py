"""
Counterfactual F1 lap-time model.

The model separates race/context pace from tire effects so that
the strategy optimizer can evaluate hypothetical tire choices.

Architecture:

    Context Model
        ↓
    Expected race pace without tire information
        ↓
    Tire Effect Model
        ↓
    Compound + tire-age adjustment
        ↓
    Counterfactual lap-time prediction
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

MODEL_FILE = Path(
    "models/lap_time_model.joblib"
)

SUPPORTED_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


def load_dataset():
    """Load and prepare the processed F1 dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE
    )

    data = data[
        data["Compound"].isin(
            SUPPORTED_COMPOUNDS
        )
    ].copy()

    data = data[
        data["TrackStatus"] == 1
    ].copy()

    data = data[
        data["LapTimeSec"].notna()
        & data["TyreLife"].notna()
        & data["LapNumber"].notna()
        & data["Driver"].notna()
        & data["GrandPrix"].notna()
        & data["Season"].notna()
    ].copy()

    data["RaceKey"] = (
        data["Season"].astype(str)
        + "_"
        + data["GrandPrix"].astype(str)
    )

    data["RaceProgress"] = (
        data["LapNumber"]
        / data.groupby("RaceKey")["LapNumber"].transform("max")
    )

    data["TyreLife"] = (
        data["TyreLife"].astype(int)
    )

    data["LapNumber"] = (
        data["LapNumber"].astype(int)
    )

    data["Season"] = (
        data["Season"].astype(int)
    )

    return data


def split_by_race(data):
    """Create a chronological unseen-race split."""

    races = (
        data[
            [
                "Season",
                "GrandPrix",
                "RaceKey",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Season",
                "GrandPrix",
            ]
        )
    )

    split_index = int(
        len(races) * 0.80
    )

    training_races = set(
        races.iloc[:split_index][
            "RaceKey"
        ]
    )

    testing_races = set(
        races.iloc[split_index:][
            "RaceKey"
        ]
    )

    train = data[
        data["RaceKey"].isin(
            training_races
        )
    ].copy()

    test = data[
        data["RaceKey"].isin(
            testing_races
        )
    ].copy()

    return (
        train,
        test,
        training_races,
        testing_races,
    )


def build_context_model():
    """
    Build a model that predicts race pace without tire information.

    Tire compound and tire age are deliberately excluded.
    """

    numeric_features = [
        "LapNumber",
        "RaceProgress",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "WindSpeed",
    ]

    categorical_features = [
        "Driver",
        "GrandPrix",
        "Season",
    ]

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
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
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
                "model",
                Ridge(
                    alpha=20.0
                ),
            ),
        ]
    )

    return model


def build_tire_effect_model():
    """
    Build a model for tire-related residual effects.

    Tire age uses polynomial features so the relationship can be
    nonlinear. Compound is interacted with tire age explicitly.
    """

    numeric_features = [
        "TyreLife",
        "TyreLifeSquared",
        "CompoundTyreLife",
        "CompoundTyreLifeSquared",
    ]

    categorical_features = [
        "Compound",
    ]

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
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
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
                "model",
                Ridge(
                    alpha=50.0
                ),
            ),
        ]
    )

    return model


def create_tire_features(data):
    """Create tire-age interaction features."""

    data = data.copy()

    data["TyreLifeSquared"] = (
        data["TyreLife"] ** 2
    )

    data["CompoundTyreLife"] = (
        data["TyreLife"]
        * data["Compound"].map(
            {
                "SOFT": 1.0,
                "MEDIUM": 2.0,
                "HARD": 3.0,
            }
        )
    )

    data["CompoundTyreLifeSquared"] = (
        data["TyreLifeSquared"]
        * data["Compound"].map(
            {
                "SOFT": 1.0,
                "MEDIUM": 2.0,
                "HARD": 3.0,
            }
        )
    )

    return data


def fit_model():
    """Train the counterfactual lap-time model."""

    print()
    print("=" * 70)
    print("COUNTERFACTUAL F1 LAP-TIME MODEL")
    print("=" * 70)

    data = load_dataset()

    print(
        f"Usable laps: {len(data):,}"
    )

    print(
        f"Races: "
        f"{data['RaceKey'].nunique():,}"
    )

    train, test, training_races, testing_races = (
        split_by_race(data)
    )

    print(
        f"Training laps: {len(train):,}"
    )

    print(
        f"Testing laps : {len(test):,}"
    )

    print(
        f"Training races: "
        f"{len(training_races):,}"
    )

    print(
        f"Testing races: "
        f"{len(testing_races):,}"
    )

    # ------------------------------------------------------------
    # STEP 1: CONTEXT MODEL
    # ------------------------------------------------------------

    print()
    print(
        "Training contextual race-pace model..."
    )

    context_model = (
        build_context_model()
    )

    context_features = [
        "LapNumber",
        "RaceProgress",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "WindSpeed",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    context_model.fit(
        train[context_features],
        train["LapTimeSec"],
    )

    train["ContextPrediction"] = (
        context_model.predict(
            train[context_features]
        )
    )

    test["ContextPrediction"] = (
        context_model.predict(
            test[context_features]
        )
    )

    train["TireResidual"] = (
        train["LapTimeSec"]
        - train["ContextPrediction"]
    )

    test["TireResidual"] = (
        test["LapTimeSec"]
        - test["ContextPrediction"]
    )

    # ------------------------------------------------------------
    # STEP 2: TIRE EFFECT MODEL
    # ------------------------------------------------------------

    print(
        "Training tire-effect model..."
    )

    train = create_tire_features(
        train
    )

    test = create_tire_features(
        test
    )

    tire_model = (
        build_tire_effect_model()
    )

    tire_features = [
        "TyreLife",
        "TyreLifeSquared",
        "CompoundTyreLife",
        "CompoundTyreLifeSquared",
        "Compound",
    ]

    tire_model.fit(
        train[tire_features],
        train["TireResidual"],
    )

    train["TirePrediction"] = (
        tire_model.predict(
            train[tire_features]
        )
    )

    test["TirePrediction"] = (
        tire_model.predict(
            test[tire_features]
        )
    )

    train["FinalPrediction"] = (
        train["ContextPrediction"]
        + train["TirePrediction"]
    )

    test["FinalPrediction"] = (
        test["ContextPrediction"]
        + test["TirePrediction"]
    )

    # ------------------------------------------------------------
    # STEP 3: MODEL PERFORMANCE
    # ------------------------------------------------------------

    train_r2 = r2_score(
        train["LapTimeSec"],
        train["FinalPrediction"],
    )

    train_mae = mean_absolute_error(
        train["LapTimeSec"],
        train["FinalPrediction"],
    )

    test_r2 = r2_score(
        test["LapTimeSec"],
        test["FinalPrediction"],
    )

    test_mae = mean_absolute_error(
        test["LapTimeSec"],
        test["FinalPrediction"],
    )

    print()
    print(
        "UNSEEN-RACE MODEL PERFORMANCE"
    )

    print(
        f"Training R² : "
        f"{train_r2:.4f}"
    )

    print(
        f"Training MAE: "
        f"{train_mae:.4f} sec"
    )

    print(
        f"Test R²     : "
        f"{test_r2:.4f}"
    )

    print(
        f"Test MAE    : "
        f"{test_mae:.4f} sec"
    )

    # ------------------------------------------------------------
    # STEP 4: LEARN PHYSICALLY MONOTONIC DEGRADATION CURVES
    # ------------------------------------------------------------

    print()
    print(
        "LEARNED TIRE EFFECTS"
    )

    degradation_curves = {}

    for compound in SUPPORTED_COMPOUNDS:

        compound_data = train[
            train["Compound"] == compound
        ]

        grouped = (
            compound_data
            .groupby("TyreLife")[
                "TireResidual"
            ]
            .median()
            .reset_index()
        )

        if grouped.empty:
            continue

        x = grouped[
            "TyreLife"
        ].to_numpy()

        y = grouped[
            "TireResidual"
        ].to_numpy()

        isotonic = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )

        isotonic.fit(
            x,
            y,
        )

        ages = np.arange(
            1,
            41,
        )

        values = isotonic.predict(
            ages
        )

        values = (
            values
            - values[0]
        )

        # Prevent the learned degradation curve from
        # decreasing with tire age.
        values = np.maximum.accumulate(
            values
        )

        degradation_curves[
            compound
        ] = {
            int(age): float(value)
            for age, value in zip(
                ages,
                values,
            )
        }

        print()
        print(
            f"{compound}:"
        )

        for age in [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
        ]:

            if age in degradation_curves[
                compound
            ]:
                print(
                    f"Age {age:>2}: "
                    f"{degradation_curves[compound][age]:+.4f} sec"
                )

    # ------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------

    model_package = {
        "context_model": context_model,
        "tire_model": tire_model,
        "degradation_curves": degradation_curves,
        "context_features": context_features,
        "tire_features": tire_features,
        "supported_compounds": SUPPORTED_COMPOUNDS,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_r2": test_r2,
        "test_mae": test_mae,
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    print()
    print(
        f"Saved model: {MODEL_FILE}"
    )

    print("=" * 70)

    return model_package


if __name__ == "__main__":
    fit_model()