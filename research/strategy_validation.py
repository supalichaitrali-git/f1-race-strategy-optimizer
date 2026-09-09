"""
Strategy-model validation diagnostics.

Tests whether the counterfactual lap-time model produces
physically sensible predictions for different tire compounds
under identical race conditions.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_FILE = Path(
    "models/lap_time_model.joblib"
)

DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


def load_model():
    """Load the trained counterfactual model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


def load_reference_conditions():
    """
    Select a realistic reference race condition.

    We use a clean dry lap from the dataset and keep the
    race context fixed while changing only compound and age.
    """

    data = pd.read_csv(
        DATA_FILE
    )

    data = data[
        data["TrackStatus"] == 1
    ].copy()

    data = data[
        data["Compound"].isin(
            COMPOUNDS
        )
    ].copy()

    data = data[
        data["LapTimeSec"].notna()
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
        / data.groupby("RaceKey")[
            "LapNumber"
        ].transform("max")
    )

    # Pick a representative mid-race lap.
    candidates = data[
        data["RaceProgress"].between(
            0.40,
            0.60,
        )
    ].copy()

    if candidates.empty:
        raise ValueError(
            "Could not find a suitable reference lap."
        )

    reference = (
        candidates
        .sort_values(
            [
                "Season",
                "GrandPrix",
                "LapNumber",
            ]
        )
        .iloc[
            len(candidates) // 2
        ]
    )

    return reference


def create_prediction_row(
    reference,
    compound,
    tyre_life,
):
    """Create one counterfactual prediction input."""

    return {
        "LapNumber": int(
            reference["LapNumber"]
        ),
        "RaceProgress": float(
            reference["RaceProgress"]
        ),
        "TrackTemp": float(
            reference["TrackTemp"]
        ),
        "AirTemp": float(
            reference["AirTemp"]
        ),
        "Humidity": float(
            reference["Humidity"]
        ),
        "WindSpeed": float(
            reference["WindSpeed"]
        ),
        "Driver": reference["Driver"],
        "GrandPrix": reference["GrandPrix"],
        "Season": int(
            reference["Season"]
        ),
        "TyreLife": int(
            tyre_life
        ),
        "TyreLifeSquared": int(
            tyre_life
        ) ** 2,
        "CompoundTyreLife": (
            float(tyre_life)
            * {
                "SOFT": 1.0,
                "MEDIUM": 2.0,
                "HARD": 3.0,
            }[compound]
        ),
        "CompoundTyreLifeSquared": (
            float(tyre_life) ** 2
            * {
                "SOFT": 1.0,
                "MEDIUM": 2.0,
                "HARD": 3.0,
            }[compound]
        ),
        "Compound": compound,
    }


def predict(
    model_package,
    row,
):
    """Generate a counterfactual lap-time prediction."""

    context_model = (
        model_package[
            "context_model"
        ]
    )

    tire_model = (
        model_package[
            "tire_model"
        ]
    )

    context_features = (
        model_package[
            "context_features"
        ]
    )

    tire_features = (
        model_package[
            "tire_features"
        ]
    )

    row_df = pd.DataFrame(
        [row]
    )

    context_prediction = (
        context_model.predict(
            row_df[
                context_features
            ]
        )[0]
    )

    tire_prediction = (
        tire_model.predict(
            row_df[
                tire_features
            ]
        )[0]
    )

    return (
        context_prediction
        + tire_prediction
    )


def test_fresh_tire_pace(
    model_package,
    reference,
):
    """Compare fresh tire performance."""

    print()
    print("=" * 70)
    print("TEST 1 — FRESH TIRE COMPOUND PACE")
    print("=" * 70)

    results = []

    for compound in COMPOUNDS:

        row = create_prediction_row(
            reference=reference,
            compound=compound,
            tyre_life=1,
        )

        prediction = predict(
            model_package,
            row,
        )

        results.append(
            (
                compound,
                prediction,
            )
        )

    for compound, prediction in results:

        print(
            f"{compound:<8} "
            f"{prediction:.3f} sec"
        )

    print()

    ordered = sorted(
        results,
        key=lambda x: x[1],
    )

    print(
        "Predicted order:"
    )

    print(
        " -> ".join(
            compound
            for compound, _ in ordered
        )
    )

    print()

    if ordered[0][0] == "SOFT":
        print(
            "PASS: SOFT is predicted as the fastest fresh compound."
        )
    else:
        print(
            "WARNING: SOFT is NOT predicted as the fastest fresh compound."
        )


def test_tire_age_effect(
    model_package,
    reference,
):
    """Test how predicted lap time changes with tire age."""

    print()
    print("=" * 70)
    print("TEST 2 — TIRE AGE EFFECT")
    print("=" * 70)

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

    for compound in COMPOUNDS:

        print()
        print(
            f"{compound}:"
        )

        previous = None

        for age in ages:

            row = create_prediction_row(
                reference=reference,
                compound=compound,
                tyre_life=age,
            )

            prediction = predict(
                model_package,
                row,
            )

            if previous is None:
                change = 0.0
            else:
                change = (
                    prediction
                    - previous
                )

            print(
                f"Age {age:>2}: "
                f"{prediction:.3f} sec "
                f"({change:+.3f})"
            )

            previous = prediction


def test_compound_matrix(
    model_package,
    reference,
):
    """Compare compounds across several tire ages."""

    print()
    print("=" * 70)
    print("TEST 3 — COMPOUND / AGE MATRIX")
    print("=" * 70)

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

    header = (
        f"{'Age':>5}"
        f"{'SOFT':>12}"
        f"{'MEDIUM':>12}"
        f"{'HARD':>12}"
    )

    print(header)
    print("-" * len(header))

    for age in ages:

        values = {}

        for compound in COMPOUNDS:

            row = create_prediction_row(
                reference=reference,
                compound=compound,
                tyre_life=age,
            )

            values[
                compound
            ] = predict(
                model_package,
                row,
            )

        print(
            f"{age:>5}"
            f"{values['SOFT']:>12.3f}"
            f"{values['MEDIUM']:>12.3f}"
            f"{values['HARD']:>12.3f}"
        )


def test_physical_sanity(
    model_package,
    reference,
):
    """
    Perform basic physical sanity checks.

    These are not strict FIA rules. They are model diagnostics.
    """

    print()
    print("=" * 70)
    print("TEST 4 — PHYSICAL SANITY CHECK")
    print("=" * 70)

    failures = []

    # ------------------------------------------------------------
    # Fresh compound ordering
    # ------------------------------------------------------------

    fresh = {}

    for compound in COMPOUNDS:

        row = create_prediction_row(
            reference,
            compound,
            1,
        )

        fresh[
            compound
        ] = predict(
            model_package,
            row,
        )

    if not (
        fresh["SOFT"]
        < fresh["MEDIUM"]
        < fresh["HARD"]
    ):
        failures.append(
            "Fresh tire pace does not follow SOFT < MEDIUM < HARD."
        )

    # ------------------------------------------------------------
    # Degradation monotonicity
    # ------------------------------------------------------------

    for compound in COMPOUNDS:

        predictions = []

        for age in range(
            1,
            41,
        ):

            row = create_prediction_row(
                reference,
                compound,
                age,
            )

            predictions.append(
                predict(
                    model_package,
                    row,
                )
            )

        differences = np.diff(
            predictions
        )

        if np.any(
            differences < -0.001
        ):
            failures.append(
                f"{compound} predicted lap time decreases "
                "with tire age."
            )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()

    if not failures:

        print(
            "PASS: All basic physical sanity checks passed."
        )

    else:

        print(
            "FAIL: Physical sanity problems detected:"
        )

        for failure in failures:

            print(
                f" - {failure}"
            )


def main():

    print()
    print("=" * 70)
    print("F1 STRATEGY MODEL VALIDATION")
    print("=" * 70)

    model_package = load_model()

    reference = (
        load_reference_conditions()
    )

    print()
    print(
        "REFERENCE CONDITIONS"
    )

    print(
        f"Season       : "
        f"{reference['Season']}"
    )

    print(
        f"Grand Prix   : "
        f"{reference['GrandPrix']}"
    )

    print(
        f"Driver       : "
        f"{reference['Driver']}"
    )

    print(
        f"Lap          : "
        f"{int(reference['LapNumber'])}"
    )

    print(
        f"Race progress: "
        f"{reference['RaceProgress']:.3f}"
    )

    print(
        f"Track temp   : "
        f"{reference['TrackTemp']:.1f} °C"
    )

    print(
        f"Air temp     : "
        f"{reference['AirTemp']:.1f} °C"
    )

    print(
        f"Humidity     : "
        f"{reference['Humidity']:.1f}%"
    )

    print(
        f"Wind speed   : "
        f"{reference['WindSpeed']:.1f} m/s"
    )

    test_fresh_tire_pace(
        model_package,
        reference,
    )

    test_tire_age_effect(
        model_package,
        reference,
    )

    test_compound_matrix(
        model_package,
        reference,
    )

    test_physical_sanity(
        model_package,
        reference,
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()