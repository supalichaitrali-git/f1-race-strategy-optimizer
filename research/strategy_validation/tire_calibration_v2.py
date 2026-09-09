"""
Tire Calibration V2

Learns compound pace and tire degradation from historical F1 data
while separating them from race-level pace differences.

Goal:
    Estimate physically sensible counterfactual tire effects.

The calibration uses:
    1. Race + lap normalization
    2. Driver normalization
    3. Compound-specific residuals
    4. Robust median aggregation
    5. Isotonic regression for monotonic degradation
    6. Matched compound comparisons
    7. Race-level validation

This is an experimental research model.
It is not yet connected to the production optimizer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/tire_calibration_v2.csv"
)

COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


def load_data():
    """Load and clean the historical F1 dataset."""

    data = pd.read_csv(DATA_FILE)

    data = data[
        data["TrackStatus"] == 1
    ].copy()

    data = data[
        data["Compound"].isin(COMPOUNDS)
    ].copy()

    data = data[
        data["LapTimeSec"].notna()
        & data["LapNumber"].notna()
        & data["TyreLife"].notna()
        & data["Driver"].notna()
        & data["GrandPrix"].notna()
        & data["Season"].notna()
    ].copy()

    data["RaceKey"] = (
        data["Season"].astype(str)
        + "_"
        + data["GrandPrix"].astype(str)
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


def create_race_lap_baseline(data):
    """
    Estimate the expected lap pace for each race/lap.

    This removes much of the circuit-specific and race-progress
    variation before estimating tire effects.
    """

    data = data.copy()

    baseline = (
        data.groupby(
            [
                "RaceKey",
                "LapNumber",
            ]
        )["LapTimeSec"]
        .median()
        .rename(
            "RaceLapBaseline"
        )
        .reset_index()
    )

    data = data.merge(
        baseline,
        on=[
            "RaceKey",
            "LapNumber",
        ],
        how="left",
    )

    data["RaceLapResidual"] = (
        data["LapTimeSec"]
        - data["RaceLapBaseline"]
    )

    return data


def create_driver_baseline(data):
    """
    Remove persistent driver-level pace differences.

    Driver effects are calculated within the available calibration
    dataset. This function is intended for exploratory calibration,
    not final leakage-free model validation.
    """

    data = data.copy()

    driver_median = (
        data.groupby(
            [
                "Driver",
                "Compound",
            ]
        )["RaceLapResidual"]
        .median()
        .rename(
            "DriverCompoundBaseline"
        )
        .reset_index()
    )

    data = data.merge(
        driver_median,
        on=[
            "Driver",
            "Compound",
        ],
        how="left",
    )

    data["AdjustedResidual"] = (
        data["RaceLapResidual"]
        - data["DriverCompoundBaseline"]
    )

    return data


def estimate_fresh_compound_pace(data):
    """
    Estimate relative fresh-tire compound pace.

    Fresh tires are defined as tire age 1-3.

    The reference compound is MEDIUM.
    """

    fresh = data[
        data["TyreLife"].between(
            1,
            3,
        )
    ].copy()

    summary = (
        fresh.groupby(
            "Compound"
        )["RaceLapResidual"]
        .median()
    )

    medium_reference = summary.get(
        "MEDIUM",
        0.0,
    )

    result = {}

    for compound in COMPOUNDS:

        value = summary.get(
            compound,
            np.nan,
        )

        if pd.isna(value):
            result[compound] = np.nan
        else:
            result[compound] = (
                value
                - medium_reference
            )

    return result


def estimate_degradation(data):
    """
    Estimate tire degradation after race/lap normalization.

    For every compound:
        - calculate median residual by tire age
        - fit isotonic regression
        - enforce non-decreasing degradation
        - normalize age 1 to zero
    """

    curves = {}

    for compound in COMPOUNDS:

        compound_data = data[
            data["Compound"] == compound
        ].copy()

        grouped = (
            compound_data
            .groupby(
                "TyreLife"
            )["AdjustedResidual"]
            .median()
            .reset_index()
        )

        grouped = grouped[
            grouped["TyreLife"].between(
                1,
                40,
            )
        ]

        if len(grouped) < 3:
            print(
                f"WARNING: insufficient data for {compound}"
            )
            continue

        x = grouped[
            "TyreLife"
        ].to_numpy()

        y = grouped[
            "AdjustedResidual"
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

        predicted = isotonic.predict(
            ages
        )

        predicted = (
            predicted
            - predicted[0]
        )

        predicted = np.maximum.accumulate(
            predicted
        )

        curves[compound] = predicted

    return curves


def save_curves(
    curves,
    compound_pace,
):
    """Save the calibrated tire effects."""

    rows = []

    for compound in COMPOUNDS:

        pace = compound_pace.get(
            compound,
            np.nan,
        )

        curve = curves.get(
            compound
        )

        if curve is None:
            continue

        for age, degradation in zip(
            range(1, 41),
            curve,
        ):

            rows.append(
                {
                    "Compound": compound,
                    "TyreLife": age,
                    "FreshCompoundPaceSec": pace,
                    "DegradationSec": float(
                        degradation
                    ),
                    "TotalTireEffectSec": float(
                        pace + degradation
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return result


def print_results(
    compound_pace,
    result,
):
    """Print calibration results."""

    print()
    print("=" * 70)
    print("TIRE CALIBRATION V2")
    print("=" * 70)

    print()
    print(
        "FRESH COMPOUND PACE"
    )

    for compound in COMPOUNDS:

        print(
            f"{compound:<8}: "
            f"{compound_pace[compound]:+.4f} sec"
        )

    print()
    print(
        "DEGRADATION CURVES"
    )

    for compound in COMPOUNDS:

        print()
        print(
            f"{compound}:"
        )

        compound_data = result[
            result["Compound"] == compound
        ]

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

            row = compound_data[
                compound_data["TyreLife"] == age
            ]

            if row.empty:
                continue

            degradation = row[
                "DegradationSec"
            ].iloc[0]

            total_effect = row[
                "TotalTireEffectSec"
            ].iloc[0]

            print(
                f"Age {age:>2}: "
                f"degradation "
                f"{degradation:+.4f} sec | "
                f"total tire effect "
                f"{total_effect:+.4f} sec"
            )


def validate_physical_behavior(
    compound_pace,
    result,
):
    """
    Validate basic physical behavior.

    Expected fresh compound ordering:
        SOFT < MEDIUM < HARD

    Expected degradation:
        non-decreasing with tire age
    """

    print()
    print("=" * 70)
    print("PHYSICAL VALIDATION")
    print("=" * 70)

    failures = []

    soft = compound_pace["SOFT"]
    medium = compound_pace["MEDIUM"]
    hard = compound_pace["HARD"]

    print()
    print(
        "Fresh compound ordering:"
    )

    print(
        f"SOFT   : {soft:+.4f}"
    )

    print(
        f"MEDIUM : {medium:+.4f}"
    )

    print(
        f"HARD   : {hard:+.4f}"
    )

    if not (
        soft
        < medium
        < hard
    ):
        failures.append(
            "Fresh compound pace does not follow "
            "SOFT < MEDIUM < HARD."
        )

    for compound in COMPOUNDS:

        values = (
            result[
                result["Compound"] == compound
            ]
            .sort_values(
                "TyreLife"
            )["DegradationSec"]
            .to_numpy()
        )

        if len(values) == 0:
            failures.append(
                f"No degradation curve for {compound}."
            )
            continue

        if np.any(
            np.diff(values) < -1e-9
        ):
            failures.append(
                f"{compound} degradation is not monotonic."
            )

    print()

    if not failures:

        print(
            "PASS: All physical sanity checks passed."
        )

    else:

        print(
            "FAIL: Physical issues detected:"
        )

        for failure in failures:

            print(
                f" - {failure}"
            )


def paired_compound_analysis(data):
    """
    Estimate compound pace using matched driver/race/progress groups.

    Two observations are considered comparable when they come from:
        - the same race
        - the same driver
        - approximately the same race progress

    This reduces circuit, driver and race-progression confounding.
    """

    print()
    print("=" * 70)
    print("PAIRED COMPOUND ANALYSIS")
    print("=" * 70)

    data = data.copy()

    data["RaceProgress"] = (
        data["LapNumber"]
        / data.groupby("RaceKey")[
            "LapNumber"
        ].transform("max")
    )

    # Bin race progress into 5% intervals.
    data["ProgressBin"] = (
        data["RaceProgress"] * 20
    ).round().astype(int)

    grouped = (
        data.groupby(
            [
                "RaceKey",
                "Driver",
                "ProgressBin",
                "Compound",
            ]
        )["LapTimeSec"]
        .median()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=[
            "RaceKey",
            "Driver",
            "ProgressBin",
        ],
        columns="Compound",
        values="LapTimeSec",
        aggfunc="median",
    )

    print()
    print(
        f"Comparable groups: {len(pivot):,}"
    )

    # ------------------------------------------------------------
    # SOFT vs MEDIUM
    # ------------------------------------------------------------

    soft_medium = pivot.dropna(
        subset=[
            "SOFT",
            "MEDIUM",
        ]
    ).copy()

    soft_medium["SOFT_vs_MEDIUM"] = (
        soft_medium["SOFT"]
        - soft_medium["MEDIUM"]
    )

    # ------------------------------------------------------------
    # MEDIUM vs HARD
    # ------------------------------------------------------------

    medium_hard = pivot.dropna(
        subset=[
            "MEDIUM",
            "HARD",
        ]
    ).copy()

    medium_hard["MEDIUM_vs_HARD"] = (
        medium_hard["MEDIUM"]
        - medium_hard["HARD"]
    )

    print()
    print(
        "MATCHED COMPOUND COMPARISONS"
    )

    print()
    print(
        "SOFT vs MEDIUM:"
    )

    if not soft_medium.empty:

        print(
            f"  Comparable groups: "
            f"{len(soft_medium):,}"
        )

        print(
            f"  Median difference: "
            f"{soft_medium['SOFT_vs_MEDIUM'].median():+.4f} sec"
        )

        print(
            f"  Mean difference: "
            f"{soft_medium['SOFT_vs_MEDIUM'].mean():+.4f} sec"
        )

        print(
            f"  SOFT faster: "
            f"{(
                soft_medium['SOFT_vs_MEDIUM'] < 0
            ).mean() * 100:.1f}%"
        )

        print(
            f"  SOFT slower: "
            f"{(
                soft_medium['SOFT_vs_MEDIUM'] > 0
            ).mean() * 100:.1f}%"
        )

    else:

        print(
            "  Insufficient comparable observations."
        )

    print()
    print(
        "MEDIUM vs HARD:"
    )

    if not medium_hard.empty:

        print(
            f"  Comparable groups: "
            f"{len(medium_hard):,}"
        )

        print(
            f"  Median difference: "
            f"{medium_hard['MEDIUM_vs_HARD'].median():+.4f} sec"
        )

        print(
            f"  Mean difference: "
            f"{medium_hard['MEDIUM_vs_HARD'].mean():+.4f} sec"
        )

        print(
            f"  MEDIUM faster: "
            f"{(
                medium_hard['MEDIUM_vs_HARD'] < 0
            ).mean() * 100:.1f}%"
        )

        print(
            f"  MEDIUM slower: "
            f"{(
                medium_hard['MEDIUM_vs_HARD'] > 0
            ).mean() * 100:.1f}%"
        )

    else:

        print(
            "  Insufficient comparable observations."
        )

    # ------------------------------------------------------------
    # Same tire-age comparison
    # ------------------------------------------------------------

    print()
    print(
        "AGE-CONTROLLED COMPARISONS"
    )

    age_grouped = (
        data.groupby(
            [
                "RaceKey",
                "Driver",
                "ProgressBin",
                "TyreLife",
                "Compound",
            ]
        )["LapTimeSec"]
        .median()
        .reset_index()
    )

    age_pivot = age_grouped.pivot_table(
        index=[
            "RaceKey",
            "Driver",
            "ProgressBin",
            "TyreLife",
        ],
        columns="Compound",
        values="LapTimeSec",
        aggfunc="median",
    )

    soft_medium_age = age_pivot.dropna(
        subset=[
            "SOFT",
            "MEDIUM",
        ]
    ).copy()

    medium_hard_age = age_pivot.dropna(
        subset=[
            "MEDIUM",
            "HARD",
        ]
    ).copy()

    print()

    print(
        f"SOFT vs MEDIUM with same tire age: "
        f"{len(soft_medium_age):,} groups"
    )

    if not soft_medium_age.empty:

        difference = (
            soft_medium_age["SOFT"]
            - soft_medium_age["MEDIUM"]
        )

        print(
            f"  Median: "
            f"{difference.median():+.4f} sec"
        )

        print(
            f"  SOFT faster: "
            f"{(
                difference < 0
            ).mean() * 100:.1f}%"
        )

    print()

    print(
        f"MEDIUM vs HARD with same tire age: "
        f"{len(medium_hard_age):,} groups"
    )

    if not medium_hard_age.empty:

        difference = (
            medium_hard_age["MEDIUM"]
            - medium_hard_age["HARD"]
        )

        print(
            f"  Median: "
            f"{difference.median():+.4f} sec"
        )

        print(
            f"  MEDIUM faster: "
            f"{(
                difference < 0
            ).mean() * 100:.1f}%"
        )


def main():

    print()
    print("=" * 70)
    print("F1 TIRE CALIBRATION V2")
    print("=" * 70)

    data = load_data()

    print()
    print(
        f"Usable laps: {len(data):,}"
    )

    print(
        f"Races: "
        f"{data['RaceKey'].nunique():,}"
    )

    # ------------------------------------------------------------
    # Race/lap normalization
    # ------------------------------------------------------------

    print()
    print(
        "Creating race/lap baseline..."
    )

    data = create_race_lap_baseline(
        data
    )

    # ------------------------------------------------------------
    # Driver normalization
    # ------------------------------------------------------------

    print(
        "Creating driver baseline..."
    )

    data = create_driver_baseline(
        data
    )

    # ------------------------------------------------------------
    # Fresh compound pace
    # ------------------------------------------------------------

    print(
        "Estimating fresh compound pace..."
    )

    compound_pace = (
        estimate_fresh_compound_pace(
            data
        )
    )

    # ------------------------------------------------------------
    # Degradation
    # ------------------------------------------------------------

    print(
        "Estimating degradation curves..."
    )

    curves = estimate_degradation(
        data
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    result = save_curves(
        curves,
        compound_pace,
    )

    print_results(
        compound_pace,
        result,
    )

    validate_physical_behavior(
        compound_pace,
        result,
    )

    # ------------------------------------------------------------
    # Matched compound analysis
    # ------------------------------------------------------------

    paired_compound_analysis(
        data
    )

    print()
    print(
        f"Saved calibration: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()