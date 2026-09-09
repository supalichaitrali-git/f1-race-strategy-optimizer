from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ============================================================
# F1 TIRE CALIBRATION V3
# Hybrid physical-prior + historical degradation calibration
# ============================================================

DATA_PATH = Path("data/processed/f1_race_strategy_dataset.csv")
OUTPUT_PATH = Path("data/processed/tire_calibration_v3.csv")


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


# ------------------------------------------------------------
# Compound pace priors
# ------------------------------------------------------------
#
# These are deliberately treated as EXPERIMENTAL priors.
# We will test multiple strengths instead of assuming that
# one fixed tire delta is correct.
#
# MEDIUM is the reference compound.
#
# Example:
#   SOFT   = -0.50 sec
#   MEDIUM =  0.00 sec
#   HARD   = +0.30 sec
#
# Negative = faster than Medium.
# Positive = slower than Medium.
# ------------------------------------------------------------

PACE_PRIORS = {
    "weak": {
        "SOFT": -0.25,
        "MEDIUM": 0.00,
        "HARD": 0.15,
    },
    "medium": {
        "SOFT": -0.50,
        "MEDIUM": 0.00,
        "HARD": 0.30,
    },
    "strong": {
        "SOFT": -0.75,
        "MEDIUM": 0.00,
        "HARD": 0.45,
    },
}


# ------------------------------------------------------------
# Maximum useful tire ages
# ------------------------------------------------------------

MAX_TYRE_AGE = {
    "SOFT": 20,
    "MEDIUM": 30,
    "HARD": 40,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    print("Loading processed dataset...")

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Season",
        "GrandPrix",
        "Driver",
        "LapNumber",
        "LapTimeSec",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "WindSpeed",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = data.copy()

    data["Compound"] = (
        data["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data = data[
        data["Compound"].isin(COMPOUNDS)
    ].copy()

    data["LapTimeSec"] = pd.to_numeric(
        data["LapTimeSec"],
        errors="coerce"
    )

    data["TyreLife"] = pd.to_numeric(
        data["TyreLife"],
        errors="coerce"
    )

    data["LapNumber"] = pd.to_numeric(
        data["LapNumber"],
        errors="coerce"
    )

    data["TrackStatus"] = pd.to_numeric(
        data["TrackStatus"],
        errors="coerce"
    )

    data["TrackTemp"] = pd.to_numeric(
        data["TrackTemp"],
        errors="coerce"
    )

    data["AirTemp"] = pd.to_numeric(
        data["AirTemp"],
        errors="coerce"
    )

    data["Humidity"] = pd.to_numeric(
        data["Humidity"],
        errors="coerce"
    )

    data["WindSpeed"] = pd.to_numeric(
        data["WindSpeed"],
        errors="coerce"
    )

    # Dry, green-flag laps only.
    data = data[
        data["TrackStatus"] == 1
    ].copy()

    # Remove obviously unusable observations.
    data = data[
        data["LapTimeSec"].notna()
        & data["TyreLife"].notna()
        & data["LapNumber"].notna()
    ].copy()

    # Reasonable physical bounds.
    data = data[
        (data["LapTimeSec"] >= 50)
        & (data["LapTimeSec"] <= 150)
        & (data["TyreLife"] >= 1)
        & (data["TyreLife"] <= 80)
        & (data["LapNumber"] >= 1)
    ].copy()

    data["RaceKey"] = (
        data["Season"].astype(str)
        + "_"
        + data["GrandPrix"].astype(str)
    )

    # Approximate race progress.
    race_max_lap = (
        data.groupby("RaceKey")["LapNumber"]
        .transform("max")
    )

    data["RaceProgress"] = (
        data["LapNumber"] / race_max_lap
    )

    print(f"Usable laps: {len(data):,}")
    print(f"Races: {data['RaceKey'].nunique():,}")

    return data


# ============================================================
# RACE / LAP BASELINE
# ============================================================

def create_race_lap_baseline(data):
    print("Creating race/lap baseline...")

    baseline = (
        data.groupby(
            ["RaceKey", "LapNumber"]
        )["LapTimeSec"]
        .median()
        .rename("RaceLapMedian")
        .reset_index()
    )

    data = data.merge(
        baseline,
        on=["RaceKey", "LapNumber"],
        how="left",
    )

    data["RaceLapResidual"] = (
        data["LapTimeSec"]
        - data["RaceLapMedian"]
    )

    return data


# ============================================================
# DRIVER BASELINE
# ============================================================

def create_driver_baseline(data):
    print("Creating driver baseline...")

    baseline = (
        data.groupby(
            ["RaceKey", "Driver"]
        )["RaceLapResidual"]
        .median()
        .rename("DriverRaceResidual")
        .reset_index()
    )

    data = data.merge(
        baseline,
        on=["RaceKey", "Driver"],
        how="left",
    )

    data["AdjustedResidual"] = (
        data["RaceLapResidual"]
        - data["DriverRaceResidual"]
    )

    return data


# ============================================================
# HISTORICAL DEGRADATION
# ============================================================

def estimate_degradation(data):
    print("Estimating historical degradation curves...")

    curves = {}

    for compound in COMPOUNDS:

        subset = data[
            data["Compound"] == compound
        ].copy()

        subset = subset[
            subset["TyreLife"] <= MAX_TYRE_AGE[compound]
        ].copy()

        if subset.empty:
            curves[compound] = {}
            continue

        age_stats = (
            subset.groupby("TyreLife")[
                "AdjustedResidual"
            ]
            .median()
            .reset_index()
        )

        age_stats = age_stats.sort_values(
            "TyreLife"
        )

        # Shift the first observed age to zero.
        first_value = age_stats[
            "AdjustedResidual"
        ].iloc[0]

        age_stats["RelativeResidual"] = (
            age_stats["AdjustedResidual"]
            - first_value
        )

        # Isotonic regression forces physically sensible
        # non-decreasing degradation.
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )

        iso.fit(
            age_stats["TyreLife"],
            age_stats["RelativeResidual"],
        )

        ages = np.arange(
            1,
            MAX_TYRE_AGE[compound] + 1,
        )

        predictions = iso.predict(ages)

        # Normalize age 1 exactly to zero.
        predictions = predictions - predictions[0]

        curves[compound] = {
            int(age): float(value)
            for age, value in zip(
                ages,
                predictions,
            )
        }

    return curves


# ============================================================
# PRIOR + DEGRADATION
# ============================================================

def build_calibration(curves, prior_name):
    prior = PACE_PRIORS[prior_name]

    rows = []

    for compound in COMPOUNDS:

        max_age = MAX_TYRE_AGE[compound]

        for age in range(1, max_age + 1):

            degradation = curves[
                compound
            ].get(age, np.nan)

            if pd.isna(degradation):
                continue

            rows.append({
                "PriorStrength": prior_name,
                "Compound": compound,
                "TyreAge": age,
                "FreshPacePriorSec": prior[compound],
                "DegradationSec": degradation,
                "TotalTireEffectSec": (
                    prior[compound]
                    + degradation
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# PHYSICAL VALIDATION
# ============================================================

def validate_physics(calibration):
    print()
    print("=" * 70)
    print("PHYSICAL VALIDATION")
    print("=" * 70)

    all_passed = True

    for prior_name in PACE_PRIORS:

        subset = calibration[
            calibration["PriorStrength"] == prior_name
        ]

        fresh = (
            subset[
                subset["TyreAge"] == 1
            ]
            .set_index("Compound")
            ["TotalTireEffectSec"]
        )

        soft = fresh["SOFT"]
        medium = fresh["MEDIUM"]
        hard = fresh["HARD"]

        ordering_ok = (
            soft < medium < hard
        )

        print()
        print(
            f"{prior_name.upper()} PRIOR"
        )

        print(
            f"  SOFT   : {soft:+.4f} sec"
        )

        print(
            f"  MEDIUM : {medium:+.4f} sec"
        )

        print(
            f"  HARD   : {hard:+.4f} sec"
        )

        print(
            f"  Fresh ordering SOFT < MEDIUM < HARD: "
            f"{'PASS' if ordering_ok else 'FAIL'}"
        )

        if not ordering_ok:
            all_passed = False

        # Check monotonic degradation.
        for compound in COMPOUNDS:

            values = (
                subset[
                    subset["Compound"] == compound
                ]
                .sort_values("TyreAge")
                ["DegradationSec"]
                .values
            )

            monotonic = np.all(
                np.diff(values) >= -1e-9
            )

            print(
                f"  {compound:<6} degradation monotonic: "
                f"{'PASS' if monotonic else 'FAIL'}"
            )

            if not monotonic:
                all_passed = False

    print()

    if all_passed:
        print(
            "RESULT: All physical constraints passed."
        )
    else:
        print(
            "RESULT: Some physical constraints failed."
        )

    return all_passed


# ============================================================
# PRINT DEGRADATION CURVES
# ============================================================

def print_curves(calibration):
    for prior_name in PACE_PRIORS:

        print()
        print("=" * 70)
        print(
            f"{prior_name.upper()} PRIOR — CALIBRATED CURVES"
        )
        print("=" * 70)

        subset = calibration[
            calibration["PriorStrength"] == prior_name
        ]

        for compound in COMPOUNDS:

            print()
            print(f"{compound}:")

            compound_data = subset[
                subset["Compound"] == compound
            ]

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

            for age in selected_ages:

                row = compound_data[
                    compound_data["TyreAge"] == age
                ]

                if row.empty:
                    continue

                degradation = row[
                    "DegradationSec"
                ].iloc[0]

                total = row[
                    "TotalTireEffectSec"
                ].iloc[0]

                print(
                    f"Age {age:2d}: "
                    f"degradation {degradation:+.4f} sec | "
                    f"total tire effect {total:+.4f} sec"
                )


# ============================================================
# COMPARE PRIOR STRENGTHS
# ============================================================

def compare_priors(calibration):
    print()
    print("=" * 70)
    print("PRIOR STRENGTH COMPARISON")
    print("=" * 70)

    rows = []

    for prior_name in PACE_PRIORS:

        subset = calibration[
            calibration["PriorStrength"] == prior_name
        ]

        fresh = (
            subset[
                subset["TyreAge"] == 1
            ]
            .set_index("Compound")
            ["TotalTireEffectSec"]
        )

        rows.append({
            "Prior": prior_name,
            "SoftFresh": fresh["SOFT"],
            "MediumFresh": fresh["MEDIUM"],
            "HardFresh": fresh["HARD"],
            "SoftMediumGap": (
                fresh["MEDIUM"]
                - fresh["SOFT"]
            ),
            "MediumHardGap": (
                fresh["HARD"]
                - fresh["MEDIUM"]
            ),
        })

    comparison = pd.DataFrame(rows)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 TIRE CALIBRATION V3")
    print("Hybrid Physical Prior + Historical Degradation")
    print("=" * 70)
    print()

    data = load_data()

    data = create_race_lap_baseline(data)

    data = create_driver_baseline(data)

    curves = estimate_degradation(data)

    all_calibrations = []

    for prior_name in PACE_PRIORS:

        calibration = build_calibration(
            curves,
            prior_name,
        )

        all_calibrations.append(
            calibration
        )

    calibration = pd.concat(
        all_calibrations,
        ignore_index=True,
    )

    print_curves(calibration)

    compare_priors(calibration)

    validate_physics(calibration)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved calibration: {OUTPUT_PATH}"
    )

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()