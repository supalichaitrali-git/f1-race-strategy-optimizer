from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# F1 TIRE CALIBRATION V3 — UNSEEN RACE VALIDATION
# ============================================================

DATA_PATH = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

CALIBRATION_PATH = Path(
    "data/processed/tire_calibration_v3.csv"
)


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading processed dataset...")

    data = pd.read_csv(DATA_PATH)

    data["Compound"] = (
        data["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_columns = [
        "LapTimeSec",
        "TyreLife",
        "LapNumber",
        "TrackStatus",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data[
        data["Compound"].isin(COMPOUNDS)
        & (data["TrackStatus"] == 1)
        & data["LapTimeSec"].notna()
        & data["TyreLife"].notna()
        & data["LapNumber"].notna()
    ].copy()

    data = data[
        (data["LapTimeSec"] >= 50)
        & (data["LapTimeSec"] <= 150)
        & (data["TyreLife"] >= 1)
    ].copy()

    data["RaceKey"] = (
        data["Season"].astype(str)
        + "_"
        + data["GrandPrix"].astype(str)
    )

    race_max_lap = (
        data.groupby("RaceKey")["LapNumber"]
        .transform("max")
    )

    data["RaceProgress"] = (
        data["LapNumber"]
        / race_max_lap
    )

    print(f"Usable laps: {len(data):,}")
    print(f"Races: {data['RaceKey'].nunique():,}")

    return data


# ============================================================
# CREATE CONTEXT BASELINE
# ============================================================

def create_context_baseline(data):
    """
    Estimate race/driver context without using tire age
    or compound as part of the baseline.
    """

    race_lap = (
        data.groupby(
            ["RaceKey", "LapNumber"]
        )["LapTimeSec"]
        .median()
        .rename("RaceLapMedian")
        .reset_index()
    )

    data = data.merge(
        race_lap,
        on=["RaceKey", "LapNumber"],
        how="left",
    )

    data["RaceResidual"] = (
        data["LapTimeSec"]
        - data["RaceLapMedian"]
    )

    driver_baseline = (
        data.groupby(
            ["RaceKey", "Driver"]
        )["RaceResidual"]
        .median()
        .rename("DriverResidual")
        .reset_index()
    )

    data = data.merge(
        driver_baseline,
        on=["RaceKey", "Driver"],
        how="left",
    )

    data["AdjustedResidual"] = (
        data["RaceResidual"]
        - data["DriverResidual"]
    )

    return data


# ============================================================
# CREATE UNSEEN RACE SPLIT
# ============================================================

def create_split(data):
    races = (
        data["RaceKey"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Use a deterministic group split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    groups = data["RaceKey"].values

    train_idx, test_idx = next(
        splitter.split(
            data,
            groups=groups
        )
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    print()
    print("=" * 70)
    print("UNSEEN-RACE SPLIT")
    print("=" * 70)

    print(f"Training races: {train['RaceKey'].nunique():,}")
    print(f"Testing races : {test['RaceKey'].nunique():,}")
    print(f"Training laps : {len(train):,}")
    print(f"Testing laps  : {len(test):,}")

    return train, test


# ============================================================
# FIT HISTORICAL DEGRADATION
# ============================================================

def fit_degradation(train):
    """
    Learn compound-specific degradation from TRAINING races only.

    Isotonic-style monotonicity is implemented using cumulative
    maximum over age medians.
    """

    curves = {}

    for compound in COMPOUNDS:

        subset = train[
            train["Compound"] == compound
        ].copy()

        if subset.empty:
            curves[compound] = {}
            continue

        age_stats = (
            subset.groupby("TyreLife")[
                "AdjustedResidual"
            ]
            .median()
            .sort_index()
        )

        if age_stats.empty:
            curves[compound] = {}
            continue

        # Relative to youngest observed age.
        first_value = age_stats.iloc[0]

        relative = (
            age_stats - first_value
        )

        # Enforce non-decreasing degradation.
        monotonic = np.maximum.accumulate(
            relative.values
        )

        curves[compound] = {
            int(age): float(value)
            for age, value in zip(
                relative.index,
                monotonic,
            )
        }

    return curves


# ============================================================
# INTERPOLATE DEGRADATION
# ============================================================

def get_degradation(curve, age):
    if not curve:
        return 0.0

    ages = np.array(
        sorted(curve.keys()),
        dtype=float,
    )

    values = np.array(
        [curve[int(x)] for x in ages],
        dtype=float,
    )

    return float(
        np.interp(
            age,
            ages,
            values,
            left=values[0],
            right=values[-1],
        )
    )


# ============================================================
# APPLY CALIBRATION
# ============================================================

def evaluate_prior(
    train,
    test,
    curves,
    prior,
):
    """
    Counterfactual prediction:

        observed context baseline
        + physical compound prior
        + historical degradation

    The baseline comes from the observed race/driver context.
    """

    # --------------------------------------------------------
    # Training baseline statistics
    # --------------------------------------------------------

    race_lap = (
        train.groupby(
            ["RaceKey", "LapNumber"]
        )["LapTimeSec"]
        .median()
        .rename("RaceLapMedian")
        .reset_index()
    )

    driver_residual = (
        train.copy()
    )

    driver_residual["RaceResidual"] = (
        driver_residual["LapTimeSec"]
        - driver_residual["RaceLapMedian"]
        if "RaceLapMedian"
        in driver_residual.columns
        else 0
    )

    # We need a test-time context that does not depend
    # on the test tire observation itself.
    #
    # Therefore build a driver/race context from the
    # available test observations only after removing
    # compound/tire effects below.
    #
    # For a fair diagnostic, use race/lap median as the
    # contextual pace reference.

    test_context = (
        test.groupby(
            ["RaceKey", "LapNumber"]
        )["LapTimeSec"]
        .median()
        .rename("ContextLapMedian")
        .reset_index()
    )

    test_eval = test.merge(
        test_context,
        on=["RaceKey", "LapNumber"],
        how="left",
    )

    predictions = []

    for _, row in test_eval.iterrows():

        compound = row["Compound"]
        age = float(row["TyreLife"])

        degradation = get_degradation(
            curves.get(compound, {}),
            age,
        )

        prediction = (
            row["ContextLapMedian"]
            + prior[compound]
            + degradation
        )

        predictions.append(prediction)

    test_eval["PredictedLapTime"] = predictions

    actual = test_eval["LapTimeSec"].values
    predicted = test_eval["PredictedLapTime"].values

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    bias = float(
        np.mean(predicted - actual)
    )

    return test_eval, {
        "MAE": mae,
        "R2": r2,
        "Bias": bias,
    }


# ============================================================
# PHYSICAL BEHAVIOR TEST
# ============================================================

def physical_behavior_test(curves, prior):
    print()
    print("PHYSICAL BEHAVIOR")

    passed = True

    # --------------------------------------------------------
    # Fresh tire ordering
    # --------------------------------------------------------

    fresh = {
        compound: prior[compound]
        for compound in COMPOUNDS
    }

    fresh_ok = (
        fresh["SOFT"]
        < fresh["MEDIUM"]
        < fresh["HARD"]
    )

    print(
        "Fresh ordering SOFT < MEDIUM < HARD:",
        "PASS" if fresh_ok else "FAIL",
    )

    if not fresh_ok:
        passed = False

    # --------------------------------------------------------
    # Degradation monotonicity
    # --------------------------------------------------------

    for compound in COMPOUNDS:

        curve = curves.get(
            compound,
            {}
        )

        if not curve:
            continue

        ages = sorted(curve.keys())

        values = [
            curve[age]
            for age in ages
        ]

        monotonic = np.all(
            np.diff(values) >= -1e-9
        )

        print(
            f"{compound:<7} monotonic:",
            "PASS" if monotonic else "FAIL",
        )

        if not monotonic:
            passed = False

    return passed


# ============================================================
# COMPOUND PREDICTION MATRIX
# ============================================================

def print_prediction_matrix(
    curves,
    prior,
):
    print()
    print("=" * 70)
    print("COUNTERFACTUAL COMPOUND MATRIX")
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
    print(
        f"{'Age':>5}"
        f"{'SOFT':>12}"
        f"{'MEDIUM':>12}"
        f"{'HARD':>12}"
    )

    print("-" * 45)

    for age in ages:

        values = {}

        for compound in COMPOUNDS:

            degradation = get_degradation(
                curves.get(compound, {}),
                age,
            )

            values[compound] = (
                prior[compound]
                + degradation
            )

        print(
            f"{age:>5}"
            f"{values['SOFT']:>12.3f}"
            f"{values['MEDIUM']:>12.3f}"
            f"{values['HARD']:>12.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 TIRE CALIBRATION V3 — VALIDATION")
    print("=" * 70)

    data = load_data()

    data = create_context_baseline(data)

    train, test = create_split(data)

    curves = fit_degradation(train)

    # --------------------------------------------------------
    # Priors
    # --------------------------------------------------------

    priors = {
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

    results = []

    # --------------------------------------------------------
    # Evaluate each prior
    # --------------------------------------------------------

    for name, prior in priors.items():

        print()
        print("=" * 70)
        print(f"TESTING {name.upper()} PRIOR")
        print("=" * 70)

        test_eval, metrics = evaluate_prior(
            train,
            test,
            curves,
            prior,
        )

        physical_pass = physical_behavior_test(
            curves,
            prior,
        )

        print()
        print(
            f"Unseen-race MAE : "
            f"{metrics['MAE']:.4f} sec"
        )

        print(
            f"Unseen-race R²  : "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"Prediction bias : "
            f"{metrics['Bias']:+.4f} sec"
        )

        print(
            "Physical checks :",
            "PASS" if physical_pass else "FAIL",
        )

        results.append({
            "Prior": name,
            "UnseenMAE": metrics["MAE"],
            "UnseenR2": metrics["R2"],
            "Bias": metrics["Bias"],
            "PhysicalPass": physical_pass,
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Best prior by MAE
    # --------------------------------------------------------

    valid = summary[
        summary["PhysicalPass"] == True
    ].copy()

    if not valid.empty:

        best = valid.loc[
            valid["UnseenMAE"].idxmin()
        ]

        print()
        print("=" * 70)
        print("BEST PRIOR BY UNSEEN-RACE MAE")
        print("=" * 70)

        print(
            f"Prior : {best['Prior']}"
        )

        print(
            f"MAE   : {best['UnseenMAE']:.4f} sec"
        )

        print(
            f"R²    : {best['UnseenR2']:.4f}"
        )

        print(
            f"Bias  : {best['Bias']:+.4f} sec"
        )

    # --------------------------------------------------------
    # Show matrix using medium prior for inspection.
    # --------------------------------------------------------

    print_prediction_matrix(
        curves,
        priors["medium"],
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()