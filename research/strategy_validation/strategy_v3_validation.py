from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


# ============================================================
# F1 STRATEGY V3 VALIDATION
#
# Hybrid tire calibration + strategy-level validation
#
# IMPORTANT:
# This is a research/validation experiment.
# It does NOT modify the production optimizer.
# ============================================================

DATA_PATH = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

CALIBRATION_PATH = Path(
    "data/processed/tire_calibration_v3.csv"
)


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

MAX_TYRE_AGE = {
    "SOFT": 20,
    "MEDIUM": 30,
    "HARD": 40,
}


PRIORS = {
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading processed dataset...")

    data = pd.read_csv(DATA_PATH)

    required = [
        "Season",
        "GrandPrix",
        "Driver",
        "LapNumber",
        "LapTimeSec",
        "Compound",
        "TyreLife",
        "TrackStatus",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["Compound"] = (
        data["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_columns = [
        "LapNumber",
        "LapTimeSec",
        "TyreLife",
        "TrackStatus",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data[
        data["Compound"].isin(COMPOUNDS)
        & data["LapNumber"].notna()
        & data["LapTimeSec"].notna()
        & data["TyreLife"].notna()
        & (data["TrackStatus"] == 1)
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

    print(
        f"Usable laps: {len(data):,}"
    )

    print(
        f"Races: {data['RaceKey'].nunique():,}"
    )

    return data


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():

    print("Loading V3 calibration...")

    calibration = pd.read_csv(
        CALIBRATION_PATH
    )

    required = [
        "PriorStrength",
        "Compound",
        "TyreAge",
        "FreshPacePriorSec",
        "DegradationSec",
        "TotalTireEffectSec",
    ]

    missing = [
        column
        for column in required
        if column not in calibration.columns
    ]

    if missing:
        raise ValueError(
            f"Calibration is missing columns: {missing}"
        )

    calibration["PriorStrength"] = (
        calibration["PriorStrength"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    calibration["Compound"] = (
        calibration["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    calibration["TyreAge"] = pd.to_numeric(
        calibration["TyreAge"],
        errors="coerce",
    )

    return calibration


# ============================================================
# BUILD CALIBRATION LOOKUP
# ============================================================

def build_lookup(calibration):

    lookup = {}

    for _, row in calibration.iterrows():

        key = (
            row["PriorStrength"],
            row["Compound"],
            int(row["TyreAge"]),
        )

        lookup[key] = {
            "fresh": float(
                row["FreshPacePriorSec"]
            ),
            "degradation": float(
                row["DegradationSec"]
            ),
            "total": float(
                row["TotalTireEffectSec"]
            ),
        }

    return lookup


# ============================================================
# GET TIRE EFFECT
# ============================================================

def get_tire_effect(
    lookup,
    prior_name,
    compound,
    age,
):

    max_age = MAX_TYRE_AGE[compound]

    # --------------------------------------------------------
    # If a simulated stint exceeds the calibrated age range,
    # clamp it to the maximum modeled age.
    #
    # This is ONLY for validation. It does not claim that
    # degradation stops physically after this age.
    # --------------------------------------------------------

    effective_age = min(
        max(1, int(age)),
        max_age,
    )

    key = (
        prior_name,
        compound,
        effective_age,
    )

    if key not in lookup:

        return PRIORS[
            prior_name
        ][compound]

    return lookup[key]["total"]


# ============================================================
# CREATE HELD-OUT RACE SPLIT
# ============================================================

def create_holdout_split(data):

    races = sorted(
        data["RaceKey"].unique()
    )

    rng = np.random.default_rng(42)

    shuffled = races.copy()

    rng.shuffle(shuffled)

    test_count = max(
        1,
        int(len(shuffled) * 0.20),
    )

    test_races = set(
        shuffled[:test_count]
    )

    train = data[
        ~data["RaceKey"].isin(test_races)
    ].copy()

    test = data[
        data["RaceKey"].isin(test_races)
    ].copy()

    print()
    print("=" * 70)
    print("HELD-OUT RACE SPLIT")
    print("=" * 70)

    print(
        f"Training races: "
        f"{train['RaceKey'].nunique():,}"
    )

    print(
        f"Testing races : "
        f"{test['RaceKey'].nunique():,}"
    )

    print(
        f"Training laps : "
        f"{len(train):,}"
    )

    print(
        f"Testing laps  : "
        f"{len(test):,}"
    )

    return train, test


# ============================================================
# BUILD COMPLETE RACE LAP BASELINE
# ============================================================

def build_race_lap_baseline(
    race_data,
):

    observed = (
        race_data.groupby(
            "LapNumber"
        )["LapTimeSec"]
        .median()
        .sort_index()
    )

    if observed.empty:
        return {}

    max_lap = int(
        race_data["LapNumber"].max()
    )

    # --------------------------------------------------------
    # Create every lap number from 1 to final race lap.
    # --------------------------------------------------------

    complete_index = pd.Index(
        range(
            1,
            max_lap + 1,
        ),
        name="LapNumber",
    )

    baseline = observed.reindex(
        complete_index
    )

    # --------------------------------------------------------
    # Interpolate missing laps.
    #
    # Missing laps normally occur because the original
    # preprocessing removed non-normal TrackStatus laps.
    # --------------------------------------------------------

    baseline = (
        baseline
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    # --------------------------------------------------------
    # Final safety fallback.
    # --------------------------------------------------------

    baseline = baseline.ffill().bfill()

    return baseline.to_dict()


# ============================================================
# GENERATE STRATEGIES
# ============================================================

def generate_strategies(
    total_laps,
    max_stops=2,
    min_stint=5,
):

    strategies = []

    # --------------------------------------------------------
    # 0 STOP
    # --------------------------------------------------------

    for compound in COMPOUNDS:

        if total_laps >= min_stint:

            strategies.append({
                "compounds": [
                    compound
                ],
                "stints": [
                    total_laps
                ],
            })

    # --------------------------------------------------------
    # 1 STOP
    # --------------------------------------------------------

    for c1, c2 in product(
        COMPOUNDS,
        repeat=2,
    ):

        for split in range(
            min_stint,
            total_laps - min_stint + 1,
        ):

            stint1 = split
            stint2 = (
                total_laps - split
            )

            if stint1 < min_stint:
                continue

            if stint2 < min_stint:
                continue

            strategies.append({
                "compounds": [
                    c1,
                    c2,
                ],
                "stints": [
                    stint1,
                    stint2,
                ],
            })

    # --------------------------------------------------------
    # 2 STOP
    # --------------------------------------------------------

    if max_stops >= 2:

        for c1, c2, c3 in product(
            COMPOUNDS,
            repeat=3,
        ):

            for split1 in range(
                min_stint,
                total_laps - 2 * min_stint + 1,
            ):

                remaining = (
                    total_laps
                    - split1
                )

                for split2 in range(
                    min_stint,
                    remaining - min_stint + 1,
                ):

                    split3 = (
                        total_laps
                        - split1
                        - split2
                    )

                    if split3 < min_stint:
                        continue

                    strategies.append({
                        "compounds": [
                            c1,
                            c2,
                            c3,
                        ],
                        "stints": [
                            split1,
                            split2,
                            split3,
                        ],
                    })

    return strategies


# ============================================================
# SIMULATE STRATEGY
# ============================================================

def simulate_strategy(
    strategy,
    race_lap_baseline,
    prior_name,
    lookup,
    pit_loss=22.0,
):

    total_time = 0.0

    lap_number = 1

    compounds = strategy[
        "compounds"
    ]

    stints = strategy[
        "stints"
    ]

    for compound, stint_length in zip(
        compounds,
        stints,
    ):

        for tyre_age in range(
            1,
            stint_length + 1,
        ):

            context = race_lap_baseline.get(
                lap_number
            )

            if context is None:
                return np.inf

            if not np.isfinite(context):
                return np.inf

            tire_effect = get_tire_effect(
                lookup,
                prior_name,
                compound,
                tyre_age,
            )

            predicted_lap = (
                context
                + tire_effect
            )

            total_time += predicted_lap

            lap_number += 1

    # --------------------------------------------------------
    # Pit-stop loss
    # --------------------------------------------------------

    number_of_stops = (
        len(stints) - 1
    )

    total_time += (
        number_of_stops
        * pit_loss
    )

    return total_time


# ============================================================
# STRATEGY SIGNATURE
# ============================================================

def strategy_signature(
    strategy,
):

    compounds = "-".join(
        strategy["compounds"]
    )

    stints = "-".join(
        str(value)
        for value in strategy["stints"]
    )

    return (
        f"{compounds} | {stints}"
    )


# ============================================================
# ANALYZE ONE RACE
# ============================================================

def analyze_race(
    race_key,
    test,
    prior_name,
    lookup,
):

    race_data = test[
        test["RaceKey"] == race_key
    ].copy()

    if race_data.empty:
        return None

    total_laps = int(
        race_data["LapNumber"].max()
    )

    if total_laps < 30:
        return None

    # --------------------------------------------------------
    # Build complete race timeline.
    # --------------------------------------------------------

    race_lap_baseline = (
        build_race_lap_baseline(
            race_data
        )
    )

    if not race_lap_baseline:
        return None

    strategies = generate_strategies(
        total_laps=total_laps,
        max_stops=2,
        min_stint=5,
    )

    if not strategies:
        return None

    evaluated = []

    for strategy in strategies:

        total_time = simulate_strategy(
            strategy=strategy,
            race_lap_baseline=race_lap_baseline,
            prior_name=prior_name,
            lookup=lookup,
        )

        if np.isfinite(total_time):

            evaluated.append({
                "strategy": strategy,
                "time": total_time,
            })

    if not evaluated:
        return None

    evaluated.sort(
        key=lambda item: item["time"]
    )

    best = evaluated[0]

    second = (
        evaluated[1]
        if len(evaluated) > 1
        else None
    )

    strategy_gap = (
        second["time"]
        - best["time"]
        if second is not None
        else np.nan
    )

    return {
        "RaceKey": race_key,
        "TotalLaps": total_laps,
        "StrategiesEvaluated": len(
            evaluated
        ),
        "BestStrategy": strategy_signature(
            best["strategy"]
        ),
        "BestTime": best["time"],
        "SecondBestTime": (
            second["time"]
            if second is not None
            else np.nan
        ),
        "StrategyGapSec": strategy_gap,
    }


# ============================================================
# COMPOUND BEHAVIOR
# ============================================================

def compound_behavior_test(
    lookup,
    prior_name,
):

    print()
    print(
        f"COMPOUND BEHAVIOR — "
        f"{prior_name.upper()} PRIOR"
    )

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

    for age in ages:

        values = {}

        for compound in COMPOUNDS:

            values[
                compound
            ] = get_tire_effect(
                lookup,
                prior_name,
                compound,
                age,
            )

        print(
            f"Age {age:2d}: "
            f"SOFT {values['SOFT']:+.3f} | "
            f"MEDIUM {values['MEDIUM']:+.3f} | "
            f"HARD {values['HARD']:+.3f}"
        )


# ============================================================
# COMPARE PRIORS
# ============================================================

def compare_priors(
    race_results,
):

    print()
    print("=" * 70)
    print("STRATEGY PRIOR COMPARISON")
    print("=" * 70)

    summaries = []

    for prior_name, results in (
        race_results.items()
    ):

        if not results:
            continue

        frame = pd.DataFrame(
            results
        )

        summaries.append({
            "Prior": prior_name,
            "Races": len(frame),
            "MeanBestTime": frame[
                "BestTime"
            ].mean(),
            "MedianBestTime": frame[
                "BestTime"
            ].median(),
            "MedianStrategyGap": frame[
                "StrategyGapSec"
            ].median(),
            "MeanStrategyGap": frame[
                "StrategyGapSec"
            ].mean(),
        })

    if not summaries:

        print(
            "No strategy results available."
        )

        return None

    summary = pd.DataFrame(
        summaries
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    return summary


# ============================================================
# STRATEGY STABILITY
# ============================================================

def strategy_stability(
    race_results,
):

    print()
    print("=" * 70)
    print("STRATEGY STABILITY ACROSS PRIORS")
    print("=" * 70)

    strategy_tables = {}

    for prior_name, results in (
        race_results.items()
    ):

        if not results:
            continue

        frame = pd.DataFrame(
            results
        )

        strategy_tables[
            prior_name
        ] = (
            frame
            .set_index("RaceKey")
            ["BestStrategy"]
        )

    if len(strategy_tables) < 2:

        print(
            "Not enough prior results "
            "for stability analysis."
        )

        return

    combined = pd.concat(
        strategy_tables,
        axis=1,
    )

    print()
    print(
        combined.head(15).to_string()
    )

    comparisons = [
        ("weak", "medium"),
        ("medium", "strong"),
        ("weak", "strong"),
    ]

    for first, second in comparisons:

        if (
            first not in combined.columns
            or second not in combined.columns
        ):
            continue

        valid = combined[
            [
                first,
                second,
            ]
        ].dropna()

        if valid.empty:
            continue

        agreement = (
            valid[first]
            == valid[second]
        ).mean()

        print(
            f"{first.title()} vs "
            f"{second.title()} agreement: "
            f"{agreement * 100:.1f}%"
        )


# ============================================================
# BEST STRATEGY SUMMARY
# ============================================================

def print_best_strategy_summary(
    race_results,
):

    print()
    print("=" * 70)
    print("SAMPLE BEST STRATEGIES")
    print("=" * 70)

    for prior_name, results in (
        race_results.items()
    ):

        print()
        print(
            f"{prior_name.upper()} PRIOR"
        )

        if not results:

            print(
                "No results."
            )

            continue

        frame = pd.DataFrame(
            results
        )

        columns = [
            "RaceKey",
            "TotalLaps",
            "StrategiesEvaluated",
            "BestStrategy",
            "BestTime",
            "StrategyGapSec",
        ]

        print(
            frame[columns]
            .head(10)
            .to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.3f}",
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY V3 VALIDATION")
    print("=" * 70)

    data = load_data()

    calibration = load_calibration()

    lookup = build_lookup(
        calibration
    )

    train, test = create_holdout_split(
        data
    )

    # --------------------------------------------------------
    # Compound behavior
    # --------------------------------------------------------

    for prior_name in PRIORS:

        compound_behavior_test(
            lookup,
            prior_name,
        )

    # --------------------------------------------------------
    # Evaluate held-out races
    # --------------------------------------------------------

    race_results = {}

    test_races = sorted(
        test["RaceKey"].unique()
    )

    for prior_name in PRIORS:

        print()
        print("=" * 70)
        print(
            f"EVALUATING {prior_name.upper()} PRIOR"
        )
        print("=" * 70)

        results = []

        for index, race_key in enumerate(
            test_races,
            start=1,
        ):

            result = analyze_race(
                race_key=race_key,
                test=test,
                prior_name=prior_name,
                lookup=lookup,
            )

            if result is not None:

                results.append(
                    result
                )

            if index % 5 == 0:

                print(
                    f"Processed "
                    f"{index}/{len(test_races)} "
                    f"held-out races"
                )

        race_results[
            prior_name
        ] = results

        print(
            f"Valid races: "
            f"{len(results)}"
        )

    # --------------------------------------------------------
    # Compare priors
    # --------------------------------------------------------

    compare_priors(
        race_results
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    strategy_stability(
        race_results
    )

    # --------------------------------------------------------
    # Sample strategies
    # --------------------------------------------------------

    print_best_strategy_summary(
        race_results
    )

    print()
    print("=" * 70)
    print("STRATEGY VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()