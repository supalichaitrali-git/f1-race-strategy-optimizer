"""
F1 Strategy Physics V4
======================

Research experiment to diagnose unrealistic strategy optimization.

This version DOES NOT modify the production optimizer.

Experiments:
1. Pit-stop loss sensitivity
2. Tire degradation scaling
3. Late-stint degradation penalty
4. 0/1/2-stop strategy comparison
5. Race-length sensitivity
6. Break-even analysis
7. Strategy stability across physics assumptions

IMPORTANT:
This is a physics sensitivity study, not an end-to-end historical
strategy-accuracy model.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tire_calibration_v3.csv"
)

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "f1_race_strategy_dataset.csv"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

MAX_AGE = {
    "SOFT": 20,
    "MEDIUM": 30,
    "HARD": 40,
}


FRESH_PRIORS = {
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


PIT_LOSSES = (
    18.0,
    20.0,
    22.0,
    24.0,
    26.0,
)


DEGRADATION_SCALES = (
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)


LATE_PENALTIES = (
    0.00,
    0.02,
    0.05,
)


MIN_STINT_LAPS = 5
MAX_STOPS = 2

MAX_VALIDATION_RACES = 12


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Race:
    race_key: str
    season: int
    grand_prix: str
    total_laps: int


@dataclass(frozen=True)
class StrategyResult:
    compounds: tuple[str, ...]
    stints: tuple[int, ...]
    pit_stops: int
    total_time: float

    @property
    def strategy_string(self) -> str:
        return " -> ".join(self.compounds)


# ============================================================
# LOAD V3 CALIBRATION
# ============================================================

def load_calibration() -> pd.DataFrame:
    """
    Load the V3 calibration table.

    Actual V3 schema:

        PriorStrength
        Compound
        TyreAge
        FreshPacePriorSec
        DegradationSec
        TotalTireEffectSec
    """

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{CALIBRATION_FILE}\n\n"
            "Run tire_calibration_v3.py first."
        )

    df = pd.read_csv(
        CALIBRATION_FILE
    )

    required_columns = {
        "PriorStrength",
        "Compound",
        "TyreAge",
        "FreshPacePriorSec",
        "DegradationSec",
        "TotalTireEffectSec",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Calibration file is missing columns: "
            f"{sorted(missing)}"
        )

    df["Compound"] = (
        df["Compound"]
        .astype(str)
        .str.upper()
    )

    df["TyreAge"] = pd.to_numeric(
        df["TyreAge"],
        errors="coerce",
    )

    df["FreshPacePriorSec"] = pd.to_numeric(
        df["FreshPacePriorSec"],
        errors="coerce",
    )

    df["DegradationSec"] = pd.to_numeric(
        df["DegradationSec"],
        errors="coerce",
    )

    df["TotalTireEffectSec"] = pd.to_numeric(
        df["TotalTireEffectSec"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Compound",
            "TyreAge",
            "FreshPacePriorSec",
            "DegradationSec",
            "TotalTireEffectSec",
        ]
    )

    return df


# ============================================================
# BUILD V3 DEGRADATION CURVES
# ============================================================

def build_degradation_curves(
    calibration: pd.DataFrame,
) -> dict[str, dict[int, float]]:
    """
    Extract the V3 historical degradation curves.

    V3 already provides DegradationSec directly, so we do not
    reconstruct it from raw laps here.
    """

    curves: dict[str, dict[int, float]] = {}

    for compound in COMPOUNDS:

        subset = calibration[
            calibration["Compound"] == compound
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No calibration data found for {compound}."
            )

        # V3 produces the same historical degradation curve
        # for each prior strength. Therefore use the median
        # across PriorStrength to avoid duplicating information.
        grouped = (
            subset.groupby("TyreAge")[
                "DegradationSec"
            ]
            .median()
            .sort_index()
        )

        grouped = grouped[
            (grouped.index >= 1)
            & (
                grouped.index
                <= MAX_AGE[compound]
            )
        ]

        if grouped.empty:
            raise ValueError(
                f"No usable degradation curve for {compound}."
            )

        ages = (
            grouped.index
            .astype(int)
            .tolist()
        )

        values = (
            grouped.values
            .astype(float)
        )

        # Force monotonic non-decreasing degradation.
        values = np.maximum.accumulate(
            values
        )

        # Age 1 represents a fresh tire.
        values = values - values[0]

        curves[compound] = {
            age: float(value)
            for age, value in zip(
                ages,
                values,
            )
        }

    return curves


# ============================================================
# CURVE INTERPOLATION
# ============================================================

def interpolate_curve_value(
    curve: dict[int, float],
    age: int,
) -> float:
    """
    Interpolate the historical degradation curve.

    Beyond the calibrated maximum age, use a positive linear tail
    instead of freezing degradation.
    """

    ages = np.array(
        sorted(curve.keys()),
        dtype=float,
    )

    values = np.array(
        [
            curve[int(age_value)]
            for age_value in ages
        ],
        dtype=float,
    )

    if age <= ages[0]:
        return float(values[0])

    if age <= ages[-1]:
        return float(
            np.interp(
                age,
                ages,
                values,
            )
        )

    if len(ages) >= 2:

        x1, x2 = ages[-2:]

        y1, y2 = values[-2:]

        slope = (
            (y2 - y1)
            / max(x2 - x1, 1.0)
        )

        # Prevent a flat or negative tail.
        slope = max(
            float(slope),
            0.01,
        )

        return float(
            values[-1]
            + slope
            * (age - ages[-1])
        )

    return float(values[-1])


# ============================================================
# EXPERIMENTAL TIRE MODEL
# ============================================================

class ExperimentalTireModel:
    """
    Experimental tire model used ONLY by V4.

    tire effect =
        experimental fresh-tire prior
        +
        scaled V3 historical degradation
        +
        optional late-stint penalty
    """

    def __init__(
        self,
        curves: dict[str, dict[int, float]],
        prior_name: str,
        degradation_scale: float,
        late_penalty: float,
    ) -> None:

        if prior_name not in FRESH_PRIORS:
            raise ValueError(
                f"Unknown prior: {prior_name}"
            )

        self.curves = curves

        self.priors = (
            FRESH_PRIORS[prior_name]
        )

        self.degradation_scale = (
            degradation_scale
        )

        self.late_penalty = (
            late_penalty
        )

    def tire_effect(
        self,
        compound: str,
        age: int,
    ) -> float:

        compound = compound.upper()

        if compound not in COMPOUNDS:
            raise ValueError(
                f"Unsupported compound: {compound}"
            )

        age = max(
            int(age),
            1,
        )

        fresh_effect = (
            self.priors[compound]
        )

        historical_deg = (
            interpolate_curve_value(
                self.curves[compound],
                age,
            )
        )

        max_age = MAX_AGE[compound]

        late_age = max(
            0,
            age - max_age,
        )

        late_penalty = (
            self.late_penalty
            * late_age
            * late_age
        )

        return (
            fresh_effect
            + self.degradation_scale
            * historical_deg
            + late_penalty
        )


# ============================================================
# PRINT PHYSICS TABLE
# ============================================================

def print_physics_table(
    model: ExperimentalTireModel,
) -> None:

    ages = (
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    )

    print("\nTIRE PHYSICS")

    for compound in COMPOUNDS:

        print(
            f"\n{compound}"
        )

        for age in ages:

            effect = (
                model.tire_effect(
                    compound,
                    age,
                )
            )

            print(
                f"  Age {age:2d}: "
                f"{effect:+.4f} sec"
            )


# ============================================================
# STRATEGY GENERATION
# ============================================================

def generate_compound_sequences(
    stops: int,
) -> list[tuple[str, ...]]:
    """
    Generate realistic compound sequences.

    Consecutive identical compounds are forbidden.
    """

    number_of_stints = (
        stops + 1
    )

    sequences = []

    for sequence in itertools.product(
        COMPOUNDS,
        repeat=number_of_stints,
    ):

        valid = True

        for i in range(
            len(sequence) - 1
        ):

            if (
                sequence[i]
                == sequence[i + 1]
            ):
                valid = False
                break

        if valid:
            sequences.append(
                sequence
            )

    return sequences


def generate_stint_lengths(
    total_laps: int,
    number_of_stints: int,
) -> list[tuple[int, ...]]:
    """
    Generate all stint-length combinations satisfying
    the minimum stint length.
    """

    minimum_required = (
        number_of_stints
        * MIN_STINT_LAPS
    )

    if total_laps < minimum_required:
        return []

    remaining = (
        total_laps
        - minimum_required
    )

    results = []

    def recurse(
        index: int,
        remaining_laps: int,
        current: list[int],
    ) -> None:

        if (
            index
            == number_of_stints - 1
        ):

            length = (
                MIN_STINT_LAPS
                + remaining_laps
            )

            results.append(
                tuple(
                    current
                    + [length]
                )
            )

            return

        for extra in range(
            remaining_laps + 1
        ):

            recurse(
                index + 1,
                remaining_laps - extra,
                current
                + [
                    MIN_STINT_LAPS
                    + extra
                ],
            )

    recurse(
        0,
        remaining,
        [],
    )

    return results


# ============================================================
# STRATEGY VALIDATION
# ============================================================

def strategy_respects_tire_age(
    compounds: tuple[str, ...],
    stints: tuple[int, ...],
) -> bool:

    if len(compounds) != len(
        stints
    ):
        return False

    for compound, stint_length in zip(
        compounds,
        stints,
    ):

        if (
            compound
            not in COMPOUNDS
        ):
            return False

        if (
            stint_length
            < MIN_STINT_LAPS
        ):
            return False

    return True


# ============================================================
# BUILD RACE BASELINE
# ============================================================

def build_race_baseline(
    df: pd.DataFrame,
) -> dict[str, dict[int, float]]:

    required = {
        "Season",
        "GrandPrix",
        "LapNumber",
        "LapTimeSec",
        "TrackStatus",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset missing columns: "
            f"{sorted(missing)}"
        )

    work = df.copy()

    work = work[
        work["TrackStatus"]
        .astype(str)
        .isin(
            [
                "1",
                "1.0",
            ]
        )
    ].copy()

    work = work.dropna(
        subset=[
            "Season",
            "GrandPrix",
            "LapNumber",
            "LapTimeSec",
        ]
    )

    work["Season"] = (
        work["Season"]
        .astype(int)
    )

    work["LapNumber"] = (
        work["LapNumber"]
        .astype(int)
    )

    work["RaceKey"] = (
        work["Season"]
        .astype(str)
        + "_"
        + work["GrandPrix"]
        .astype(str)
    )

    grouped = (
        work.groupby(
            [
                "RaceKey",
                "LapNumber",
            ]
        )["LapTimeSec"]
        .median()
        .reset_index()
    )

    baselines = {}

    for race_key, race_df in grouped.groupby(
        "RaceKey"
    ):

        race_df = (
            race_df
            .sort_values(
                "LapNumber"
            )
        )

        laps = (
            race_df[
                "LapNumber"
            ]
            .to_numpy()
        )

        times = (
            race_df[
                "LapTimeSec"
            ]
            .to_numpy()
        )

        baselines[race_key] = {
            int(lap): float(time)
            for lap, time in zip(
                laps,
                times,
            )
        }

    return baselines


# ============================================================
# BASELINE INTERPOLATION
# ============================================================

def get_baseline_lap_time(
    baseline: dict[int, float],
    lap_number: int,
) -> float:

    if not baseline:
        raise ValueError(
            "Race baseline is empty."
        )

    laps = np.array(
        sorted(
            baseline.keys()
        ),
        dtype=float,
    )

    times = np.array(
        [
            baseline[int(lap)]
            for lap in laps
        ],
        dtype=float,
    )

    return float(
        np.interp(
            lap_number,
            laps,
            times,
        )
    )


# ============================================================
# SIMULATE STRATEGY
# ============================================================

def simulate_strategy(
    race: Race,
    compounds: tuple[str, ...],
    stints: tuple[int, ...],
    tire_model: ExperimentalTireModel,
    baseline: dict[int, float],
    pit_loss: float,
) -> StrategyResult:

    if sum(stints) != (
        race.total_laps
    ):
        raise ValueError(
            "Stints must sum to total race laps."
        )

    total_time = 0.0

    lap_number = 1

    for compound, stint_length in zip(
        compounds,
        stints,
    ):

        for tire_age in range(
            1,
            stint_length + 1,
        ):

            baseline_time = (
                get_baseline_lap_time(
                    baseline,
                    lap_number,
                )
            )

            tire_effect = (
                tire_model.tire_effect(
                    compound,
                    tire_age,
                )
            )

            total_time += (
                baseline_time
                + tire_effect
            )

            lap_number += 1

    total_time += (
        len(compounds) - 1
    ) * pit_loss

    return StrategyResult(
        compounds=compounds,
        stints=stints,
        pit_stops=(
            len(compounds) - 1
        ),
        total_time=total_time,
    )


# ============================================================
# OPTIMIZE ONE RACE
# ============================================================

def optimize_race(
    race: Race,
    tire_model: ExperimentalTireModel,
    baseline: dict[int, float],
    pit_loss: float,
) -> StrategyResult | None:

    best = None

    for stops in range(
        MAX_STOPS + 1
    ):

        sequences = (
            generate_compound_sequences(
                stops
            )
        )

        stint_lengths = (
            generate_stint_lengths(
                race.total_laps,
                stops + 1,
            )
        )

        for compounds in sequences:

            for stints in stint_lengths:

                if not strategy_respects_tire_age(
                    compounds,
                    stints,
                ):
                    continue

                result = (
                    simulate_strategy(
                        race=race,
                        compounds=compounds,
                        stints=stints,
                        tire_model=tire_model,
                        baseline=baseline,
                        pit_loss=pit_loss,
                    )
                )

                if (
                    best is None
                    or result.total_time
                    < best.total_time
                ):
                    best = result

    return best


# ============================================================
# LOAD RACES
# ============================================================

def load_races() -> tuple[
    pd.DataFrame,
    list[Race],
]:

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n"
            f"{DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE,
        low_memory=False,
    )

    df = df.dropna(
        subset=[
            "Season",
            "GrandPrix",
            "LapNumber",
            "LapTimeSec",
        ]
    )

    df["Season"] = (
        df["Season"]
        .astype(int)
    )

    df["LapNumber"] = (
        df["LapNumber"]
        .astype(int)
    )

    races_df = (
        df.groupby(
            [
                "Season",
                "GrandPrix",
            ]
        )["LapNumber"]
        .max()
        .reset_index()
    )

    races = []

    for _, row in races_df.iterrows():

        race_key = (
            f"{int(row['Season'])}_"
            f"{row['GrandPrix']}"
        )

        races.append(
            Race(
                race_key=race_key,
                season=int(
                    row["Season"]
                ),
                grand_prix=str(
                    row["GrandPrix"]
                ),
                total_laps=int(
                    row["LapNumber"]
                ),
            )
        )

    races.sort(
        key=lambda race: (
            race.season,
            race.grand_prix,
        )
    )

    return df, races


# ============================================================
# REPRESENTATIVE RACES
# ============================================================

def select_representative_races(
    races: list[Race],
    maximum: int,
) -> list[Race]:

    if len(races) <= maximum:
        return races

    races_sorted = sorted(
        races,
        key=lambda race: (
            race.total_laps,
            race.season,
        ),
    )

    indices = np.linspace(
        0,
        len(races_sorted) - 1,
        maximum,
        dtype=int,
    )

    selected = []

    for index in indices:
        selected.append(
            races_sorted[int(index)]
        )

    unique = {}

    for race in selected:
        unique[race.race_key] = race

    return list(
        unique.values()
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    races: list[Race],
    baselines: dict[str, dict[int, float]],
    curves: dict[str, dict[int, float]],
    prior_name: str,
    degradation_scale: float,
    late_penalty: float,
    pit_loss: float,
) -> list[StrategyResult]:

    model = ExperimentalTireModel(
        curves=curves,
        prior_name=prior_name,
        degradation_scale=degradation_scale,
        late_penalty=late_penalty,
    )

    results = []

    for index, race in enumerate(
        races,
        start=1,
    ):

        baseline = baselines.get(
            race.race_key
        )

        if baseline is None:
            continue

        result = optimize_race(
            race=race,
            tire_model=model,
            baseline=baseline,
            pit_loss=pit_loss,
        )

        if result is not None:
            results.append(
                result
            )

        print(
            f"\r  Evaluated "
            f"{index}/{len(races)} races",
            end="",
        )

    print()

    return results


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def summarize_results(
    results: list[StrategyResult],
    prior_name: str,
    degradation_scale: float,
    late_penalty: float,
    pit_loss: float,
) -> dict:

    if not results:

        return {
            "Prior": prior_name,
            "DegScale": degradation_scale,
            "LatePenalty": late_penalty,
            "PitLoss": pit_loss,
            "Races": 0,
            "MeanBestTime": np.nan,
            "MedianBestTime": np.nan,
            "MeanStops": np.nan,
            "SoftOnlyPct": np.nan,
            "OneStopPct": np.nan,
            "TwoStopPct": np.nan,
        }

    times = np.array(
        [
            result.total_time
            for result in results
        ]
    )

    stops = np.array(
        [
            result.pit_stops
            for result in results
        ]
    )

    soft_only = sum(
        1
        for result in results
        if (
            len(
                set(
                    result.compounds
                )
            )
            == 1
            and result.compounds[0]
            == "SOFT"
        )
    )

    one_stop = sum(
        1
        for result in results
        if result.pit_stops == 1
    )

    two_stop = sum(
        1
        for result in results
        if result.pit_stops == 2
    )

    n = len(results)

    return {
        "Prior": prior_name,
        "DegScale": degradation_scale,
        "LatePenalty": late_penalty,
        "PitLoss": pit_loss,
        "Races": n,
        "MeanBestTime": float(
            times.mean()
        ),
        "MedianBestTime": float(
            np.median(times)
        ),
        "MeanStops": float(
            stops.mean()
        ),
        "SoftOnlyPct": (
            100.0
            * soft_only
            / n
        ),
        "OneStopPct": (
            100.0
            * one_stop
            / n
        ),
        "TwoStopPct": (
            100.0
            * two_stop
            / n
        ),
    }


# ============================================================
# BREAK-EVEN PIT LOSS
# ============================================================

def calculate_break_even_pit_loss(
    race: Race,
    tire_model: ExperimentalTireModel,
    baseline: dict[int, float],
    no_stop_compound: str,
    one_stop_compounds: tuple[str, str],
    first_stint: int,
) -> float | None:

    second_stint = (
        race.total_laps
        - first_stint
    )

    if (
        second_stint
        < MIN_STINT_LAPS
    ):
        return None

    no_stop = simulate_strategy(
        race=race,
        compounds=(
            no_stop_compound,
        ),
        stints=(
            race.total_laps,
        ),
        tire_model=tire_model,
        baseline=baseline,
        pit_loss=0.0,
    )

    one_stop = simulate_strategy(
        race=race,
        compounds=(
            one_stop_compounds
        ),
        stints=(
            first_stint,
            second_stint,
        ),
        tire_model=tire_model,
        baseline=baseline,
        pit_loss=0.0,
    )

    return float(
        no_stop.total_time
        - one_stop.total_time
    )


# ============================================================
# PHYSICAL VALIDATION
# ============================================================

def validate_physics(
    model: ExperimentalTireModel,
) -> bool:

    print(
        "\nPHYSICAL CHECKS"
    )

    valid = True

    fresh = {
        compound: model.tire_effect(
            compound,
            1,
        )
        for compound in COMPOUNDS
    }

    print(
        "\nFresh tire ordering:"
    )

    for compound in COMPOUNDS:

        print(
            f"  {compound}: "
            f"{fresh[compound]:+.4f} sec"
        )

    if not (
        fresh["SOFT"]
        < fresh["MEDIUM"]
        < fresh["HARD"]
    ):

        print(
            "  FAIL: fresh compound ordering"
        )

        valid = False

    else:

        print(
            "  PASS: fresh compound ordering"
        )

    for compound in COMPOUNDS:

        ages = range(
            1,
            MAX_AGE[compound] + 1,
        )

        values = [
            model.tire_effect(
                compound,
                age,
            )
            for age in ages
        ]

        monotonic = all(
            values[i]
            <= values[i + 1]
            + 1e-9
            for i in range(
                len(values) - 1
            )
        )

        if monotonic:

            print(
                f"  PASS: "
                f"{compound} degradation monotonic"
            )

        else:

            print(
                f"  FAIL: "
                f"{compound} degradation not monotonic"
            )

            valid = False

    return valid


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "F1 STRATEGY PHYSICS V4"
    )

    print(
        "Nonlinear Degradation + Pit-Loss Sensitivity"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "\nLoading processed dataset..."
    )

    df, races = load_races()

    print(
        f"Dataset rows: "
        f"{len(df):,}"
    )

    print(
        f"Available races: "
        f"{len(races)}"
    )

    print(
        "\nBuilding race/lap baselines..."
    )

    baselines = (
        build_race_baseline(
            df
        )
    )

    print(
        f"Race baselines: "
        f"{len(baselines)}"
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    print(
        "\nLoading V3 tire calibration..."
    )

    calibration = (
        load_calibration()
    )

    print(
        f"Calibration rows: "
        f"{len(calibration):,}"
    )

    print(
        "\nCalibration columns:"
    )

    print(
        "  "
        + ", ".join(
            calibration.columns
        )
    )

    curves = (
        build_degradation_curves(
            calibration
        )
    )

    # --------------------------------------------------------
    # RACES
    # --------------------------------------------------------

    validation_races = (
        select_representative_races(
            races,
            MAX_VALIDATION_RACES,
        )
    )

    print(
        "\nREPRESENTATIVE RACES"
    )

    for race in validation_races:

        print(
            f"  {race.race_key:<35} "
            f"{race.total_laps:>3} laps"
        )

    # --------------------------------------------------------
    # BASELINE PHYSICS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BASELINE PHYSICS CHECK"
    )

    print(
        "=" * 70
    )

    baseline_model = (
        ExperimentalTireModel(
            curves=curves,
            prior_name="medium",
            degradation_scale=1.0,
            late_penalty=0.02,
        )
    )

    physics_valid = (
        validate_physics(
            baseline_model
        )
    )

    print_physics_table(
        baseline_model
    )

    # --------------------------------------------------------
    # SENSITIVITY
    # --------------------------------------------------------

    all_summaries = []

    experiment_counter = 0

    total_experiments = (
        len(FRESH_PRIORS)
        * len(DEGRADATION_SCALES)
        * len(LATE_PENALTIES)
        * len(PIT_LOSSES)
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SENSITIVITY EXPERIMENTS"
    )

    print(
        f"Total configurations: "
        f"{total_experiments}"
    )

    print(
        "=" * 70
    )

    for prior_name in FRESH_PRIORS:

        for degradation_scale in (
            DEGRADATION_SCALES
        ):

            for late_penalty in (
                LATE_PENALTIES
            ):

                for pit_loss in PIT_LOSSES:

                    experiment_counter += 1

                    print(
                        "\n"
                        f"[{experiment_counter}/"
                        f"{total_experiments}] "
                        f"prior={prior_name}, "
                        f"deg={degradation_scale:.2f}, "
                        f"late={late_penalty:.2f}, "
                        f"pit={pit_loss:.0f}"
                    )

                    results = (
                        run_experiment(
                            races=validation_races,
                            baselines=baselines,
                            curves=curves,
                            prior_name=prior_name,
                            degradation_scale=(
                                degradation_scale
                            ),
                            late_penalty=(
                                late_penalty
                            ),
                            pit_loss=pit_loss,
                        )
                    )

                    summary = (
                        summarize_results(
                            results=results,
                            prior_name=prior_name,
                            degradation_scale=(
                                degradation_scale
                            ),
                            late_penalty=(
                                late_penalty
                            ),
                            pit_loss=pit_loss,
                        )
                    )

                    all_summaries.append(
                        summary
                    )

                    print(
                        f"  Races: "
                        f"{summary['Races']}"
                    )

                    print(
                        f"  Mean stops: "
                        f"{summary['MeanStops']:.2f}"
                    )

                    print(
                        f"  SOFT-only: "
                        f"{summary['SoftOnlyPct']:.1f}%"
                    )

                    print(
                        f"  1-stop: "
                        f"{summary['OneStopPct']:.1f}%"
                    )

                    print(
                        f"  2-stop: "
                        f"{summary['TwoStopPct']:.1f}%"
                    )

    summary_df = pd.DataFrame(
        all_summaries
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "strategy_physics_v4_results.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:"
        f"\n{output_path}"
    )

    # --------------------------------------------------------
    # DIVERSITY RANKING
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MOST PROMISING PHYSICS CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    ranked = (
        summary_df.copy()
    )

    ranked["DiversityScore"] = (
        (
            100
            - ranked[
                "SoftOnlyPct"
            ]
        ).clip(lower=0)
        + ranked[
            "OneStopPct"
        ]
        + ranked[
            "TwoStopPct"
        ]
    )

    ranked = ranked.sort_values(
        [
            "DiversityScore",
            "TwoStopPct",
            "OneStopPct",
        ],
        ascending=False,
    )

    print(
        ranked[
            [
                "Prior",
                "DegScale",
                "LatePenalty",
                "PitLoss",
                "MeanStops",
                "SoftOnlyPct",
                "OneStopPct",
                "TwoStopPct",
                "DiversityScore",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LOWEST SOFT-ONLY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONFIGURATIONS WITH LOWEST SOFT-ONLY RATE"
    )

    print(
        "=" * 70
    )

    lowest_soft = (
        summary_df.sort_values(
            [
                "SoftOnlyPct",
                "MeanStops",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )

    print(
        lowest_soft[
            [
                "Prior",
                "DegScale",
                "LatePenalty",
                "PitLoss",
                "SoftOnlyPct",
                "OneStopPct",
                "TwoStopPct",
                "MeanStops",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BREAK-EVEN
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BREAK-EVEN PIT-LOSS ANALYSIS"
    )

    print(
        "=" * 70
    )

    break_even_model = (
        ExperimentalTireModel(
            curves=curves,
            prior_name="medium",
            degradation_scale=1.50,
            late_penalty=0.02,
        )
    )

    break_even_rows = []

    for race in validation_races:

        baseline = baselines.get(
            race.race_key
        )

        if baseline is None:
            continue

        for first_stint in range(
            MIN_STINT_LAPS,
            race.total_laps
            - MIN_STINT_LAPS
            + 1,
            5,
        ):

            value = (
                calculate_break_even_pit_loss(
                    race=race,
                    tire_model=(
                        break_even_model
                    ),
                    baseline=baseline,
                    no_stop_compound="SOFT",
                    one_stop_compounds=(
                        "MEDIUM",
                        "HARD",
                    ),
                    first_stint=first_stint,
                )
            )

            if value is not None:

                break_even_rows.append(
                    {
                        "RaceKey": (
                            race.race_key
                        ),
                        "TotalLaps": (
                            race.total_laps
                        ),
                        "FirstStint": (
                            first_stint
                        ),
                        "BreakEvenPitLoss": (
                            value
                        ),
                    }
                )

    if break_even_rows:

        break_even_df = (
            pd.DataFrame(
                break_even_rows
            )
        )

        break_even_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "strategy_physics_v4_break_even.csv"
        )

        break_even_df.to_csv(
            break_even_path,
            index=False,
        )

        print(
            "\nBreak-even results saved:"
            f"\n{break_even_path}"
        )

        print(
            "\nMedian break-even pit loss:"
            f" {break_even_df['BreakEvenPitLoss'].median():.2f} sec"
        )

        print(
            "Mean break-even pit loss:"
            f" {break_even_df['BreakEvenPitLoss'].mean():.2f} sec"
        )

    # --------------------------------------------------------
    # FINAL DIAGNOSIS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "V4 INTERPRETATION"
    )

    print(
        "=" * 70
    )

    if not physics_valid:

        print(
            "\nWARNING:"
            "\nBaseline physics constraints failed."
        )

    lowest_soft_rate = (
        summary_df[
            "SoftOnlyPct"
        ].min()
    )

    highest_two_stop = (
        summary_df[
            "TwoStopPct"
        ].max()
    )

    highest_one_stop = (
        summary_df[
            "OneStopPct"
        ].max()
    )

    print(
        f"\nLowest SOFT-only rate: "
        f"{lowest_soft_rate:.1f}%"
    )

    print(
        f"Highest 1-stop rate: "
        f"{highest_one_stop:.1f}%"
    )

    print(
        f"Highest 2-stop rate: "
        f"{highest_two_stop:.1f}%"
    )

    if lowest_soft_rate >= 90:

        print(
            "\nDIAGNOSIS:"
            "\nSOFT-only dominance persists across "
            "the tested physics range."
        )

        print(
            "\nThis suggests the problem is not merely "
            "the fresh-tire prior or pit-stop loss."
        )

        print(
            "\nThe next model should investigate:"
            "\n  - fuel-load effects"
            "\n  - traffic / clean-air effects"
            "\n  - weather evolution"
            "\n  - circuit-specific tire behavior"
            "\n  - stint-level uncertainty"
            "\n  - safety-car probability"
        )

    elif lowest_soft_rate < 50:

        print(
            "\nPROMISING:"
            "\nSome parameter regions produce "
            "diversified strategies."
        )

        print(
            "\nThose configurations should be investigated "
            "before changing the production optimizer."
        )

    else:

        print(
            "\nPARTIAL IMPROVEMENT:"
            "\nThe sensitivity study reduces SOFT-only "
            "dominance but does not eliminate it."
        )

    print(
        "\nIMPORTANT:"
        "\nV4 is a research experiment."
        "\nProduction optimizer has NOT been modified."
    )

    print(
        "\nV4 COMPLETE."
    )


if __name__ == "__main__":
    main()