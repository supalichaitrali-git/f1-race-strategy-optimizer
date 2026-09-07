"""
Observed F1 tire degradation curves.

Calculates how lap time changes with tire age using
real F1 race data, normalized within each driver/race/stint.
"""

from pathlib import Path

import pandas as pd


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
# Load data
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the processed F1 dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


# -------------------------------------------------------------------
# Prepare data
# -------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare clean dry-tire race laps."""

    data = df[
        df["Compound"].isin(DRY_COMPOUNDS)
    ].copy()

    # Normal racing conditions
    data = data[
        data["TrackStatus"] == 1
    ]

    # Reasonable tire ages
    data = data[
        (data["TyreLife"] >= 1)
        & (data["TyreLife"] <= 40)
    ]

    # Remove extreme lap times
    data = data[
        (data["LapTimeSec"] > 0)
        & (data["LapTimeSec"] <= 120)
    ]

    required_columns = [
        "LapTimeSec",
        "TyreLife",
        "Compound",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    data = data.dropna(
        subset=required_columns
    )

    return data.reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# Calculate normalized lap time
# -------------------------------------------------------------------

def calculate_relative_lap_time(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize lap times within each driver/race/compound.

    This reduces the effect of different driver pace and
    circuit characteristics.
    """

    result = data.copy()

    group_columns = [
        "Season",
        "GrandPrix",
        "Driver",
        "Compound",
    ]

    baseline = (
        result
        .groupby(group_columns)["LapTimeSec"]
        .transform("median")
    )

    result["BaselineLapTime"] = baseline

    result["RelativeLapTime"] = (
        result["LapTimeSec"]
        - result["BaselineLapTime"]
    )

    return result


# -------------------------------------------------------------------
# Calculate degradation curve
# -------------------------------------------------------------------

def calculate_degradation_curve(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate median relative lap time for each compound
    and tire age.
    """

    curve = (
        data
        .groupby(
            [
                "Compound",
                "TyreLife",
            ]
        )
        .agg(
            LapCount=(
                "RelativeLapTime",
                "size",
            ),
            MedianRelativeLapTime=(
                "RelativeLapTime",
                "median",
            ),
            MeanRelativeLapTime=(
                "RelativeLapTime",
                "mean",
            ),
        )
        .reset_index()
    )

    return curve


# -------------------------------------------------------------------
# Print curve
# -------------------------------------------------------------------

def print_curve(
    curve: pd.DataFrame,
) -> None:
    """Print degradation curves for each compound."""

    print()
    print("=" * 80)
    print("OBSERVED F1 TIRE DEGRADATION CURVES")
    print("=" * 80)

    for compound in DRY_COMPOUNDS:

        compound_curve = curve[
            curve["Compound"] == compound
        ].copy()

        print()
        print("-" * 80)
        print(f"{compound}")
        print("-" * 80)

        print(
            compound_curve[
                [
                    "TyreLife",
                    "LapCount",
                    "MedianRelativeLapTime",
                    "MeanRelativeLapTime",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


# -------------------------------------------------------------------
# Save curve
# -------------------------------------------------------------------

def save_curve(
    curve: pd.DataFrame,
) -> None:
    """Save degradation curve to processed data."""

    output_file = Path(
        "data/processed/tire_degradation_curve.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"Saved degradation curve: {output_file}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run observed tire degradation analysis."""

    print()
    print("Loading F1 dataset...")

    df = load_data()

    print(
        f"Dataset rows: {len(df):,}"
    )

    print()
    print("Preparing dry-tire data...")

    data = prepare_data(
        df
    )

    print(
        f"Usable dry-tire laps: {len(data):,}"
    )

    print()
    print(
        "Normalizing lap times..."
    )

    data = calculate_relative_lap_time(
        data
    )

    print()
    print(
        "Calculating degradation curves..."
    )

    curve = calculate_degradation_curve(
        data
    )

    print_curve(
        curve
    )

    save_curve(
        curve
    )

    print()
    print("=" * 80)
    print("DEGRADATION CURVE ANALYSIS COMPLETE")
    print("=" * 80)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()