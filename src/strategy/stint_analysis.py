"""
F1 stint-level tire degradation analysis.

Analyzes lap-time progression within individual tire stints
to estimate how lap time changes as tire age increases.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/stint_degradation_data.csv"
)

COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


def load_data() -> pd.DataFrame:
    """Load and filter clean dry-race data."""

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
        (df["TyreLife"] >= 1)
        & (df["TyreLife"] <= 40)
    ].copy()

    df = df[
        (df["LapTimeSec"] > 0)
        & (df["LapTimeSec"] <= 120)
    ].copy()

    required = [
        "LapTimeSec",
        "TyreLife",
        "LapNumber",
        "Compound",
        "Driver",
        "GrandPrix",
        "Season",
    ]

    df = df.dropna(
        subset=required
    )

    return df


def create_stints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify individual tire stints."""

    result = df.copy()

    group_columns = [
        "Season",
        "GrandPrix",
        "Driver",
    ]

    result = result.sort_values(
        group_columns + ["LapNumber"]
    ).reset_index(drop=True)

    previous_compound = (
        result.groupby(group_columns)[
            "Compound"
        ].shift()
    )

    previous_age = (
        result.groupby(group_columns)[
            "TyreLife"
        ].shift()
    )

    result["NewStint"] = (
        result["Compound"]
        != previous_compound
    ) | (
        result["TyreLife"]
        < previous_age
    )

    result["NewStint"] = (
        result["NewStint"]
        .fillna(True)
    )

    result["StintNumber"] = (
        result.groupby(
            group_columns
        )["NewStint"]
        .cumsum()
    )

    result["StintID"] = (
        result["Season"].astype(str)
        + "_"
        + result["GrandPrix"].astype(str)
        + "_"
        + result["Driver"].astype(str)
        + "_"
        + result["StintNumber"].astype(str)
    )

    return result


def calculate_stint_relative_times(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate lap time relative to the early portion
    of each tire stint.

    The reference is the median lap time from the first
    three usable laps of the stint.
    """

    result = df.copy()

    result["StintLapIndex"] = (
        result.groupby("StintID")
        .cumcount()
        + 1
    )

    early_laps = result[
        result["StintLapIndex"] <= 3
    ]

    reference_times = (
        early_laps.groupby("StintID")[
            "LapTimeSec"
        ]
        .median()
        .rename("FreshReferenceLapTime")
    )

    result = result.merge(
        reference_times,
        on="StintID",
        how="left",
    )

    result = result.dropna(
        subset=["FreshReferenceLapTime"]
    )

    result["RelativeLapTimeSec"] = (
        result["LapTimeSec"]
        - result["FreshReferenceLapTime"]
    )

    return result


def remove_short_stints(
    df: pd.DataFrame,
    minimum_laps: int = 5,
) -> pd.DataFrame:
    """Remove very short tire stints."""

    stint_sizes = (
        df.groupby("StintID")
        .size()
    )

    valid_stints = stint_sizes[
        stint_sizes >= minimum_laps
    ].index

    return df[
        df["StintID"].isin(valid_stints)
    ].copy()


def summarize_by_compound(
    df: pd.DataFrame,
) -> None:
    """Print degradation statistics by compound."""

    print()
    print("=" * 70)
    print("STINT-LEVEL DEGRADATION SUMMARY")
    print("=" * 70)

    for compound in COMPOUNDS:

        compound_df = df[
            df["Compound"] == compound
        ]

        print()
        print("-" * 70)
        print(compound)
        print("-" * 70)

        print(
            f"Stints: "
            f"{compound_df['StintID'].nunique():,}"
        )

        print(
            f"Laps: "
            f"{len(compound_df):,}"
        )

        if len(compound_df) == 0:
            continue

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

            age_df = compound_df[
                compound_df["TyreLife"] == age
            ]

            if len(age_df) == 0:
                continue

            median_value = (
                age_df[
                    "RelativeLapTimeSec"
                ].median()
            )

            print(
                f"Age {age:>2}: "
                f"{median_value:+.4f} sec "
                f"({len(age_df):,} laps)"
            )


def estimate_slopes(
    df: pd.DataFrame,
) -> None:
    """
    Estimate simple degradation slopes for individual stints.

    Only stints with at least 8 laps are used.
    """

    print()
    print("=" * 70)
    print("STINT DEGRADATION SLOPES")
    print("=" * 70)

    slopes = []

    for stint_id, stint in df.groupby(
        "StintID"
    ):

        if len(stint) < 8:
            continue

        x = (
            stint["TyreLife"]
            .to_numpy()
        )

        y = (
            stint["RelativeLapTimeSec"]
            .to_numpy()
        )

        if len(np.unique(x)) < 5:
            continue

        slope = np.polyfit(
            x,
            y,
            1,
        )[0]

        slopes.append(
            {
                "StintID": stint_id,
                "Compound": stint[
                    "Compound"
                ].iloc[0],
                "SlopeSecPerLap": slope,
                "Laps": len(stint),
            }
        )

    slope_df = pd.DataFrame(
        slopes
    )

    for compound in COMPOUNDS:

        compound_slopes = slope_df[
            slope_df["Compound"]
            == compound
        ]

        if len(compound_slopes) == 0:
            continue

        print()
        print(
            f"{compound}:"
        )

        print(
            f"  Stints analyzed: "
            f"{len(compound_slopes):,}"
        )

        print(
            f"  Median slope: "
            f"{compound_slopes['SlopeSecPerLap'].median():+.4f} sec/lap"
        )

        print(
            f"  Mean slope: "
            f"{compound_slopes['SlopeSecPerLap'].mean():+.4f} sec/lap"
        )

        print(
            f"  Positive slope: "
            f"{(compound_slopes['SlopeSecPerLap'] > 0).mean() * 100:.1f}%"
        )


def save_data(
    df: pd.DataFrame,
) -> None:
    """Save stint-level degradation data."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "Season",
        "GrandPrix",
        "Driver",
        "Compound",
        "StintID",
        "LapNumber",
        "TyreLife",
        "StintLapIndex",
        "LapTimeSec",
        "FreshReferenceLapTime",
        "RelativeLapTimeSec",
    ]

    df[
        columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved stint data: "
        f"{OUTPUT_FILE}"
    )


def main() -> None:
    """Run stint-level degradation analysis."""

    print()
    print("=" * 70)
    print("F1 STINT-LEVEL TIRE DEGRADATION ANALYSIS")
    print("=" * 70)

    print()
    print("Loading data...")

    df = load_data()

    print(
        f"Usable laps: "
        f"{len(df):,}"
    )

    print()
    print("Creating tire stints...")

    df = create_stints(
        df
    )

    print(
        f"Initial stints: "
        f"{df['StintID'].nunique():,}"
    )

    print()
    print(
        "Calculating stint-relative lap times..."
    )

    df = calculate_stint_relative_times(
        df
    )

    print()
    print(
        "Removing short stints..."
    )

    df = remove_short_stints(
        df,
        minimum_laps=5,
    )

    print(
        f"Usable stints: "
        f"{df['StintID'].nunique():,}"
    )

    print(
        f"Usable laps: "
        f"{len(df):,}"
    )

    summarize_by_compound(
        df
    )

    estimate_slopes(
        df
    )

    save_data(
        df
    )

    print()
    print("=" * 70)
    print("STINT ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()