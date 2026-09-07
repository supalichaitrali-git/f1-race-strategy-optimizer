"""
Data preprocessing for the F1 Race Strategy Optimizer.

Cleans FastF1 lap data and prepares it for tire
degradation analysis and strategy modeling.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/2024_monza_laps.csv")
OUTPUT_FILE = Path("data/processed/2024_monza_clean.csv")


def load_lap_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load raw FastF1 lap data."""

    return pd.read_csv(input_file)


def clean_lap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FastF1 lap data for strategy analysis.

    Removes inaccurate laps and pit-in/pit-out laps.
    """

    clean = df[
        (df["IsAccurate"] == True)
        & (df["PitInTime"].isna())
        & (df["PitOutTime"].isna())
    ].copy()

    clean = clean.dropna(
        subset=[
            "LapTime",
            "Compound",
            "TyreLife",
        ]
    )

    clean["LapTimeSec"] = (
        pd.to_timedelta(clean["LapTime"])
        .dt.total_seconds()
    )

    columns = [
        "Driver",
        "Team",
        "LapNumber",
        "Stint",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "LapTimeSec",
        "TrackStatus",
        "IsAccurate",
    ]

    return clean[columns].reset_index(drop=True)


def save_processed_data(
    df: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> None:
    """Save processed lap data."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved {len(df)} clean laps.")
    print(f"File: {output_file}")


def main() -> None:
    """Run the preprocessing pipeline."""

    print("Loading raw F1 lap data...")

    df = load_lap_data()

    print(f"Raw laps: {len(df)}")

    clean = clean_lap_data(df)

    print(f"Clean laps: {len(clean)}")

    print()
    print("Compound distribution:")
    print(clean["Compound"].value_counts().to_string())

    save_processed_data(clean)


if __name__ == "__main__":
    main()
