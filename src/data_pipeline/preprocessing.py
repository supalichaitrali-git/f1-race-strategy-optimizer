"""
Unified preprocessing pipeline for the F1 Race Strategy Optimizer.

Processes all downloaded Grand Prix lap and weather data,
cleans invalid laps, merges weather information using the
nearest timestamp, and creates one unified dataset.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "f1_race_strategy_dataset.csv"
)


# -------------------------------------------------------------------
# Required columns
# -------------------------------------------------------------------

LAP_COLUMNS = [
    "Time",
    "Driver",
    "DriverNumber",
    "LapTime",
    "LapNumber",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "IsPersonalBest",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Team",
    "LapStartTime",
    "LapStartDate",
    "TrackStatus",
    "Position",
    "Deleted",
    "DeletedReason",
    "FastF1Generated",
    "IsAccurate",
]


WEATHER_COLUMNS = [
    "Time",
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindDirection",
    "WindSpeed",
]


# -------------------------------------------------------------------
# Find race files
# -------------------------------------------------------------------

def find_lap_files() -> list[Path]:
    """
    Find all downloaded Grand Prix lap CSV files.
    """

    lap_files = sorted(
        RAW_DATA_DIR.glob(
            "*/*_laps.csv"
        )
    )

    return lap_files


# -------------------------------------------------------------------
# Parse time columns
# -------------------------------------------------------------------

def parse_time_column(
    series: pd.Series,
) -> pd.Series:
    """
    Convert FastF1 time strings into pandas timedeltas.
    """

    return pd.to_timedelta(
        series,
        errors="coerce",
    )


# -------------------------------------------------------------------
# Load lap data
# -------------------------------------------------------------------

def load_lap_data(
    lap_file: Path,
) -> pd.DataFrame:
    """
    Load one race lap CSV.
    """

    df = pd.read_csv(
        lap_file,
    )

    return df


# -------------------------------------------------------------------
# Load weather data
# -------------------------------------------------------------------

def load_weather_data(
    weather_file: Path,
) -> pd.DataFrame:
    """
    Load one race weather CSV.
    """

    df = pd.read_csv(
        weather_file,
    )

    return df


# -------------------------------------------------------------------
# Clean lap data
# -------------------------------------------------------------------

def clean_lap_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean FastF1 lap data.

    Removes:
    - inaccurate laps
    - pit-in laps
    - pit-out laps
    - laps without lap time
    - laps without tire compound
    - laps without tire age
    """

    clean = df.copy()

    # ---------------------------------------------------------------
    # Accurate laps only
    # ---------------------------------------------------------------

    clean = clean[
        clean["IsAccurate"].fillna(False)
    ]

    # ---------------------------------------------------------------
    # Remove pit-in and pit-out laps
    # ---------------------------------------------------------------

    clean = clean[
        clean["PitInTime"].isna()
        & clean["PitOutTime"].isna()
    ]

    # ---------------------------------------------------------------
    # Remove rows required for strategy analysis
    # ---------------------------------------------------------------

    clean = clean.dropna(
        subset=[
            "LapTime",
            "Compound",
            "TyreLife",
            "LapNumber",
            "Driver",
        ]
    )

    # ---------------------------------------------------------------
    # Convert lap time to seconds
    # ---------------------------------------------------------------

    clean["LapTimeSec"] = (
        parse_time_column(
            clean["LapTime"]
        ).dt.total_seconds()
    )

    # ---------------------------------------------------------------
    # Convert session time
    # ---------------------------------------------------------------

    clean["Time"] = parse_time_column(
        clean["Time"]
    )

    clean["LapStartTime"] = parse_time_column(
        clean["LapStartTime"]
    )

    # ---------------------------------------------------------------
    # Remove invalid lap times
    # ---------------------------------------------------------------

    clean = clean[
        clean["LapTimeSec"].notna()
        & (clean["LapTimeSec"] > 0)
    ]

    # ---------------------------------------------------------------
    # Normalize tire compound
    # ---------------------------------------------------------------

    clean["Compound"] = (
        clean["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return clean


# -------------------------------------------------------------------
# Merge weather
# -------------------------------------------------------------------

def merge_weather(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the nearest weather observation onto each lap.

    FastF1 weather data is sampled at intervals, while lap data
    occurs once per lap. Therefore an exact timestamp match
    is not appropriate.

    merge_asof() finds the closest weather observation.
    """

    laps = laps.copy()
    weather = weather.copy()

    # ---------------------------------------------------------------
    # Convert weather time
    # ---------------------------------------------------------------

    weather["Time"] = parse_time_column(
        weather["Time"]
    )

    # ---------------------------------------------------------------
    # Remove invalid timestamps
    # ---------------------------------------------------------------

    laps = laps[
        laps["Time"].notna()
    ].copy()

    weather = weather[
        weather["Time"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Sort by time
    # ---------------------------------------------------------------

    laps = laps.sort_values(
        "Time"
    )

    weather = weather.sort_values(
        "Time"
    )

    # ---------------------------------------------------------------
    # Merge nearest weather observation
    # ---------------------------------------------------------------

    merged = pd.merge_asof(
        laps,
        weather,
        on="Time",
        direction="nearest",
        suffixes=("", "_Weather"),
    )

    # ---------------------------------------------------------------
    # Calculate weather timestamp difference
    # ---------------------------------------------------------------

    weather_times = weather[
        ["Time"]
    ].rename(
        columns={
            "Time": "WeatherTime"
        }
    )

    merged = pd.merge_asof(
        merged.sort_values("Time"),
        weather_times.sort_values("WeatherTime"),
        left_on="Time",
        right_on="WeatherTime",
        direction="nearest",
    )

    merged["WeatherTimeDifferenceSec"] = (
        (
            merged["Time"]
            - merged["WeatherTime"]
        )
        .abs()
        .dt.total_seconds()
    )

    merged = merged.drop(
        columns=["WeatherTime"]
    )

    return merged


# -------------------------------------------------------------------
# Add race metadata
# -------------------------------------------------------------------

def add_race_metadata(
    df: pd.DataFrame,
    year: int,
    grand_prix: str,
) -> pd.DataFrame:
    """
    Add season and Grand Prix metadata.
    """

    result = df.copy()

    result["Season"] = year
    result["GrandPrix"] = grand_prix

    return result


# -------------------------------------------------------------------
# Process one race
# -------------------------------------------------------------------

def process_race(
    lap_file: Path,
) -> pd.DataFrame:
    """
    Process one Grand Prix.

    Loads lap and weather data, cleans the laps,
    merges weather, and adds race metadata.
    """

    print()
    print("-" * 70)
    print(
        f"Processing: {lap_file}"
    )
    print("-" * 70)

    # ---------------------------------------------------------------
    # Determine season
    # ---------------------------------------------------------------

    year = int(
        lap_file.parent.name
    )

    # ---------------------------------------------------------------
    # Determine weather file
    # ---------------------------------------------------------------

    weather_file = Path(
        str(lap_file)
        .replace(
            "_laps.csv",
            "_weather.csv",
        )
    )

    if not weather_file.exists():

        raise FileNotFoundError(
            f"Weather file missing: {weather_file}"
        )

    # ---------------------------------------------------------------
    # Grand Prix name
    # ---------------------------------------------------------------

    filename = lap_file.stem

    grand_prix = (
        filename
        .replace(
            f"{year}_",
            "",
            1,
        )
        .replace(
            "_laps",
            "",
        )
        .replace(
            "_",
            " ",
        )
        .title()
    )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    laps = load_lap_data(
        lap_file
    )

    weather = load_weather_data(
        weather_file
    )

    print(
        f"Raw laps: {len(laps)}"
    )

    print(
        f"Weather records: {len(weather)}"
    )

    # ---------------------------------------------------------------
    # Clean laps
    # ---------------------------------------------------------------

    clean_laps = clean_lap_data(
        laps
    )

    print(
        f"Clean laps: {len(clean_laps)}"
    )

    # ---------------------------------------------------------------
    # Merge weather
    # ---------------------------------------------------------------

    merged = merge_weather(
        clean_laps,
        weather,
    )

    # ---------------------------------------------------------------
    # Add race metadata
    # ---------------------------------------------------------------

    merged = add_race_metadata(
        merged,
        year,
        grand_prix,
    )

    # ---------------------------------------------------------------
    # Reorder important columns
    # ---------------------------------------------------------------

    priority_columns = [
        "Season",
        "GrandPrix",
        "Driver",
        "DriverNumber",
        "Team",
        "LapNumber",
        "LapTimeSec",
        "Stint",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "Position",
        "TrackStatus",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "WindDirection",
        "WindSpeed",
        "WeatherTimeDifferenceSec",
    ]

    existing_priority_columns = [
        column
        for column in priority_columns
        if column in merged.columns
    ]

    remaining_columns = [
        column
        for column in merged.columns
        if column not in existing_priority_columns
    ]

    merged = merged[
        existing_priority_columns
        + remaining_columns
    ]

    return merged.reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# Process all races
# -------------------------------------------------------------------

def process_all_races() -> pd.DataFrame:
    """
    Process every downloaded Grand Prix.
    """

    lap_files = find_lap_files()

    print()
    print("=" * 70)
    print("F1 UNIFIED PREPROCESSING PIPELINE")
    print("=" * 70)

    print(
        f"Lap files found: {len(lap_files)}"
    )

    if not lap_files:

        raise FileNotFoundError(
            "No lap files found in data/raw."
        )

    all_races = []

    successful = 0
    failed = 0

    # ---------------------------------------------------------------
    # Process every race
    # ---------------------------------------------------------------

    for number, lap_file in enumerate(
        lap_files,
        start=1,
    ):

        print()
        print(
            f"Race {number}/{len(lap_files)}"
        )

        try:

            processed = process_race(
                lap_file
            )

            all_races.append(
                processed
            )

            successful += 1

        except Exception as error:

            failed += 1

            print()
            print(
                f"FAILED: {lap_file}"
            )

            print(
                f"Reason: {error}"
            )

    # ---------------------------------------------------------------
    # Combine all races
    # ---------------------------------------------------------------

    if not all_races:

        raise RuntimeError(
            "No races were successfully processed."
        )

    combined = pd.concat(
        all_races,
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # Sort final dataset
    # ---------------------------------------------------------------

    combined = combined.sort_values(
        [
            "Season",
            "GrandPrix",
            "Driver",
            "LapNumber",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Save processed dataset
    # ---------------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Successful races: {successful}"
    )

    print(
        f"Failed races: {failed}"
    )

    print(
        f"Total rows: {len(combined):,}"
    )

    print(
        f"Total columns: {len(combined.columns)}"
    )

    print()
    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print()
    print("Seasons:")
    print(
        combined["Season"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Tire compounds:")
    print(
        combined["Compound"]
        .value_counts()
        .to_string()
    )

    return combined


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    process_all_races()