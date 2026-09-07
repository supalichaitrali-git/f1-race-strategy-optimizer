"""
Multi-season FastF1 data loader.

Automatically discovers completed F1 Grand Prix races for a season,
downloads lap and weather data, and skips races that have already
been downloaded.

Testing sessions, pre-season tests, and non-race track sessions
are excluded.
"""

from datetime import datetime
from pathlib import Path

import fastf1

from src.data_pipeline.fastf1_loader import load_race
from src.data_pipeline.fastf1_loader import save_laps
from src.data_pipeline.fastf1_loader import save_weather


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")


# -------------------------------------------------------------------
# Discover completed races
# -------------------------------------------------------------------

def get_completed_races(
    year: int,
) -> list[tuple[str, str]]:
    """
    Get completed F1 Grand Prix races for a season.

    Parameters
    ----------
    year : int
        F1 season year.

    Returns
    -------
    list[tuple[str, str]]
        List of (event name, safe file name).
    """

    schedule = fastf1.get_event_schedule(year)

    today = datetime.now().date()

    races = []

    for _, event in schedule.iterrows():

        event_name = str(event["EventName"])
        event_name_lower = event_name.lower()

        event_date = event["EventDate"]

        # -----------------------------------------------------------
        # Convert pandas Timestamp to Python date
        # -----------------------------------------------------------

        if hasattr(event_date, "date"):
            event_date = event_date.date()

        # -----------------------------------------------------------
        # Ignore future events
        # -----------------------------------------------------------

        if event_date > today:
            continue

        # -----------------------------------------------------------
        # Ignore testing and non-race sessions
        # -----------------------------------------------------------

        excluded_keywords = [
            "testing",
            "pre-season test",
            "pre season test",
            "track session",
            "test",
        ]

        if any(
            keyword in event_name_lower
            for keyword in excluded_keywords
        ):
            continue

        # -----------------------------------------------------------
        # Create filesystem-safe file name
        # -----------------------------------------------------------

        safe_name = (
            event_name_lower
            .replace(" grand prix", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

        races.append(
            (
                event_name,
                safe_name,
            )
        )

    return races


# -------------------------------------------------------------------
# Download one race
# -------------------------------------------------------------------

def download_race(
    year: int,
    event: str,
    safe_name: str,
) -> bool:
    """
    Download lap and weather data for one Grand Prix.

    Returns
    -------
    bool
        True if successful or already downloaded.
        False if the download failed.
    """

    output_dir = RAW_DATA_DIR / str(year)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lap_file = (
        output_dir
        / f"{year}_{safe_name}_laps.csv"
    )

    weather_file = (
        output_dir
        / f"{year}_{safe_name}_weather.csv"
    )

    # ---------------------------------------------------------------
    # Skip already downloaded races
    # ---------------------------------------------------------------

    if lap_file.exists() and weather_file.exists():

        print()
        print(f"Skipping {year} {event}")
        print("Data already exists.")

        return True

    # ---------------------------------------------------------------
    # Download race
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Downloading {year} {event}")
    print("=" * 60)

    try:

        race = load_race(
            year,
            event,
        )

        # -----------------------------------------------------------
        # Save lap data
        # -----------------------------------------------------------

        save_laps(
            race,
            str(lap_file),
        )

        # -----------------------------------------------------------
        # Save weather data
        # -----------------------------------------------------------

        save_weather(
            race,
            str(weather_file),
        )

        # -----------------------------------------------------------
        # Display summary
        # -----------------------------------------------------------

        print()
        print(f"Race: {race.event.EventName}")
        print(f"Drivers: {len(race.drivers)}")
        print(f"Lap records: {len(race.laps)}")
        print(
            f"Weather records: "
            f"{len(race.weather_data)}"
        )
        print("Download completed.")

        return True

    except Exception as error:

        print()
        print("=" * 60)
        print(f"FAILED: {year} {event}")
        print("=" * 60)
        print(f"Reason: {error}")

        return False


# -------------------------------------------------------------------
# Download one complete season
# -------------------------------------------------------------------

def download_season(
    year: int,
) -> None:
    """
    Discover and download all completed Grand Prix races
    for one F1 season.
    """

    print()
    print("#" * 60)
    print(f"DISCOVERING {year} F1 RACES")
    print("#" * 60)

    races = get_completed_races(year)

    print()
    print(
        f"Completed races found: "
        f"{len(races)}"
    )

    successful = 0
    failed = 0

    # ---------------------------------------------------------------
    # Download each race
    # ---------------------------------------------------------------

    for number, (event, safe_name) in enumerate(
        races,
        start=1,
    ):

        print()
        print(
            f"Race {number}/{len(races)}"
        )

        success = download_race(
            year,
            event,
            safe_name,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # ---------------------------------------------------------------
    # Season summary
    # ---------------------------------------------------------------

    print()
    print("#" * 60)
    print(f"{year} SEASON DOWNLOAD SUMMARY")
    print("#" * 60)

    print(
        f"Successful/skipped: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total races: {len(races)}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    SEASONS = [
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
        2026,
    ]

    for year in SEASONS:

        download_season(year)