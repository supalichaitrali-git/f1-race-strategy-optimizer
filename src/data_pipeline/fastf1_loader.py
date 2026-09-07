"""
FastF1 data loader for the F1 Race Strategy Optimizer.

Loads historical F1 race lap and weather data
and stores local CSV copies.
"""

from pathlib import Path

import fastf1


# -------------------------------------------------------------------
# FastF1 cache
# -------------------------------------------------------------------

CACHE_DIR = Path("data/raw/fastf1_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


# -------------------------------------------------------------------
# Load race
# -------------------------------------------------------------------

def load_race(
    year: int = 2024,
    event: str = "Monza",
):
    """
    Load a historical F1 race session.

    Parameters
    ----------
    year : int
        F1 season year.

    event : str
        Grand Prix event name or circuit name.

    Returns
    -------
    fastf1.core.Session
        Loaded race session.
    """

    print(f"Loading {year} {event} Grand Prix...")

    session = fastf1.get_session(
        year,
        event,
        "R",
    )

    session.load(
        telemetry=False,
        weather=True,
        messages=False,
    )

    print("Race loaded successfully.")

    return session


# -------------------------------------------------------------------
# Save lap data
# -------------------------------------------------------------------

def save_laps(
    session,
    output_file: str = "data/raw/2024_monza_laps.csv",
) -> None:
    """
    Save race lap data to CSV.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    laps = session.laps.copy()

    laps.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(laps)} lap records.")
    print(f"File: {output_path}")


# -------------------------------------------------------------------
# Save weather data
# -------------------------------------------------------------------

def save_weather(
    session,
    output_file: str = "data/raw/2024_monza_weather.csv",
) -> None:
    """
    Save race weather data to CSV.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather = session.weather_data.copy()

    weather.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(weather)} weather records.")
    print(f"File: {output_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    YEAR = 2024
    EVENT = "Monza"

    race = load_race(
        YEAR,
        EVENT,
    )

    print()
    print("Race:", race.event.EventName)
    print("Year:", race.event.EventDate.year)
    print("Drivers:", len(race.drivers))
    print("Lap records:", len(race.laps))
    print("Weather records:", len(race.weather_data))

    print()

    save_laps(
        race,
        f"data/raw/{YEAR}_monza_laps.csv",
    )

    save_weather(
        race,
        f"data/raw/{YEAR}_monza_weather.csv",
    )