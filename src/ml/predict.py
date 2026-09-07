"""
Prediction interface for the F1 Race Strategy Optimizer.

Provides a simple function for predicting lap time using the
trained production model.
"""

from pathlib import Path

import joblib
import pandas as pd


MODEL_FILE = Path(
    "models/lap_time_model.joblib"
)


def load_model():
    """Load the trained lap-time model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


def predict_lap_time(
    model,
    compound: str,
    tyre_age: int,
    lap_number: int,
    race_progress: float,
    track_temp: float,
    air_temp: float,
    humidity: float,
    wind_speed: float,
    driver: str,
    grand_prix: str,
    season: int,
) -> float:
    """
    Predict the expected lap time.

    All inputs represent information available before
    the lap is driven.
    """

    if compound.upper() not in {
        "SOFT",
        "MEDIUM",
        "HARD",
    }:
        raise ValueError(
            f"Unsupported compound: {compound}"
        )

    if tyre_age < 0:
        raise ValueError(
            "Tire age cannot be negative."
        )

    if lap_number <= 0:
        raise ValueError(
            "Lap number must be positive."
        )

    if not 0 <= race_progress <= 1:
        raise ValueError(
            "Race progress must be between 0 and 1."
        )

    input_data = pd.DataFrame(
        [
            {
                "TyreLife": tyre_age,
                "LapNumber": lap_number,
                "RaceProgress": race_progress,
                "TrackTemp": track_temp,
                "AirTemp": air_temp,
                "Humidity": humidity,
                "WindSpeed": wind_speed,
                "Compound": compound.upper(),
                "Driver": driver,
                "GrandPrix": grand_prix,
                "Season": season,
            }
        ]
    )

    prediction = model.predict(
        input_data
    )

    return float(
        prediction[0]
    )


def predict(
    compound: str,
    tyre_age: int,
    lap_number: int,
    race_progress: float,
    track_temp: float,
    air_temp: float,
    humidity: float,
    wind_speed: float,
    driver: str,
    grand_prix: str,
    season: int,
) -> float:
    """
    Load the trained model and predict lap time.
    """

    model = load_model()

    return predict_lap_time(
        model=model,
        compound=compound,
        tyre_age=tyre_age,
        lap_number=lap_number,
        race_progress=race_progress,
        track_temp=track_temp,
        air_temp=air_temp,
        humidity=humidity,
        wind_speed=wind_speed,
        driver=driver,
        grand_prix=grand_prix,
        season=season,
    )


if __name__ == "__main__":
    prediction = predict(
        compound="MEDIUM",
        tyre_age=5,
        lap_number=20,
        race_progress=0.25,
        track_temp=35.0,
        air_temp=25.0,
        humidity=50.0,
        wind_speed=3.0,
        driver="VER",
        grand_prix="Italian Grand Prix",
        season=2025,
    )

    print(
        f"Predicted lap time: "
        f"{prediction:.3f} seconds"
    )