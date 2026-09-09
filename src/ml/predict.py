"""
Prediction interface for the F1 Race Strategy Optimizer.

Provides single and batch lap-time prediction using the trained
production ML model.
"""

from pathlib import Path

import joblib
import pandas as pd


MODEL_FILE = Path(
    "models/lap_time_model.joblib"
)

SUPPORTED_COMPOUNDS = {
    "SOFT",
    "MEDIUM",
    "HARD",
}


def load_model():
    """Load the trained lap-time model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


def _validate_prediction_inputs(
    compound: str,
    tyre_age: int,
    lap_number: int,
    race_progress: float,
) -> str:
    """Validate common prediction inputs."""

    compound = compound.upper()

    if compound not in SUPPORTED_COMPOUNDS:
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

    return compound


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
    Predict one lap time.

    All inputs represent information available before
    the lap is driven.
    """

    compound = _validate_prediction_inputs(
        compound,
        tyre_age,
        lap_number,
        race_progress,
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
                "Compound": compound,
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


def predict_lap_times_batch(
    model,
    predictions: list[dict],
) -> list[float]:
    """
    Predict multiple lap times in one model call.

    Each dictionary must contain the same features used
    by the production lap-time model.
    """

    if not predictions:
        raise ValueError(
            "Prediction list cannot be empty."
        )

    validated_predictions = []

    for item in predictions:

        compound = _validate_prediction_inputs(
            item["Compound"],
            int(item["TyreLife"]),
            int(item["LapNumber"]),
            float(item["RaceProgress"]),
        )

        validated_predictions.append(
            {
                "TyreLife": int(
                    item["TyreLife"]
                ),
                "LapNumber": int(
                    item["LapNumber"]
                ),
                "RaceProgress": float(
                    item["RaceProgress"]
                ),
                "TrackTemp": float(
                    item["TrackTemp"]
                ),
                "AirTemp": float(
                    item["AirTemp"]
                ),
                "Humidity": float(
                    item["Humidity"]
                ),
                "WindSpeed": float(
                    item["WindSpeed"]
                ),
                "Compound": compound,
                "Driver": item["Driver"],
                "GrandPrix": item["GrandPrix"],
                "Season": int(
                    item["Season"]
                ),
            }
        )

    input_data = pd.DataFrame(
        validated_predictions
    )

    predictions = model.predict(
        input_data
    )

    return [
        float(value)
        for value in predictions
    ]


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
    Load the trained model and predict one lap time.
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

    model = load_model()

    single_prediction = predict_lap_time(
        model=model,
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

    batch_predictions = predict_lap_times_batch(
        model=model,
        predictions=[
            {
                "TyreLife": 1,
                "LapNumber": 1,
                "RaceProgress": 1 / 53,
                "TrackTemp": 35.0,
                "AirTemp": 25.0,
                "Humidity": 50.0,
                "WindSpeed": 3.0,
                "Compound": "MEDIUM",
                "Driver": "VER",
                "GrandPrix": "Italian Grand Prix",
                "Season": 2025,
            },
            {
                "TyreLife": 2,
                "LapNumber": 2,
                "RaceProgress": 2 / 53,
                "TrackTemp": 35.0,
                "AirTemp": 25.0,
                "Humidity": 50.0,
                "WindSpeed": 3.0,
                "Compound": "MEDIUM",
                "Driver": "VER",
                "GrandPrix": "Italian Grand Prix",
                "Season": 2025,
            },
            {
                "TyreLife": 3,
                "LapNumber": 3,
                "RaceProgress": 3 / 53,
                "TrackTemp": 35.0,
                "AirTemp": 25.0,
                "Humidity": 50.0,
                "WindSpeed": 3.0,
                "Compound": "MEDIUM",
                "Driver": "VER",
                "GrandPrix": "Italian Grand Prix",
                "Season": 2025,
            },
        ],
    )

    print()
    print("=" * 70)
    print("PREDICTION INTERFACE TEST")
    print("=" * 70)

    print(
        f"Single prediction: "
        f"{single_prediction:.3f} sec"
    )

    print(
        "Batch predictions: "
        + ", ".join(
            f"{value:.3f}"
            for value in batch_predictions
        )
        + " sec"
    )

    print("=" * 70)