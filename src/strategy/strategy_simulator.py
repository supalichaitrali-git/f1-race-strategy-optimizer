"""
F1 race strategy simulator.

Uses cached batch ML predictions so that strategy evaluation
does not repeatedly call the sklearn model for every lap.
"""

from pathlib import Path

from src.ml.predict import (
    load_model,
    predict_lap_times_batch,
)
from src.strategy.tire_model import TireModel


class StrategySimulator:
    """Simulate F1 race strategies using the trained lap-time model."""

    def __init__(
        self,
        total_laps: int,
        driver: str,
        grand_prix: str,
        season: int,
        track_temp: float,
        air_temp: float,
        humidity: float,
        wind_speed: float,
        pit_stop_loss: float = 22.0,
        min_stint_laps: int = 5,
        max_stint_laps: int = 35,
    ):
        self.total_laps = total_laps
        self.driver = driver
        self.grand_prix = grand_prix
        self.season = season

        self.track_temp = track_temp
        self.air_temp = air_temp
        self.humidity = humidity
        self.wind_speed = wind_speed

        self.pit_stop_loss = pit_stop_loss
        self.min_stint_laps = min_stint_laps
        self.max_stint_laps = max_stint_laps

        self.tire_model = TireModel()

        self.model = load_model()

        # Predictions are generated once and reused by every strategy.
        self._prediction_cache = {}

        self._build_prediction_cache()

    def _build_prediction_cache(self):
        """Precompute all valid lap-time predictions needed by the simulator."""

        predictions = []

        compounds = [
            "SOFT",
            "MEDIUM",
            "HARD",
        ]

        for compound in compounds:

            max_age = min(
                self.max_stint_laps,
                self.tire_model.max_tire_age[compound],
            )

            for lap_number in range(
                1,
                self.total_laps + 1,
            ):

                race_progress = (
                    lap_number / self.total_laps
                )

                for tyre_age in range(
                    1,
                    max_age + 1,
                ):

                    # A tire cannot be older than the current lap.
                    if tyre_age > lap_number:
                        continue

                    predictions.append(
                        {
                            "TyreLife": tyre_age,
                            "LapNumber": lap_number,
                            "RaceProgress": race_progress,
                            "TrackTemp": self.track_temp,
                            "AirTemp": self.air_temp,
                            "Humidity": self.humidity,
                            "WindSpeed": self.wind_speed,
                            "Compound": compound,
                            "Driver": self.driver,
                            "GrandPrix": self.grand_prix,
                            "Season": self.season,
                        }
                    )

        predicted_values = predict_lap_times_batch(
            model=self.model,
            predictions=predictions,
        )

        for item, predicted_time in zip(
            predictions,
            predicted_values,
        ):
            key = (
                item["Compound"],
                item["LapNumber"],
                item["TyreLife"],
            )

            self._prediction_cache[key] = predicted_time

        print(
            f"Prediction cache built: "
            f"{len(self._prediction_cache):,} entries"
        )

    def _get_stint_lengths(
        self,
        total_laps: int,
        number_of_stints: int,
    ) -> list[int]:
        """Create balanced stint lengths."""

        if number_of_stints <= 0:
            raise ValueError(
                "Number of stints must be positive."
            )

        minimum_laps = (
            number_of_stints
            * self.min_stint_laps
        )

        if total_laps < minimum_laps:
            raise ValueError(
                "Race is too short for the requested "
                "number of stints."
            )

        base_length = (
            total_laps // number_of_stints
        )

        remainder = (
            total_laps % number_of_stints
        )

        lengths = []

        for index in range(number_of_stints):

            length = base_length

            if index < remainder:
                length += 1

            lengths.append(length)

        return lengths

    def validate_stints(
        self,
        compounds: list[str],
        stint_lengths: list[int],
    ):
        """Validate compounds and stint lengths."""

        if len(compounds) != len(stint_lengths):
            raise ValueError(
                "Number of compounds must match "
                "number of stint lengths."
            )

        if sum(stint_lengths) != self.total_laps:
            raise ValueError(
                "Stint lengths must add up to "
                f"{self.total_laps} laps."
            )

        for compound, stint_length in zip(
            compounds,
            stint_lengths,
        ):

            compound = compound.upper()

            if compound not in {
                "SOFT",
                "MEDIUM",
                "HARD",
            }:
                raise ValueError(
                    f"Unsupported compound: {compound}"
                )

            if stint_length < self.min_stint_laps:
                raise ValueError(
                    f"Stint length {stint_length} is below "
                    f"minimum of {self.min_stint_laps} laps."
                )

            if stint_length > self.max_stint_laps:
                raise ValueError(
                    f"Stint length {stint_length} exceeds "
                    f"maximum of {self.max_stint_laps} laps."
                )

            if not self.tire_model.is_tire_age_valid(
                compound,
                stint_length,
            ):
                raise ValueError(
                    f"{compound} stint of {stint_length} "
                    "laps exceeds maximum tire age."
                )

    def _predict_lap(
        self,
        compound: str,
        tyre_age: int,
        lap_number: int,
    ) -> float:
        """Retrieve a precomputed lap-time prediction."""

        key = (
            compound.upper(),
            lap_number,
            tyre_age,
        )

        if key not in self._prediction_cache:
            raise KeyError(
                f"No cached prediction for "
                f"{key}"
            )

        return self._prediction_cache[key]

    def simulate(
        self,
        compounds: list[str],
        stint_lengths: list[int],
    ) -> dict:
        """Simulate one complete race strategy."""

        self.validate_stints(
            compounds,
            stint_lengths,
        )

        total_time = 0.0
        lap_records = []

        current_lap = 1

        for stint_index, (
            compound,
            stint_length,
        ) in enumerate(
            zip(
                compounds,
                stint_lengths,
            ),
            start=1,
        ):

            compound = compound.upper()

            for tyre_age in range(
                1,
                stint_length + 1,
            ):

                lap_number = current_lap

                predicted_lap_time = (
                    self._predict_lap(
                        compound=compound,
                        tyre_age=tyre_age,
                        lap_number=lap_number,
                    )
                )

                total_time += predicted_lap_time

                lap_records.append(
                    {
                        "Lap": lap_number,
                        "Stint": stint_index,
                        "Compound": compound,
                        "TyreAge": tyre_age,
                        "PredictedLapTime": predicted_lap_time,
                    }
                )

                current_lap += 1

            # Add pit-stop time between stints.
            if stint_index < len(compounds):
                total_time += self.pit_stop_loss

        return {
            "compounds": compounds,
            "stint_lengths": stint_lengths,
            "pit_stops": len(compounds) - 1,
            "total_time": total_time,
            "total_time_minutes": total_time / 60,
            "laps": lap_records,
        }


if __name__ == "__main__":

    simulator = StrategySimulator(
        total_laps=53,
        driver="VER",
        grand_prix="Italian Grand Prix",
        season=2025,
        track_temp=35.0,
        air_temp=25.0,
        humidity=50.0,
        wind_speed=3.0,
    )

    result = simulator.simulate(
        compounds=[
            "MEDIUM",
            "HARD",
        ],
        stint_lengths=[
            27,
            26,
        ],
    )

    print()
    print("=" * 70)
    print("STRATEGY SIMULATION TEST")
    print("=" * 70)

    print(
        "Strategy: "
        + " -> ".join(result["compounds"])
    )

    print(
        f"Stints: "
        f"{result['stint_lengths']}"
    )

    print(
        f"Pit stops: "
        f"{result['pit_stops']}"
    )

    print(
        f"Predicted race time: "
        f"{result['total_time']:.2f} sec"
    )

    print(
        f"Predicted race time: "
        f"{result['total_time_minutes']:.2f} min"
    )

    print("=" * 70)