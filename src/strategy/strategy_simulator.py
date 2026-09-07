"""
F1 race strategy simulator.

Uses the trained lap-time prediction model to estimate the
time required for every lap of a proposed tire strategy.
"""

from dataclasses import dataclass

from src.ml.predict import load_model, predict_lap_time
from src.strategy.tire_model import TireModel


@dataclass
class StrategyResult:
    """Result of simulating one race strategy."""

    strategy: list[str]
    stint_lengths: list[int]
    total_time: float
    pit_stops: int


class StrategySimulator:
    """Simulate F1 race strategies using the ML lap-time model."""

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

    def _get_stint_lengths(
        self,
        number_of_stints: int,
    ) -> list[int]:
        """Create balanced stint lengths."""

        if number_of_stints <= 0:
            raise ValueError(
                "Number of stints must be positive."
            )

        if (
            self.total_laps
            < number_of_stints * self.min_stint_laps
        ):
            raise ValueError(
                "Too many stints for the race distance."
            )

        if (
            self.total_laps
            > number_of_stints * self.max_stint_laps
        ):
            raise ValueError(
                "Too few stints for the race distance."
            )

        base_length = (
            self.total_laps
            // number_of_stints
        )

        remainder = (
            self.total_laps
            % number_of_stints
        )

        return [
            base_length
            + (1 if i < remainder else 0)
            for i in range(number_of_stints)
        ]

    def validate_stints(
        self,
        strategy: list[str],
        stint_lengths: list[int],
    ) -> None:
        """Validate a strategy with custom stint lengths."""

        if not strategy:
            raise ValueError(
                "Strategy cannot be empty."
            )

        if len(strategy) != len(stint_lengths):
            raise ValueError(
                "Strategy and stint lengths must have "
                "the same number of stints."
            )

        if sum(stint_lengths) != self.total_laps:
            raise ValueError(
                "Stint lengths must add up to the "
                "total race laps."
            )

        for compound, stint_length in zip(
            strategy,
            stint_lengths,
        ):
            compound = compound.upper()

            if compound not in self.tire_model.max_tire_age:
                raise ValueError(
                    f"Unknown tire compound: {compound}"
                )

            if stint_length < self.min_stint_laps:
                raise ValueError(
                    f"{compound} stint is too short: "
                    f"{stint_length} laps."
                )

            if stint_length > self.max_stint_laps:
                raise ValueError(
                    f"{compound} stint is too long: "
                    f"{stint_length} laps."
                )

            max_age = (
                self.tire_model.max_tire_age[
                    compound
                ]
            )

            if stint_length - 1 > max_age:
                raise ValueError(
                    f"{compound} stint of "
                    f"{stint_length} laps exceeds "
                    f"maximum tire age of "
                    f"{max_age} laps."
                )

    def _predict_lap(
        self,
        compound: str,
        tire_age: int,
        lap_number: int,
    ) -> float:
        """Predict the lap time for one race lap."""

        race_progress = (
            lap_number / self.total_laps
        )

        return predict_lap_time(
            model=self.model,
            compound=compound,
            tyre_age=tire_age,
            lap_number=lap_number,
            race_progress=race_progress,
            track_temp=self.track_temp,
            air_temp=self.air_temp,
            humidity=self.humidity,
            wind_speed=self.wind_speed,
            driver=self.driver,
            grand_prix=self.grand_prix,
            season=self.season,
        )

    def simulate(
        self,
        strategy: list[str],
        stint_lengths: list[int] | None = None,
    ) -> StrategyResult:
        """
        Simulate a complete race using the supplied strategy.

        The ML model predicts each lap time using the tire compound,
        tire age, race progress, weather, driver, circuit and season.
        """

        if not strategy:
            raise ValueError(
                "Strategy cannot be empty."
            )

        if stint_lengths is None:
            stint_lengths = (
                self._get_stint_lengths(
                    len(strategy)
                )
            )

        self.validate_stints(
            strategy,
            stint_lengths,
        )

        pit_stops = len(strategy) - 1
        total_time = 0.0

        lap_number = 1

        for compound, stint_laps in zip(
            strategy,
            stint_lengths,
        ):
            compound = compound.upper()

            tire_age = 1

            for _ in range(stint_laps):

                lap_time = self._predict_lap(
                    compound=compound,
                    tire_age=tire_age,
                    lap_number=lap_number,
                )

                total_time += lap_time

                tire_age += 1
                lap_number += 1

        total_time += (
            pit_stops
            * self.pit_stop_loss
        )

        return StrategyResult(
            strategy=strategy,
            stint_lengths=stint_lengths,
            total_time=total_time,
            pit_stops=pit_stops,
        )


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
        strategy=[
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
        f"Strategy: "
        f"{' -> '.join(result.strategy)}"
    )

    print(
        f"Stints: "
        f"{result.stint_lengths}"
    )

    print(
        f"Pit stops: "
        f"{result.pit_stops}"
    )

    print(
        f"Predicted race time: "
        f"{result.total_time:.2f} sec"
    )

    print(
        f"Predicted race time: "
        f"{result.total_time / 60:.2f} min"
    )

    print("=" * 70)