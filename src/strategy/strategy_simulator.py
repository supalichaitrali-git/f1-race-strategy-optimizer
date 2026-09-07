"""
F1 race strategy simulator.

Simulates a race using tire compounds and custom stint lengths.
"""

from dataclasses import dataclass

from src.strategy.tire_model import TireModel


@dataclass
class StrategyResult:
    """Result of simulating one race strategy."""

    strategy: list[str]
    stint_lengths: list[int]
    total_time: float
    pit_stops: int


class StrategySimulator:
    """Simulate F1 race strategies."""

    def __init__(
        self,
        total_laps: int,
        base_lap_time: float = 90.0,
        pit_stop_loss: float = 22.0,
        min_stint_laps: int = 5,
        max_stint_laps: int = 35,
    ):
        self.total_laps = total_laps
        self.base_lap_time = base_lap_time
        self.pit_stop_loss = pit_stop_loss
        self.min_stint_laps = min_stint_laps
        self.max_stint_laps = max_stint_laps
        self.tire_model = TireModel()

    def _get_stint_lengths(self, number_of_stints: int) -> list[int]:
        """Create balanced stint lengths."""

        if number_of_stints <= 0:
            raise ValueError("Number of stints must be positive.")

        if self.total_laps < number_of_stints * self.min_stint_laps:
            raise ValueError("Too many stints for the race distance.")

        if self.total_laps > number_of_stints * self.max_stint_laps:
            raise ValueError("Too few stints for the race distance.")

        base_length = self.total_laps // number_of_stints
        remainder = self.total_laps % number_of_stints

        return [
            base_length + (1 if i < remainder else 0)
            for i in range(number_of_stints)
        ]

    def validate_stints(
        self,
        strategy: list[str],
        stint_lengths: list[int],
    ) -> None:
        """Validate a strategy with custom stint lengths."""

        if not strategy:
            raise ValueError("Strategy cannot be empty.")

        if len(strategy) != len(stint_lengths):
            raise ValueError(
                "Strategy and stint lengths must have the same number of stints."
            )

        if sum(stint_lengths) != self.total_laps:
            raise ValueError(
                "Stint lengths must add up to the total race laps."
            )

        for compound, stint_length in zip(strategy, stint_lengths):

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

            max_age = self.tire_model.max_tire_age[compound]

            if stint_length - 1 > max_age:
                raise ValueError(
                    f"{compound} stint of {stint_length} laps "
                    f"exceeds maximum tire age of {max_age} laps."
                )

    def simulate(
        self,
        strategy: list[str],
        stint_lengths: list[int] | None = None,
    ) -> StrategyResult:
        """
        Simulate a race using the supplied tire strategy.

        If stint_lengths are not provided, the race distance is divided
        into balanced stints for backwards compatibility.
        """

        if not strategy:
            raise ValueError("Strategy cannot be empty.")

        if stint_lengths is None:
            stint_lengths = self._get_stint_lengths(len(strategy))

        self.validate_stints(strategy, stint_lengths)

        pit_stops = len(strategy) - 1
        total_time = 0.0

        for compound, stint_laps in zip(strategy, stint_lengths):

            tire_age = 0

            for _ in range(stint_laps):

                lap_time = self.tire_model.lap_time(
                    self.base_lap_time,
                    compound,
                    tire_age,
                )

                total_time += lap_time
                tire_age += 1

        total_time += pit_stops * self.pit_stop_loss

        return StrategyResult(
            strategy=strategy,
            stint_lengths=stint_lengths,
            total_time=total_time,
            pit_stops=pit_stops,
        )