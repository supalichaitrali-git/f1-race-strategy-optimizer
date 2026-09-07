"""
Tire performance and degradation model for the F1 Race Strategy Optimizer.
"""

from dataclasses import dataclass


@dataclass
class TireModel:
    """Model performance, degradation, and useful life of F1 tires."""

    degradation_rates: dict[str, float] = None
    compound_pace: dict[str, float] = None
    max_tire_age: dict[str, int] = None

    def __post_init__(self):
        if self.degradation_rates is None:
            self.degradation_rates = {
                "SOFT": 0.12,
                "MEDIUM": 0.08,
                "HARD": 0.05,
            }

        if self.compound_pace is None:
            self.compound_pace = {
                "SOFT": -0.50,
                "MEDIUM": 0.00,
                "HARD": 0.30,
            }

        if self.max_tire_age is None:
            self.max_tire_age = {
                "SOFT": 20,
                "MEDIUM": 30,
                "HARD": 40,
            }

    def _validate_compound(self, compound: str) -> str:
        """Validate and normalize tire compound."""

        compound = compound.upper()

        if compound not in self.degradation_rates:
            raise ValueError(f"Unknown tire compound: {compound}")

        return compound

    def degradation(self, compound: str, tire_age: int) -> float:
        """Calculate additional lap time caused by tire degradation."""

        compound = self._validate_compound(compound)

        if tire_age < 0:
            raise ValueError("Tire age cannot be negative.")

        return self.degradation_rates[compound] * tire_age

    def is_tire_age_valid(self, compound: str, tire_age: int) -> bool:
        """Check whether a tire age is within its useful operating range."""

        compound = self._validate_compound(compound)

        if tire_age < 0:
            return False

        return tire_age <= self.max_tire_age[compound]

    def lap_time(
        self,
        base_lap_time: float,
        compound: str,
        tire_age: int,
    ) -> float:
        """Estimate lap time using compound pace and degradation."""

        if base_lap_time <= 0:
            raise ValueError("Base lap time must be positive.")

        compound = self._validate_compound(compound)

        if not self.is_tire_age_valid(compound, tire_age):
            raise ValueError(
                f"{compound} tire age {tire_age} exceeds "
                f"maximum useful age of {self.max_tire_age[compound]} laps."
            )

        fresh_tire_pace = self.compound_pace[compound]

        tire_degradation = self.degradation(
            compound,
            tire_age,
        )

        return base_lap_time + fresh_tire_pace + tire_degradation
