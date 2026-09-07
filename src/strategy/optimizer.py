"""
F1 race strategy optimizer.

Generates and evaluates possible tire strategies
with variable stint lengths.
"""

from itertools import product

from src.strategy.strategy_simulator import StrategyResult
from src.strategy.strategy_simulator import StrategySimulator


class StrategyOptimizer:
    """Generate and evaluate race strategies."""

    def __init__(self, simulator: StrategySimulator):
        self.simulator = simulator

    def generate_strategies(
        self,
        compounds: list[str] | None = None,
        max_stops: int = 2,
    ) -> list[tuple[list[str], list[int]]]:
        """Generate tire strategies with variable stint lengths."""

        if compounds is None:
            compounds = ["SOFT", "MEDIUM", "HARD"]

        strategies = []

        for stops in range(1, max_stops + 1):

            stints = stops + 1

            for strategy in product(compounds, repeat=stints):

                min_laps = stints * self.simulator.min_stint_laps
                max_laps = stints * self.simulator.max_stint_laps

                if not (
                    min_laps
                    <= self.simulator.total_laps
                    <= max_laps
                ):
                    continue

                self._generate_stint_lengths(
                    stints,
                    strategies,
                    list(strategy),
                )

        return strategies

    def _generate_stint_lengths(
        self,
        number_of_stints: int,
        strategies: list,
        strategy: list[str],
    ) -> None:
        """Generate possible variable stint lengths."""

        min_laps = self.simulator.min_stint_laps
        max_laps = self.simulator.max_stint_laps
        total_laps = self.simulator.total_laps

        def generate(
            remaining_laps: int,
            remaining_stints: int,
            lengths: list[int],
        ) -> None:

            if remaining_stints == 1:

                if min_laps <= remaining_laps <= max_laps:
                    strategies.append(
                        (
                            strategy,
                            lengths + [remaining_laps],
                        )
                    )

                return

            minimum_remaining = (
                (remaining_stints - 1) * min_laps
            )

            maximum_remaining = (
                (remaining_stints - 1) * max_laps
            )

            for stint_length in range(
                min_laps,
                max_laps + 1,
            ):

                remaining = remaining_laps - stint_length

                if (
                    minimum_remaining
                    <= remaining
                    <= maximum_remaining
                ):
                    generate(
                        remaining,
                        remaining_stints - 1,
                        lengths + [stint_length],
                    )

        generate(
            total_laps,
            number_of_stints,
            [],
        )

    def optimize(
        self,
        strategies: list[tuple[list[str], list[int]]],
    ) -> StrategyResult:
        """Return the fastest valid strategy."""

        if not strategies:
            raise ValueError("No strategies provided.")

        valid_results = []

        for strategy, stint_lengths in strategies:

            try:
                result = self.simulator.simulate(
                    strategy,
                    stint_lengths,
                )

                valid_results.append(result)

            except ValueError:
                continue

        if not valid_results:
            raise ValueError("No valid strategies available.")

        return min(
            valid_results,
            key=lambda result: result.total_time,
        )