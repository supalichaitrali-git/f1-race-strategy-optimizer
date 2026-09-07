"""
F1 race strategy optimizer.

Generates and evaluates possible tire strategies.
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
    ) -> list[list[str]]:
        """Generate possible tire strategies."""

        if compounds is None:
            compounds = ["SOFT", "MEDIUM", "HARD"]

        strategies = []

        for stops in range(1, max_stops + 1):
            stints = stops + 1

            for strategy in product(compounds, repeat=stints):
                strategies.append(list(strategy))

        return strategies

    def optimize(
        self,
        strategies: list[list[str]],
    ) -> StrategyResult:
        """Return the fastest valid strategy."""

        if not strategies:
            raise ValueError("No strategies provided.")

        valid_results = []

        for strategy in strategies:
            try:
                result = self.simulator.simulate(strategy)
                valid_results.append(result)

            except ValueError:
                continue

        if not valid_results:
            raise ValueError("No valid strategies available.")

        return min(
            valid_results,
            key=lambda result: result.total_time,
        )
