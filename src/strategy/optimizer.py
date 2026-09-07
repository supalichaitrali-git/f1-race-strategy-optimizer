"""
F1 race strategy optimizer.

Generates, evaluates, ranks, and returns the best tire strategies
using the ML-driven race strategy simulator.
"""

from itertools import product

from src.strategy.strategy_simulator import StrategyResult
from src.strategy.strategy_simulator import StrategySimulator


class StrategyOptimizer:
    """Generate and evaluate possible F1 race strategies."""

    def __init__(self, simulator: StrategySimulator):
        self.simulator = simulator

    def generate_strategies(
        self,
        compounds: list[str] | None = None,
        max_stops: int = 2,
    ) -> list[tuple[list[str], list[int]]]:
        """Generate tire strategies with variable stint lengths."""

        if compounds is None:
            compounds = [
                "SOFT",
                "MEDIUM",
                "HARD",
            ]

        strategies = []

        for stops in range(1, max_stops + 1):

            number_of_stints = stops + 1

            for strategy in product(
                compounds,
                repeat=number_of_stints,
            ):

                min_total = (
                    number_of_stints
                    * self.simulator.min_stint_laps
                )

                max_total = (
                    number_of_stints
                    * self.simulator.max_stint_laps
                )

                if not (
                    min_total
                    <= self.simulator.total_laps
                    <= max_total
                ):
                    continue

                self._generate_stint_lengths(
                    number_of_stints,
                    strategies,
                    list(strategy),
                )

        return strategies

    def _generate_stint_lengths(
        self,
        number_of_stints: int,
        strategies: list[
            tuple[list[str], list[int]]
        ],
        strategy: list[str],
    ) -> None:
        """Generate every valid combination of stint lengths."""

        min_laps = self.simulator.min_stint_laps
        max_laps = self.simulator.max_stint_laps
        total_laps = self.simulator.total_laps

        def generate(
            remaining_laps: int,
            remaining_stints: int,
            lengths: list[int],
        ) -> None:

            if remaining_stints == 1:

                if (
                    min_laps
                    <= remaining_laps
                    <= max_laps
                ):
                    strategies.append(
                        (
                            strategy,
                            lengths
                            + [remaining_laps],
                        )
                    )

                return

            minimum_remaining = (
                (remaining_stints - 1)
                * min_laps
            )

            maximum_remaining = (
                (remaining_stints - 1)
                * max_laps
            )

            for stint_length in range(
                min_laps,
                max_laps + 1,
            ):

                remaining = (
                    remaining_laps
                    - stint_length
                )

                if (
                    minimum_remaining
                    <= remaining
                    <= maximum_remaining
                ):
                    generate(
                        remaining,
                        remaining_stints - 1,
                        lengths
                        + [stint_length],
                    )

        generate(
            total_laps,
            number_of_stints,
            [],
        )

    def evaluate(
        self,
        strategies: list[
            tuple[list[str], list[int]]
        ],
    ) -> list[StrategyResult]:
        """Evaluate all valid strategies."""

        if not strategies:
            raise ValueError(
                "No strategies provided."
            )

        results = []

        for strategy, stint_lengths in strategies:

            try:
                result = self.simulator.simulate(
                    strategy,
                    stint_lengths,
                )

                results.append(result)

            except ValueError:
                continue

        if not results:
            raise ValueError(
                "No valid strategies available."
            )

        return sorted(
            results,
            key=lambda result: result.total_time,
        )

    def optimize(
        self,
        strategies: list[
            tuple[list[str], list[int]]
        ],
    ) -> StrategyResult:
        """Return the fastest valid strategy."""

        results = self.evaluate(strategies)

        return results[0]


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

    optimizer = StrategyOptimizer(
        simulator
    )

    print()
    print("=" * 70)
    print("F1 RACE STRATEGY OPTIMIZER")
    print("=" * 70)

    print()
    print("Generating strategies...")

    strategies = optimizer.generate_strategies(
        compounds=[
            "SOFT",
            "MEDIUM",
            "HARD",
        ],
        max_stops=2,
    )

    print(
        f"Total strategies generated: "
        f"{len(strategies):,}"
    )

    print()
    print("Evaluating strategies...")

    results = optimizer.evaluate(
        strategies
    )

    print(
        f"Strategies evaluated: "
        f"{len(results):,}"
    )

    print()
    print("=" * 70)
    print("TOP 10 STRATEGIES")
    print("=" * 70)

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):

        strategy_name = " -> ".join(
            result.strategy
        )

        print(
            f"{rank:2}. "
            f"{strategy_name:<25} "
            f"Stints={str(result.stint_lengths):<12} "
            f"Stops={result.pit_stops} "
            f"Time={result.total_time:.2f}s"
        )

    best = results[0]

    print()
    print("=" * 70)
    print("🏆 OPTIMAL STRATEGY")
    print("=" * 70)

    print(
        f"Strategy      : "
        f"{' -> '.join(best.strategy)}"
    )

    print(
        f"Stint lengths : "
        f"{best.stint_lengths}"
    )

    print(
        f"Pit stops     : "
        f"{best.pit_stops}"
    )

    print(
        f"Race time     : "
        f"{best.total_time:.2f} sec"
    )

    print(
        f"Race time     : "
        f"{best.total_time / 60:.2f} min"
    )

    print("=" * 70)