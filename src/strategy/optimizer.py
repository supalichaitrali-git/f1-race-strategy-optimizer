"""
F1 race strategy optimizer.

Generates and evaluates a large strategy space using the
cached ML predictions from StrategySimulator.
"""

from itertools import product

from src.strategy.strategy_simulator import StrategySimulator


class StrategyOptimizer:
    """Find and rank the fastest expected F1 race strategies."""

    def __init__(
        self,
        simulator: StrategySimulator,
        compounds: list[str] | None = None,
        min_stint_laps: int = 5,
        max_stint_laps: int = 35,
        max_stops: int = 3,
    ):
        self.simulator = simulator

        self.compounds = (
            compounds
            if compounds is not None
            else [
                "SOFT",
                "MEDIUM",
                "HARD",
            ]
        )

        self.compounds = [
            compound.upper()
            for compound in self.compounds
        ]

        self.min_stint_laps = min_stint_laps
        self.max_stint_laps = max_stint_laps
        self.max_stops = max_stops

        # Maximum physically valid tire age for each compound.
        self.compound_max_age = {
            "SOFT": 20,
            "MEDIUM": 30,
            "HARD": 40,
        }

    def _get_compound_max_stint(
        self,
        compound: str,
    ) -> int:
        """Return the maximum valid stint length for a compound."""

        compound = compound.upper()

        if compound not in self.compound_max_age:
            raise ValueError(
                f"Unsupported compound: {compound}"
            )

        return min(
            self.max_stint_laps,
            self.compound_max_age[compound],
        )

    def _generate_stint_lengths_for_compounds(
        self,
        total_laps: int,
        compounds: tuple[str, ...],
    ):
        """
        Generate valid stint lengths for a specific compound sequence.

        Each stint receives a maximum length based on its compound.
        """

        number_of_stints = len(compounds)

        maximum_lengths = [
            self._get_compound_max_stint(
                compound
            )
            for compound in compounds
        ]

        def recursive(
            stint_index: int,
            remaining_laps: int,
            current_lengths: list[int],
        ):
            """Recursively generate valid stint-length combinations."""

            # Last stint.
            if stint_index == number_of_stints - 1:

                final_length = remaining_laps

                if (
                    self.min_stint_laps
                    <= final_length
                    <= maximum_lengths[
                        stint_index
                    ]
                ):
                    yield current_lengths + [
                        final_length
                    ]

                return

            remaining_stints = (
                number_of_stints
                - stint_index
                - 1
            )

            # Minimum laps required by all remaining stints.
            min_remaining = (
                remaining_stints
                * self.min_stint_laps
            )

            # Maximum laps available from all remaining
            # compounds.
            max_remaining = sum(
                maximum_lengths[
                    stint_index + 1:
                ]
            )

            current_min = max(
                self.min_stint_laps,
                remaining_laps
                - max_remaining,
            )

            current_max = min(
                maximum_lengths[
                    stint_index
                ],
                remaining_laps
                - min_remaining,
            )

            if current_min > current_max:
                return

            for length in range(
                current_min,
                current_max + 1,
            ):
                yield from recursive(
                    stint_index + 1,
                    remaining_laps - length,
                    current_lengths + [
                        length
                    ],
                )

        yield from recursive(
            stint_index=0,
            remaining_laps=total_laps,
            current_lengths=[],
        )

    def generate_strategies(self):
        """
        Generate all valid compound/stint combinations.

        Each strategy is represented as:

        {
            "compounds": [...],
            "stint_lengths": [...]
        }

        Tire-age limits are respected for every compound.
        """

        strategies = []

        total_laps = self.simulator.total_laps

        for stops in range(
            1,
            self.max_stops + 1,
        ):
            number_of_stints = stops + 1

            compound_combinations = product(
                self.compounds,
                repeat=number_of_stints,
            )

            for compounds in compound_combinations:

                # Skip unsupported compounds.
                if any(
                    compound
                    not in self.compound_max_age
                    for compound in compounds
                ):
                    continue

                stint_lengths = (
                    self._generate_stint_lengths_for_compounds(
                        total_laps=total_laps,
                        compounds=compounds,
                    )
                )

                for lengths in stint_lengths:

                    strategies.append(
                        {
                            "compounds": list(
                                compounds
                            ),
                            "stint_lengths": list(
                                lengths
                            ),
                        }
                    )

        return strategies

    def evaluate(
        self,
        strategies: list[dict],
    ):
        """Evaluate all generated strategies."""

        results = []

        total_strategies = len(
            strategies
        )

        for index, strategy in enumerate(
            strategies,
            start=1,
        ):

            simulation = (
                self.simulator.simulate(
                    compounds=strategy[
                        "compounds"
                    ],
                    stint_lengths=strategy[
                        "stint_lengths"
                    ],
                )
            )

            results.append(
                {
                    "compounds": simulation[
                        "compounds"
                    ],
                    "stint_lengths": simulation[
                        "stint_lengths"
                    ],
                    "pit_stops": simulation[
                        "pit_stops"
                    ],
                    "total_time": simulation[
                        "total_time"
                    ],
                    "total_time_minutes": simulation[
                        "total_time_minutes"
                    ],
                    "laps": simulation[
                        "laps"
                    ],
                }
            )

            if index % 5000 == 0:
                print(
                    f"Evaluated "
                    f"{index:,} / "
                    f"{total_strategies:,} strategies"
                )

        results.sort(
            key=lambda result: result[
                "total_time"
            ]
        )

        return results

    def optimize(self):
        """Generate, evaluate, and rank all valid strategies."""

        print(
            "Generating strategies..."
        )

        strategies = (
            self.generate_strategies()
        )

        print(
            f"Total strategies generated: "
            f"{len(strategies):,}"
        )

        if not strategies:
            raise ValueError(
                "No valid strategies could be generated "
                "for the configured race length and "
                "stint constraints."
            )

        print(
            "Evaluating strategies..."
        )

        results = self.evaluate(
            strategies
        )

        return results


if __name__ == "__main__":

    import time

    print()
    print("=" * 70)
    print("F1 RACE STRATEGY OPTIMIZER")
    print("=" * 70)

    start_time = time.perf_counter()

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
        simulator=simulator,
        max_stops=2,
    )

    results = optimizer.optimize()

    best_strategy = results[0]

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("BEST STRATEGY")
    print("=" * 70)

    print(
        "Strategy: "
        + " -> ".join(
            best_strategy["compounds"]
        )
    )

    print(
        f"Stints: "
        f"{best_strategy['stint_lengths']}"
    )

    print(
        f"Pit stops: "
        f"{best_strategy['pit_stops']}"
    )

    print(
        f"Predicted race time: "
        f"{best_strategy['total_time']:.2f} sec"
    )

    print(
        f"Predicted race time: "
        f"{best_strategy['total_time_minutes']:.2f} min"
    )

    print(
        f"Optimization time: "
        f"{elapsed:.2f} sec"
    )

    print()
    print("=" * 70)
    print("TOP 10 STRATEGIES")
    print("=" * 70)

    for rank, strategy in enumerate(
        results[:10],
        start=1,
    ):
        strategy_name = " -> ".join(
            strategy["compounds"]
        )

        print(
            f"#{rank:<2} "
            f"{strategy_name:<25} "
            f"Stints: "
            f"{strategy['stint_lengths']} "
            f"| Stops: "
            f"{strategy['pit_stops']} "
            f"| Time: "
            f"{strategy['total_time']:.2f} sec"
        )

    print("=" * 70)