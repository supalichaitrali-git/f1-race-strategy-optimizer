"""
Real-world F1 tire degradation analysis.

Uses the processed FastF1 dataset to estimate how lap time
changes with tire age for SOFT, MEDIUM, and HARD compounds.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATA_FILE = Path(
    "data/processed/f1_race_strategy_dataset.csv"
)

DRY_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the processed F1 dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


# -------------------------------------------------------------------
# Prepare dry-tire data
# -------------------------------------------------------------------

def prepare_dry_tire_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare clean dry-tire laps for degradation analysis.
    """

    data = df[
        df["Compound"].isin(DRY_COMPOUNDS)
    ].copy()

    # ---------------------------------------------------------------
    # Required values
    # ---------------------------------------------------------------

    data = data.dropna(
        subset=[
            "LapTimeSec",
            "TyreLife",
            "Compound",
            "Driver",
            "GrandPrix",
            "Season",
        ]
    )

    # ---------------------------------------------------------------
    # Tire age must be positive
    # ---------------------------------------------------------------

    data = data[
        data["TyreLife"] >= 1
    ]

    # ---------------------------------------------------------------
    # Remove extreme tire ages.
    #
    # Very old tires are rare and can represent unusual
    # race circumstances rather than normal degradation.
    # ---------------------------------------------------------------

    data = data[
        data["TyreLife"] <= 40
    ]

    # ---------------------------------------------------------------
    # Remove extremely slow laps.
    #
    # This is deliberately conservative. We don't want safety
    # car / yellow flag / major traffic laps dominating the model.
    # ---------------------------------------------------------------

    data = data[
        data["LapTimeSec"] <= 120
    ]

    return data.reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# Calculate baseline lap time
# -------------------------------------------------------------------

def calculate_baseline(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a baseline lap time for each
    driver / race / compound combination.

    The baseline represents the typical pace of that
    combination before tire degradation becomes dominant.
    """

    result = data.copy()

    baseline = (
        result
        .groupby(
            [
                "Season",
                "GrandPrix",
                "Driver",
                "Compound",
            ]
        )["LapTimeSec"]
        .transform("median")
    )

    result["BaselineLapTime"] = baseline

    # ---------------------------------------------------------------
    # Calculate lap time relative to baseline.
    # ---------------------------------------------------------------

    result["RelativeLapTime"] = (
        result["LapTimeSec"]
        - result["BaselineLapTime"]
    )

    return result


# -------------------------------------------------------------------
# Fit degradation model
# -------------------------------------------------------------------

def fit_degradation_models(
    data: pd.DataFrame,
) -> dict[str, LinearRegression]:
    """
    Fit a linear tire degradation model for each dry compound.

    Model:

        RelativeLapTime = intercept + degradation_rate * TyreLife

    The slope represents estimated seconds of degradation
    per additional tire-life lap.
    """

    models = {}

    for compound in DRY_COMPOUNDS:

        compound_data = data[
            data["Compound"] == compound
        ].copy()

        X = compound_data[
            ["TyreLife"]
        ]

        y = compound_data[
            "RelativeLapTime"
        ]

        model = LinearRegression()

        model.fit(
            X,
            y,
        )

        models[compound] = model

    return models


# -------------------------------------------------------------------
# Print model results
# -------------------------------------------------------------------

def print_model_results(
    data: pd.DataFrame,
    models: dict[str, LinearRegression],
) -> None:
    """Print degradation statistics."""

    print()
    print("=" * 70)
    print("REAL F1 TIRE DEGRADATION ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Usable dry-tire laps: {len(data):,}"
    )

    print()

    for compound in DRY_COMPOUNDS:

        compound_data = data[
            data["Compound"] == compound
        ]

        model = models[compound]

        degradation_rate = model.coef_[0]

        intercept = model.intercept_

        r_squared = model.score(
            compound_data[["TyreLife"]],
            compound_data["RelativeLapTime"],
        )

        print("-" * 70)
        print(f"Compound: {compound}")

        print(
            f"Laps used: {len(compound_data):,}"
        )

        print(
            f"Degradation rate: "
            f"{degradation_rate:.4f} sec/lap"
        )

        print(
            f"Intercept: "
            f"{intercept:.4f} sec"
        )

        print(
            f"R² score: "
            f"{r_squared:.4f}"
        )

        print(
            f"Average tire age: "
            f"{compound_data['TyreLife'].mean():.2f}"
        )

        print(
            f"Maximum tire age: "
            f"{compound_data['TyreLife'].max():.0f}"
        )


# -------------------------------------------------------------------
# Tire age summary
# -------------------------------------------------------------------

def print_age_summary(
    data: pd.DataFrame,
) -> None:
    """Print lap counts by tire age."""

    print()
    print("=" * 70)
    print("TIRE AGE DISTRIBUTION")
    print("=" * 70)

    summary = (
        data
        .groupby(
            [
                "Compound",
                "TyreLife",
            ]
        )
        .size()
        .reset_index(
            name="LapCount"
        )
    )

    for compound in DRY_COMPOUNDS:

        compound_summary = summary[
            summary["Compound"] == compound
        ]

        print()
        print(
            compound_summary
            .head(15)
            .to_string(index=False)
        )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run tire degradation analysis."""

    print()
    print("Loading F1 dataset...")

    df = load_data()

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    print()
    print("Preparing dry-tire data...")

    dry_data = prepare_dry_tire_data(
        df
    )

    print(
        f"Dry-tire rows after filtering: "
        f"{len(dry_data):,}"
    )

    print()
    print("Calculating baseline lap times...")

    dry_data = calculate_baseline(
        dry_data
    )

    print()
    print("Fitting degradation models...")

    models = fit_degradation_models(
        dry_data
    )

    print_model_results(
        dry_data,
        models,
    )

    print_age_summary(
        dry_data
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    main()