from typing import Dict
import pandas as pd
import numpy as np
from tabulate import tabulate

class RIMWeightingPandas:
    def __init__(
        self,
        data: pd.DataFrame,
        spec: Dict,
        pre_weight: str = None,
        tolerance: float = 0.005,
        weight_col_name: str = 'rim_weight',
        target: float = None
    ):
        """
        Initialize the RIM weighting class.

        Parameters:
        - data: Pandas DataFrame with survey data.
        - spec: Dictionary with variable names as keys and target distributions as values.
          Each value is itself a dict { category_name: proportion or absolute_count }.
        - pre_weight: Column name containing existing weights (if None, defaults to 1.0 for all).
        - tolerance: Convergence threshold for RMS error.
        - weight_col_name: Name of the weight column in the DataFrame.
        - target: Desired total weighted sum. If set to None, defaults to the actual number of records.
        """
        self.data = data.copy(deep=True)
        self.spec = spec
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = len(self.data)  # fixed reference for proportion scaling
        # Set target to provided value or default to the number of records
        self.target = target if target is not None else self.total_sample

        # Validate the specification
        self.validate_spec()

        # Convert any proportions in spec to absolute counts (only once)
        self.convert_targets_to_counts()

        # Set pre-weight column
        if pre_weight is None or pre_weight not in self.data.columns:
            self.data["pre_weight"] = 1.0  # default to 1.0 if no pre_weight column provided
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight

        # Initialize rim weight column based on pre-weight (or default to 1.0)
        if pre_weight:
            self.data[self.weight_col_name] = self.data[self.pre_weight_col_name]
        else:
            self.data[self.weight_col_name] = 1.0

    def validate_spec(self):
        """
        Validates the specification dictionary (self.spec) against the dataset.

        Raises:
        - ValueError if variables in spec do not exist in data.
        - ValueError if any categories in spec do not exist in the corresponding variable in data.
        - ValueError if the sum of all target proportions for any variable does not equal 1.0
          (only checked if they look like proportions).
        """
        for var, targets in self.spec.items():
            # 1. Check if variable exists in data
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")

            # 2. Get unique categories present in the data
            unique_categories = set(self.data[var].dropna().unique())

            # 3. Check if all specified categories exist in the data
            missing_categories = [cat for cat in targets.keys() if cat not in unique_categories]
            if missing_categories:
                raise ValueError(
                    f"❌ Categories {missing_categories} in spec for variable '{var}' do not exist in the data."
                )

            # 4. If the user is passing proportions, ensure they sum to ~1.0
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                total_target = sum(targets.values())
                if not np.isclose(total_target, 1.0, atol=1e-6):
                    raise ValueError(
                        f"❌ Target proportions for '{var}' sum to {total_target:.6f}, but must sum to exactly 1.0."
                    )

    def convert_targets_to_counts(self):
        """
        Convert any proportions in self.spec to absolute counts (i.e., multiply by total_sample).
        This is done exactly once to avoid re-scaling on every iteration.
        """
        for var, targets in self.spec.items():
            # If the first value is between 0 and 1, assume these are proportions
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=12, min_weight=0.6, max_weight=1.4) -> pd.DataFrame:
        """
        Applies RIM weighting using an iterative approach with RMS-error-based convergence.

        Parameters:
        - max_iterations (int): Maximum number of iterations allowed for weight adjustments.
        - min_weight (float): Minimum allowable weight for each observation.
        - max_weight (float): Maximum allowable weight for each observation.

        Returns:
        - pd.DataFrame: The original DataFrame with the updated `rim_weight` column.
        """
        for iteration in range(max_iterations):
            # 1. Iteratively adjust weights for each variable based on the targets
            for var, targets in self.spec.items():
                # Current weighted totals by category
                current_totals = self.data.groupby(var)[self.weight_col_name].sum()

                # Compute adjustment factors = (target_count / observed_count)
                adjustment_factors = {}
                for cat, target_value in targets.items():
                    observed = current_totals.get(cat, 0)
                    if observed > 0:
                        adjustment_factors[cat] = target_value / observed
                    else:
                        # If no observations for the category, use a neutral factor of 1.0
                        adjustment_factors[cat] = 1.0

                # Apply the adjustment factor to each row for the variable
                self.data[self.weight_col_name] *= self.data[var].map(adjustment_factors).fillna(1.0)

            # 2. Normalize total weights to match the target sum
            total_weight_sum = self.data[self.weight_col_name].sum()
            if total_weight_sum > 0:
                scale_factor = self.target / total_weight_sum
                self.data[self.weight_col_name] *= scale_factor

            # 3. Clip weights to the range [min_weight, max_weight]
            self.data[self.weight_col_name] = np.clip(
                self.data[self.weight_col_name],
                min_weight,
                max_weight
            )

            # 4. Compute RMS Error across all variables
            rms_error = 0.0
            for var, targets in self.spec.items():
                weighted_totals = self.data.groupby(var)[self.weight_col_name].sum()
                for cat, target_value in targets.items():
                    observed = weighted_totals.get(cat, 0)
                    rms_error += (target_value - observed) ** 2

            # Average the squared errors and take the square root
            rms_error = np.sqrt(rms_error / len(self.spec))

            # 5. Print iteration diagnostics
            max_weight_value = self.data[self.weight_col_name].max()
            min_weight_value = self.data[self.weight_col_name].min()
            efficiency = self.weighting_efficiency()
            print(
                f"Iteration {iteration + 1}: "
                f"RMS Error = {rms_error:.6f}, "
                f"Efficiency = {efficiency:.2f}%, "
                f"Max Weight = {max_weight_value:.4f}, "
                f"Min Weight = {min_weight_value:.4f}"
            )

            # 6. Check if RMS error is below tolerance; if so, break out of the loop
            if rms_error < self.tolerance:
                print(f"✅ Converged by `RMS error < {self.tolerance}` in {iteration + 1} iterations.")
                break

        return self.data

    def weighting_efficiency(self):
        """
        Computes the RIM weighting efficiency as per the given formula:

            Efficiency (%) = 100 * ( Σ(Pj * Rj) )^2  /  ( Σ(Pj) * Σ(Pj * Rj^2) )

        where:
        - Pj is the pre-weight for each case (before RIM weighting).
        - Rj is the RIM weight for each case (after weighting).

        Returns:
            float: RIM weighting efficiency percentage.
        """
        Pj = self.data[self.pre_weight_col_name]  # pre-weight
        Rj = self.data[self.weight_col_name]       # RIM weight

        numerator = (np.sum(Pj * Rj)) ** 2
        denominator = np.sum(Pj) * np.sum(Pj * (Rj ** 2))

        if denominator == 0:
            return 0.0  # avoid division by zero

        return 100.0 * (numerator / denominator)

    def generate_summary(self):
        """
        Generates and prints a formatted summary of unweighted and weighted counts per variable.

        The summary includes:
        - Unweighted counts and percentages.
        - Weighted counts and percentages.
        - Minimum and maximum weights per category.
        """
        for var in self.spec.keys():
            # Calculate unweighted counts
            unweighted_df = self.data[var].value_counts(dropna=False).reset_index()
            unweighted_df.columns = [var, "Unweighted Count"]
            unweighted_df["Unweighted %"] = (
                unweighted_df["Unweighted Count"] / unweighted_df["Unweighted Count"].sum()
            ) * 100

            # Calculate weighted counts and weight range
            weighted_stats = self.data.groupby(var).agg(
                Weighted_Count=(self.weight_col_name, "sum"),
                Min_Weight=(self.weight_col_name, "min"),
                Max_Weight=(self.weight_col_name, "max")
            ).reset_index()

            weighted_stats["Weighted %"] = (
                weighted_stats["Weighted_Count"] / weighted_stats["Weighted_Count"].sum()
            ) * 100

            # Merge the unweighted and weighted data for a complete summary
            summary_df = pd.merge(unweighted_df, weighted_stats, on=var, how="outer").fillna(0)

            # Print the summary table using tabulate
            print(tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f"))
            print("\n")  # Extra line for clarity
