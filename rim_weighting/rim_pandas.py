from typing import Dict
import pandas as pd
import numpy as np
from tabulate import tabulate

class RIMWeightingPandas:

    def __init__(self, data: pd.DataFrame, spec: Dict, pre_weight: str = None, tolerance: float = 0.001, weight_col_name: str = 'rim_weight'):
        """
        Initialize the RIM weighting class.

        Parameters:
        - data: Pandas DataFrame with survey data.
        - spec: Dictionary with variable names as keys and target distributions as values.
        - pre_weight: Column name containing existing weights (if None, defaults to 1.0 for all).
        - tolerance: Convergence threshold.
        - weight_col_name: Name of the weight column in the DataFrame.
        """
        self.data = data.copy(deep=True)
        self.spec = spec
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = len(self.data)  # Fixed reference for proportion scaling

        # Validate specification
        self.validate_spec()

        # Set pre-weight column
        if pre_weight is None or pre_weight not in self.data.columns:
            self.data["pre_weight"] = 1.0  # Default to 1.0 if no pre_weight column provided
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight
            # Use the provided pre-weights as the initial weights, not just 1.0
            self.data[self.weight_col_name] = self.data[self.pre_weight_col_name]

        # Initialize rim weight column based on pre-weight
        if pre_weight:
            self.data[self.weight_col_name] = self.data[self.pre_weight_col_name]


    def validate_spec(self):
        """
        Validates the specification dictionary (spec) against the dataset.

        Raises:
        - ValueError if variables in spec do not exist in data.
        - ValueError if any categories in spec do not exist in the corresponding variable in data.
        - ValueError if the sum of all target proportions for any variable does not equal 1.0.
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
                raise ValueError(f"❌ Categories {missing_categories} in spec for variable '{var}' do not exist in the data.")

            # 4. Check if target proportions sum up to 1.0
            total_target = sum(targets.values())
            if not np.isclose(total_target, 1.0, atol=1e-6):
                raise ValueError(f"❌ Target proportions for '{var}' sum to {total_target:.6f}, but must sum to exactly 1.0.")


    def apply_weights(self, max_iterations=30, min_weight=0.5, max_weight=1.5) -> pd.DataFrame:
        """
        Applies RIM (Random Iterative Method) weighting using an iterative approach with convergence check.

        The method adjusts the weights of survey data iteratively to align with known marginal totals (specified in `spec`).
        This method follows the principles outlined in the paper:
        - Deming, W. E., & Stephan, F. F. (1940). *On a least squares adjustment of a sampled frequency table when the expected marginal totals are known*. 
        *Annals of Mathematical Statistics, 11*, 427-444.

        The steps in the method are as follows:
        1. **Initial Weights**: The `rim_weight` column is initialized from the `pre_weight` column (or defaults to 1.0 if no pre-weight 
            is provided).
        2. **Adjustment**: For each variable in the specification, the method computes the adjustment factor for each category, based on 
            the ratio of target counts to observed counts.
        3. **Normalization**: After adjustments, the total sum of weights is scaled to match the original sample size.
        4. **Weight Capping**: After normalization, all weights are clipped to lie within the specified `min_weight` and `max_weight` 
            bounds.
        5. **Convergence Check**: The iteration continues until the weights converge within specified bounds or until the maximum 
            number of iterations is reached.

        Parameters:
        - max_iterations (int): Maximum number of iterations allowed for weight adjustments.
        - min_weight (float): Minimum allowable weight for each observation. Adjusted weights that fall below this value are clipped.
        - max_weight (float): Maximum allowable weight for each observation. Adjusted weights that exceed this value are clipped.

        Returns:
        - pd.DataFrame: The original DataFrame with the updated `rim_weight` column containing the final adjusted weights.

        Method Details:
        - **Adjustments**: The adjustment is computed by comparing the current weighted totals (grouped by categories) to the target 
            values in the `spec` dictionary. For each category, the adjustment factor is calculated as the ratio of target count to 
            observed count.
        - **Normalization**: After each round of adjustments, the weights are normalized by multiplying by a scale factor, ensuring 
            the sum of the weights matches the total sample size.
        - **Convergence Criteria**: The iteration stops when all weights are within the allowed range (`min_weight`, `max_weight`). 
            The method uses the RMS error (Root Mean Squared Error) across the target variables as an indicator of how close the current weights are to the target proportions. The iteration may also stop early if the weights are already within the bounds, indicating convergence.
        """

        for iteration in range(max_iterations):
            rms_error = 0  

            for var, targets in self.spec.items():
                current_totals = self.data.groupby(var)[self.weight_col_name].sum()

                # Convert proportions to actual target counts
                if isinstance(next(iter(targets.values())), float):
                    targets = {k: v * self.total_sample for k, v in targets.items()}

                # Compute adjustment factor
                adjustment_factors = {
                    k: targets[k] / current_totals[k] if k in current_totals and current_totals[k] > 0 else 1
                    for k in targets.keys()
                }

                # Apply adjustments
                self.data[self.weight_col_name] *= self.data[var].map(adjustment_factors).fillna(1)

            # Normalize weights to maintain total sum
            scale_factor = self.total_sample / self.data[self.weight_col_name].sum()
            self.data[self.weight_col_name] *= scale_factor

            # Apply weight capping
            self.data[self.weight_col_name] = np.clip(self.data[self.weight_col_name], min_weight, max_weight)

            # Compute RMS Error
            for var, targets in self.spec.items():
                weighted_totals = self.data.groupby(var)[self.weight_col_name].sum()

                for category, target in targets.items():
                    observed = weighted_totals.get(category, 0)
                    rms_error += (target - observed) ** 2

            rms_error = np.sqrt(rms_error / len(self.spec))  

            # Compute iteration stats
            max_weight_value = self.data[self.weight_col_name].max()
            min_weight_value = self.data[self.weight_col_name].min()
            efficiency = self.weighting_efficiency()

            print(f"Iteration {iteration + 1}: RMS Error = {rms_error:.6f}, Efficiency = {efficiency:.2f}%, "
                f"Max Weight = {max_weight_value:.4f}, Min Weight = {min_weight_value:.4f}")

            # **Check if all weights are within the min/max limits**
            if min_weight_value >= min_weight and max_weight_value <= max_weight:
                print(f"✅ Converged in {iteration + 1} iterations: All weights within limits.")
                break

        return self.data


    def weighting_efficiency(self):
        """
        Computes the rim weighting efficiency as per the given formula:

            Efficiency (%) = 100 * ( Σ(Pj * Rj) )^2  /  ( Σ(Pj) * Σ(Pj * Rj^2) )

        where:
        - Pj is the pre-weight for case j (before rim weighting).
        - Rj is the rim weight for case j (after rim weighting).
        
        If pre_weight column does not exist, it initializes all values to 1.
        
        Returns:
            float: Rim weighting efficiency percentage.
        """
        Pj = self.data[self.pre_weight_col_name]  # Pre-weight column
        Rj = self.data[self.weight_col_name]  # Rim weight column

        numerator = (np.sum(Pj * Rj)) ** 2
        denominator = np.sum(Pj) * np.sum(Pj * (Rj ** 2))

        if denominator == 0:
            return 0.0  # Avoid division by zero

        efficiency = 100.0 * (numerator / denominator)
        return efficiency

    def generate_summary(self):
        """
        Generates and prints a formatted summary of unweighted and weighted counts per variable.

        Includes:
        - Unweighted counts & percentages
        - Weighted counts & percentages
        - Min/Max weights per category
        """
        for var in self.spec.keys():
            # Unweighted counts
            unweighted_df = self.data[var].value_counts(dropna=False).reset_index()
            unweighted_df.columns = [var, "Unweighted Count"]
            unweighted_df["Unweighted %"] = (unweighted_df["Unweighted Count"] / unweighted_df["Unweighted Count"].sum()) * 100

            # Weighted counts & min/max weights
            weighted_stats = self.data.groupby(var).agg(
                Weighted_Count=(self.weight_col_name, "sum"),
                Min_Weight=(self.weight_col_name, "min"),
                Max_Weight=(self.weight_col_name, "max")
            ).reset_index()

            weighted_stats["Weighted %"] = (weighted_stats["Weighted_Count"] / weighted_stats["Weighted_Count"].sum()) * 100

            # Merge to align categories correctly
            summary_df = pd.merge(unweighted_df, weighted_stats, on=var, how="outer").fillna(0)

            # Print the summary using tabulate
            print(tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f"))
            print("\n")  # Extra line for clarity
