from typing import Dict
import numpy as np
import pandas as pd
from tabulate import tabulate

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, when
from pyspark.sql.types import DoubleType

class RIMWeightingPySpark:
    def __init__(
        self,
        data: DataFrame,
        spec: Dict,
        pre_weight: str = None,
        tolerance: float = 0.005,
        weight_col_name: str = 'rim_weight',
        target: float = None
    ):
        """
        Initialize the RIM weighting class using PySpark DataFrame.

        Parameters:
        - data: PySpark DataFrame with survey data.
        - spec: Dictionary with variable names as keys and target distributions as values.
          Each value is itself a dict { category_name: proportion or absolute_count }.
        - pre_weight: Column name containing existing weights (if None, defaults to 1.0 for all).
        - tolerance: Convergence threshold for RMS error.
        - weight_col_name: Name of the weight column in the DataFrame.
        - target: Desired total weighted sum. If set to None, defaults to the actual number of records.
        """
        # Save a reference to the Spark DataFrame.
        self.data = data
        self.spec = spec
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        # Total sample size is fixed (using count)
        self.total_sample = self.data.count()
        self.target = target if target is not None else self.total_sample

        # Validate the specification against the dataset.
        self.validate_spec()

        # Convert any proportions in spec to absolute counts (once)
        self.convert_targets_to_counts()

        # Set pre-weight column: if not provided (or not found), add a column with constant 1.0
        if pre_weight is None or pre_weight not in self.data.columns:
            self.data = self.data.withColumn("pre_weight", lit(1.0))
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight

        # Initialize the rim weight column: use pre_weight values if available, else default to 1.0.
        if self.pre_weight_col_name in self.data.columns:
            self.data = self.data.withColumn(self.weight_col_name, F.col(self.pre_weight_col_name))
        else:
            self.data = self.data.withColumn(self.weight_col_name, lit(1.0))

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
            # 1. Check if variable exists in data.
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")
            
            # 2. Get unique categories from the Spark DataFrame.
            unique_categories = [row[var] for row in self.data.select(var).distinct().collect() if row[var] is not None]
            
            # 3. Check if all specified categories exist in the data.
            missing_categories = [cat for cat in targets.keys() if cat not in unique_categories]
            if missing_categories:
                raise ValueError(
                    f"❌ Categories {missing_categories} in spec for variable '{var}' do not exist in the data."
                )
            
            # 4. If targets appear to be proportions, they must sum to 1.0.
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
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=12, min_weight=0.6, max_weight=1.4) -> DataFrame:
        """
        Applies RIM weighting using an iterative approach with RMS-error-based convergence.

        Parameters:
        - max_iterations (int): Maximum number of iterations allowed for weight adjustments.
        - min_weight (float): Minimum allowable weight for each observation.
        - max_weight (float): Maximum allowable weight for each observation.

        Returns:
        - DataFrame: The original DataFrame with the updated `rim_weight` column.
        """
        for iteration in range(max_iterations):
            # Iteratively adjust weights for each variable in the spec.
            for var, targets in self.spec.items():
                # Compute current weighted totals by category.
                totals = self.data.groupBy(var).agg(F.sum(self.weight_col_name).alias("total")).collect()
                totals_dict = {row[var]: row["total"] for row in totals}

                # Compute adjustment factors = (target_count / observed_count)
                adjustment_factors = {}
                for cat, target_value in targets.items():
                    observed = totals_dict.get(cat, 0)
                    if observed > 0:
                        adjustment_factors[cat] = target_value / observed
                    else:
                        adjustment_factors[cat] = 1.0

                # Define a UDF to map each row's category to its adjustment factor.
                def adjust(cat):
                    return float(adjustment_factors.get(cat, 1.0))
                adjust_udf = F.udf(adjust, DoubleType())

                # Multiply the current weight by the adjustment factor for the given variable.
                self.data = self.data.withColumn(
                    self.weight_col_name,
                    F.col(self.weight_col_name) * adjust_udf(F.col(var))
                )

            # Normalize total weights to match the target sum.
            total_weight_sum = self.data.agg(F.sum(self.weight_col_name).alias("sum")).collect()[0]["sum"]
            if total_weight_sum > 0:
                scale_factor = self.target / total_weight_sum
                self.data = self.data.withColumn(
                    self.weight_col_name,
                    F.col(self.weight_col_name) * lit(scale_factor)
                )

            # Clip weights to the range [min_weight, max_weight].
            self.data = self.data.withColumn(
                self.weight_col_name,
                when(F.col(self.weight_col_name) < lit(min_weight), lit(min_weight))
                .when(F.col(self.weight_col_name) > lit(max_weight), lit(max_weight))
                .otherwise(F.col(self.weight_col_name))
            )

            # Compute RMS error across all variables.
            rms_error_sum = 0.0
            for var, targets in self.spec.items():
                weighted_totals = self.data.groupBy(var).agg(F.sum(self.weight_col_name).alias("weighted_total")).collect()
                weighted_dict = {row[var]: row["weighted_total"] for row in weighted_totals}
                for cat, target_value in targets.items():
                    observed = weighted_dict.get(cat, 0)
                    rms_error_sum += (target_value - observed) ** 2
            rms_error = np.sqrt(rms_error_sum / len(self.spec))

            # Diagnostics: maximum and minimum weights.
            max_weight_value = self.data.agg(F.max(self.weight_col_name).alias("max")).collect()[0]["max"]
            min_weight_value = self.data.agg(F.min(self.weight_col_name).alias("min")).collect()[0]["min"]
            efficiency = self.weighting_efficiency()

            print(
                f"Iteration {iteration + 1}: "
                f"RMS Error = {rms_error:.6f}, "
                f"Efficiency = {efficiency:.2f}%, "
                f"Max Weight = {max_weight_value:.4f}, "
                f"Min Weight = {min_weight_value:.4f}"
            )

            # Check for convergence.
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
        # Compute sum(Pj * Rj)
        sum_PR = self.data.withColumn("prod", F.col(self.pre_weight_col_name) * F.col(self.weight_col_name)) \
                          .agg(F.sum("prod").alias("sum_PR")).collect()[0]["sum_PR"]
        # Compute sum(Pj)
        sum_P = self.data.agg(F.sum(self.pre_weight_col_name).alias("sum_P")).collect()[0]["sum_P"]
        # Compute sum(Pj * Rj^2)
        sum_PR2 = self.data.withColumn("prod2", F.col(self.pre_weight_col_name) * (F.col(self.weight_col_name) ** 2)) \
                           .agg(F.sum("prod2").alias("sum_PR2")).collect()[0]["sum_PR2"]

        numerator = sum_PR ** 2
        denominator = sum_P * sum_PR2

        if denominator == 0:
            return 0.0

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
            # Unweighted counts.
            unweighted_pdf = self.data.groupBy(var).count().toPandas()
            unweighted_pdf.rename(columns={"count": "Unweighted Count"}, inplace=True)
            total_unweighted = unweighted_pdf["Unweighted Count"].sum()
            unweighted_pdf["Unweighted %"] = (unweighted_pdf["Unweighted Count"] / total_unweighted) * 100

            # Weighted statistics.
            weighted_pdf = self.data.groupBy(var).agg(
                F.sum(self.weight_col_name).alias("Weighted_Count"),
                F.min(self.weight_col_name).alias("Min_Weight"),
                F.max(self.weight_col_name).alias("Max_Weight")
            ).toPandas()
            total_weighted = weighted_pdf["Weighted_Count"].sum()
            weighted_pdf["Weighted %"] = (weighted_pdf["Weighted_Count"] / total_weighted) * 100

            # Merge the unweighted and weighted summaries.
            summary_df = pd.merge(unweighted_pdf, weighted_pdf, on=var, how="outer").fillna(0)
            print(tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f"))
            print("\n")
