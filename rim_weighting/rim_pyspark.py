from typing import Dict
import math
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, when, sum as spark_sum, min as spark_min, max as spark_max
)

class RIMWeightingPySpark:
    def __init__(
        self,
        spark: SparkSession,
        data: DataFrame,
        spec: Dict,
        id: str,
        pre_weight: str = None,
        tolerance: float = 0.005,
        weight_col_name: str = 'rim_weight',
        target: float = None
    ):
        """
        Initialize the RIM weighting class for Spark DataFrames.

        Parameters:
        - spark: SparkSession.
        - data: Spark DataFrame with survey data.
        - spec: Dictionary with variable names as keys and target distributions as values.
          Each value is a dict { category_name: proportion or absolute_count }.
        - id: Column name to be used as a unique identifier.
        - pre_weight: Column name containing existing weights (if None, defaults to 1.0 for all).
        - tolerance: Convergence threshold for RMS error.
        - weight_col_name: Name of the weight column in the DataFrame.
        - target: Desired total weighted sum. If None, defaults to the number of records.
        """
        # Validate that the id column exists.
        if id not in data.columns:
            raise ValueError(f"❌ ID column '{id}' does not exist in the dataset.")
        # Check for null values in the id column.
        if data.filter(col(id).isNull()).count() > 0:
            raise ValueError(f"❌ ID column '{id}' contains null values.")
        # Check that the id column is unique.
        dup_count = data.groupBy(id).count().filter(col("count") > 1).count()
        if dup_count > 0:
            raise ValueError(f"❌ ID column '{id}' must be unique per row.")

        self.spark = spark
        self.data = data
        self.spec = spec
        self.id = id
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = data.count()
        self.target = target if target is not None else self.total_sample

        self.validate_spec()
        self.convert_targets_to_counts()

        # Set pre-weight column
        if pre_weight is None or pre_weight not in self.data.columns:
            self.data = self.data.withColumn("pre_weight", lit(1.0))
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight

        # Initialize the weight column based on pre_weight.
        self.data = self.data.withColumn(self.weight_col_name, col(self.pre_weight_col_name))

    def get_weighted_factors(self) -> DataFrame:
        """
        Returns a DataFrame with the ID column and the weight column.
        (Note: Spark DataFrames do not have an index, so both columns are returned.)
        """
        return self.data.select(self.id, self.weight_col_name)

    def validate_spec(self):
        """
        Validates the specification dictionary against the DataFrame.

        - Checks that each variable in spec exists in the DataFrame.
        - Verifies that all categories in the spec are present in the data.
        - If the spec uses proportions (values between 0 and 1), ensures they sum to 1.0.
        """
        for var, targets in self.spec.items():
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")

            # Get distinct non-null categories from the data.
            distinct_vals = [
                row[0] for row in self.data.select(var).distinct().collect() if row[0] is not None
            ]
            missing_categories = [cat for cat in targets.keys() if cat not in distinct_vals]
            if missing_categories:
                raise ValueError(
                    f"❌ Categories {missing_categories} in spec for variable '{var}' do not exist in the data."
                )

            # If the targets are proportions, check that they sum to 1.0.
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                total_target = sum(targets.values())
                if not math.isclose(total_target, 1.0, abs_tol=1e-6):
                    raise ValueError(
                        f"❌ Target proportions for '{var}' sum to {total_target:.6f}, but must sum to exactly 1.0."
                    )

    def convert_targets_to_counts(self):
        """
        Converts any proportion-based targets to absolute counts (using total_sample).
        This is done only once.
        """
        for var, targets in self.spec.items():
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=12, min_weight=0.6, max_weight=1.4) -> DataFrame:
        """
        Applies RIM weighting iteratively until convergence or max_iterations is reached.

        After each iteration, the following steps are performed:
          1. Adjust weights for each variable according to the spec.
          2. Normalize total weights to match the target.
          3. Clip weights to the range [min_weight, max_weight].
          4. Compute the RMS error across all variables.
          5. Print iteration diagnostics including efficiency and weight range.

        Returns:
            The Spark DataFrame with updated weight column.
        """
        for iteration in range(max_iterations):
            for var, targets in self.spec.items():
                # Compute current weighted totals by category.
                current_totals_df = self.data.groupBy(var).agg(
                    spark_sum(self.weight_col_name).alias("observed")
                )
                current_totals = {row[var]: row["observed"] for row in current_totals_df.collect()}

                # Compute adjustment factors: target / observed.
                adjustment_factors = {
                    cat: (targets[cat] / current_totals.get(cat, 0))
                    if current_totals.get(cat, 0) > 0 else 1.0
                    for cat in targets
                }

                # Build mapping expression using when conditions.
                mapping_expr = None
                for cat, factor in adjustment_factors.items():
                    if mapping_expr is None:
                        mapping_expr = when(col(var) == cat, lit(factor))
                    else:
                        mapping_expr = mapping_expr.when(col(var) == cat, lit(factor))
                mapping_expr = mapping_expr.otherwise(lit(1.0))

                # Update the weight column.
                self.data = self.data.withColumn(
                    self.weight_col_name,
                    col(self.weight_col_name) * mapping_expr
                )

            # Normalize weights so that their sum equals the target.
            total_weight_sum = self.data.agg(
                spark_sum(self.weight_col_name).alias("total")
            ).collect()[0]["total"]
            scale_factor = self.target / total_weight_sum if total_weight_sum > 0 else 1.0
            self.data = self.data.withColumn(
                self.weight_col_name,
                col(self.weight_col_name) * lit(scale_factor)
            )

            # Clip weights to be within [min_weight, max_weight].
            self.data = self.data.withColumn(
                self.weight_col_name,
                when(col(self.weight_col_name) < lit(min_weight), lit(min_weight))
                .when(col(self.weight_col_name) > lit(max_weight), lit(max_weight))
                .otherwise(col(self.weight_col_name))
            )

            # Compute RMS error across all variables.
            rms_error_sq = 0.0
            for var, targets in self.spec.items():
                weighted_totals_df = self.data.groupBy(var).agg(
                    spark_sum(self.weight_col_name).alias("weighted")
                )
                weighted_totals = {row[var]: row["weighted"] for row in weighted_totals_df.collect()}
                for cat, target_value in targets.items():
                    observed = weighted_totals.get(cat, 0)
                    rms_error_sq += (target_value - observed) ** 2
            rms_error = math.sqrt(rms_error_sq / len(self.spec))

            # Compute efficiency and min/max weight statistics.
            efficiency = self.weighting_efficiency()
            weight_stats = self.data.agg(
                spark_max(self.weight_col_name).alias("max_weight"),
                spark_min(self.weight_col_name).alias("min_weight")
            ).collect()[0]
            max_weight_value = weight_stats["max_weight"]
            min_weight_value = weight_stats["min_weight"]

            print(
                f"Iteration {iteration + 1}: RMS Error = {rms_error:.6f}, "
                f"Efficiency = {efficiency:.2f}%, "
                f"Max Weight = {max_weight_value:.4f}, "
                f"Min Weight = {min_weight_value:.4f}"
            )

            if rms_error < self.tolerance:
                print(f"✅ Converged by `RMS error < {self.tolerance}` in {iteration + 1} iterations.")
                break

        return self.data

    def weighting_efficiency(self):
        """
        Computes the RIM weighting efficiency as:

            Efficiency (%) = 100 * (Σ(Pj * Rj))^2 / (Σ(Pj) * Σ(Pj * Rj^2))

        where:
          - Pj is the pre-weight (before RIM weighting).
          - Rj is the RIM weight (after weighting).

        Returns:
            Efficiency percentage (float).
        """
        agg_row = self.data.agg(
            spark_sum(col(self.pre_weight_col_name)).alias("sum_Pj"),
            spark_sum(col(self.pre_weight_col_name) * col(self.weight_col_name)).alias("sum_PR"),
            spark_sum(col(self.pre_weight_col_name) * (col(self.weight_col_name) ** 2)).alias("sum_PR2")
        ).collect()[0]
        sum_Pj = agg_row["sum_Pj"]
        sum_PR = agg_row["sum_PR"]
        sum_PR2 = agg_row["sum_PR2"]

        denominator = sum_Pj * sum_PR2
        if denominator == 0:
            return 0.0
        efficiency = 100.0 * ((sum_PR ** 2) / denominator)
        return efficiency

    def generate_summary(self):
        """
        Generates and prints a formatted summary for each variable in the spec.

        The summary includes:
          - Unweighted counts and percentages.
          - Weighted counts and percentages.
          - Minimum and maximum weights per category.

        For convenience, the summaries are collected to a Pandas DataFrame and printed
        using the tabulate package.
        """
        import pandas as pd
        from tabulate import tabulate

        for var in self.spec.keys():
            # Unweighted counts.
            unweighted_df = self.data.groupBy(var).count().withColumnRenamed("count", "Unweighted Count")
            total_unweighted = self.data.count()
            unweighted_df = unweighted_df.withColumn(
                "Unweighted %",
                (col("Unweighted Count") / total_unweighted) * 100
            )
            unweighted_pd = unweighted_df.toPandas()

            # Weighted counts and weight range.
            weighted_stats_df = self.data.groupBy(var).agg(
                spark_sum(self.weight_col_name).alias("Weighted_Count"),
                spark_min(self.weight_col_name).alias("Min_Weight"),
                spark_max(self.weight_col_name).alias("Max_Weight")
            )
            weighted_stats_df = weighted_stats_df.withColumn(
                "Weighted %",
                (col("Weighted_Count") / self.target) * 100
            )
            weighted_pd = weighted_stats_df.toPandas()

            # Merge the unweighted and weighted summaries.
            summary_pd = pd.merge(unweighted_pd, weighted_pd, on=var, how="outer").fillna(0)
            print(tabulate(summary_pd, headers="keys", tablefmt="github", floatfmt=".4f"))
            print("\n")
