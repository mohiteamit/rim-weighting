import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from typing import Dict

class RIMWeightingPySpark:
    def __init__(
        self,
        df: DataFrame,
        spec: Dict,
        pre_weight: str = None,
        tolerance: float = 0.001,
        weight_col_name: str = 'rim_weight'
    ):
        """
        Initialize the RIM weighting class for Spark DataFrame.

        :param df: Spark DataFrame with survey data
        :param spec: Dict with {variable_name: {category: proportion_or_count}}
        :param pre_weight: Name of column containing existing weights (optional)
        :param tolerance: Convergence threshold for RMS error
        :param weight_col_name: Name of the final weight column
        """
        self.spark = df.sparkSession
        self.df = df
        self.spec = spec
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name

        # We'll treat "total_sample" as the total row count
        self.total_sample = self.df.count()

        # Validate specification
        self.validate_spec()

        # Convert any proportions in spec to absolute counts exactly once
        self.convert_targets_to_counts()

        # If no pre_weight or it's not in columns, create one = 1.0
        if pre_weight is None or pre_weight not in self.df.columns:
            self.pre_weight_col_name = "pre_weight"
            self.df = self.df.withColumn(self.pre_weight_col_name, F.lit(1.0))
        else:
            self.pre_weight_col_name = pre_weight

        # Initialize the rim_weight column
        if pre_weight and (pre_weight in self.df.columns):
            self.df = self.df.withColumn(self.weight_col_name, F.col(self.pre_weight_col_name))
        else:
            self.df = self.df.withColumn(self.weight_col_name, F.lit(1.0))

    def validate_spec(self):
        """
        Validate that each variable in spec exists and that categories exist in the data.
        Also check if proportions sum to 1.0 (if the first entry is between 0 and 1).
        """
        df_cols = self.df.columns

        for var, targets in self.spec.items():
            # 1. Check if var is in df
            if var not in df_cols:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the DataFrame.")

            # 2. Collect distinct categories from Spark
            unique_cats = [r[var] for r in self.df.select(var).distinct().collect() if r[var] is not None]

            # 3. Check missing categories
            missing_cats = [c for c in targets.keys() if c not in unique_cats]
            if missing_cats:
                raise ValueError(
                    f"❌ Categories {missing_cats} in spec for variable '{var}' do not exist in the data."
                )

            # 4. If first value is in [0,1], assume proportions => check sum ~ 1.0
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                total_target = sum(targets.values())
                if abs(total_target - 1.0) > 1e-6:
                    raise ValueError(
                        f"❌ Target proportions for '{var}' sum to {total_target:.6f}, must be ~1.0"
                    )

    def convert_targets_to_counts(self):
        """
        Convert proportion-based targets in self.spec to absolute counts once.
        """
        for var, targets in self.spec.items():
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                # Convert to absolute counts
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=30, min_weight=0.5, max_weight=1.5) -> DataFrame:
        """
        Iteratively apply RIM weighting, capping each iteration, stopping when RMS error < tolerance
        or max_iterations is reached.

        :param max_iterations: Maximum number of iterations
        :param min_weight: Minimum allowable weight
        :param max_weight: Maximum allowable weight
        :return: Spark DataFrame with updated rim_weight column
        """
        for iteration in range(max_iterations):
            # For each variable in the spec, compute adjustment factors and update weights
            for var, targets in self.spec.items():
                # 1. Get the sum of current weights by category
                sum_df = (
                    self.df
                    .groupBy(var)
                    .agg(F.sum(self.weight_col_name).alias("observed_sum"))
                )

                # 2. Build a small DF of target values for each category
                #    (category -> target_count)
                target_rows = [(cat, float(targets[cat])) for cat in targets]
                target_schema = f"{var} string, target_count double"
                targets_df = self.spark.createDataFrame(target_rows, target_schema)

                # 3. Join sums_df and targets_df to compute factor = target_count / observed_sum
                factor_df = (
                    sum_df.join(targets_df, on=var, how="left")
                    .withColumn(
                        "factor",
                        F.when(F.col("observed_sum") > 0,
                               F.col("target_count") / F.col("observed_sum"))
                         .otherwise(F.lit(1.0))
                    )
                    .select(var, "factor")
                )

                # 4. Join factor_df back to main df, multiply existing weight by factor
                #    We do a left join on var. If there's no match, factor defaults to 1.0
                self.df = (
                    self.df.join(factor_df, on=var, how="left")
                    .withColumn(
                        self.weight_col_name,
                        F.col(self.weight_col_name) * F.coalesce(F.col("factor"), F.lit(1.0))
                    )
                    .drop("factor")  # remove temp column
                )

            # 5. Normalize total weights to match self.total_sample
            total_weight_sum = self.df.agg(F.sum(self.weight_col_name)).collect()[0][0]
            if total_weight_sum and total_weight_sum != 0:
                scale_factor = self.total_sample / total_weight_sum
                self.df = self.df.withColumn(
                    self.weight_col_name,
                    F.col(self.weight_col_name) * F.lit(scale_factor)
                )

            # 6. Clip weights to [min_weight, max_weight]
            self.df = self.df.withColumn(
                self.weight_col_name,
                F.when(F.col(self.weight_col_name) < min_weight, min_weight)
                 .when(F.col(self.weight_col_name) > max_weight, max_weight)
                 .otherwise(F.col(self.weight_col_name))
            )

            # 7. Compute RMS error
            rms_error = self.compute_rms_error()

            # 8. Print iteration info
            max_w = self.df.agg(F.max(self.weight_col_name)).collect()[0][0]
            min_w = self.df.agg(F.min(self.weight_col_name)).collect()[0][0]
            efficiency = self.weighting_efficiency()

            print(
                f"Iteration {iteration + 1}: "
                f"RMS Error = {rms_error:.6f}, "
                f"Efficiency = {efficiency:.2f}%, "
                f"Max Weight = {max_w:.4f}, "
                f"Min Weight = {min_w:.4f}"
            )

            # 9. Check convergence by RMS error
            if rms_error < self.tolerance:
                print(f"✅ Converged by RMS error < {self.tolerance} in {iteration + 1} iterations.")
                break

        return self.df

    def compute_rms_error(self) -> float:
        """
        Compute RMS error across all variables:
            sqrt( (Sum over all var/cat of (target - observed)^2 ) / number_of_vars )
        """
        import math

        sum_sq = 0.0
        # We'll do one pass per variable in Python
        for var, targets in self.spec.items():
            # Weighted sum by category
            sums = (
                self.df.groupBy(var)
                .agg(F.sum(self.weight_col_name).alias("weighted_sum"))
                .collect()
            )
            cat_to_observed = {row[var]: row["weighted_sum"] for row in sums}

            # Accumulate squared differences
            for cat, target_value in targets.items():
                observed = cat_to_observed.get(cat, 0.0)
                diff = target_value - observed
                sum_sq += diff * diff

        # average across number of variables
        n_vars = len(self.spec)
        return math.sqrt(sum_sq / n_vars) if n_vars else 0.0

    def weighting_efficiency(self) -> float:
        """
        Efficiency (%) = 100 * ( Σ(Pj * Rj) )^2  /  ( Σ(Pj) * Σ(Pj * Rj^2) )
        where Pj is pre_weight, Rj is rim_weight.
        """
        import math

        # We'll do a single row aggregation:
        # sum_pre = Σ(Pj)
        # sum_pre_rim = Σ(Pj * Rj)
        # sum_pre_rim2 = Σ(Pj * Rj^2)
        agg_row = (
            self.df.agg(
                F.sum(F.col(self.pre_weight_col_name)).alias("sum_pre"),
                F.sum(F.col(self.pre_weight_col_name) * F.col(self.weight_col_name)).alias("sum_pre_rim"),
                F.sum(F.col(self.pre_weight_col_name) * (F.col(self.weight_col_name) ** 2)).alias("sum_pre_rim2"),
            )
            .collect()[0]
        )
        sum_pre = agg_row["sum_pre"]
        sum_pre_rim = agg_row["sum_pre_rim"]
        sum_pre_rim2 = agg_row["sum_pre_rim2"]

        if sum_pre is None or sum_pre == 0:
            return 0.0

        numerator = (sum_pre_rim ** 2)
        denominator = sum_pre * sum_pre_rim2
        if denominator == 0:
            return 0.0
        return 100.0 * (numerator / denominator)

    def generate_summary(self):
        """
        Print unweighted/weighted counts per variable, plus min/max weight by category.
        Note: We'll collect results for display in Python, so this is not super scalable.
        """
        for var in self.spec.keys():
            print(f"--- Summary for Variable: {var} ---")

            # 1. Unweighted counts
            # We'll assume "unweighted" means counting rows ignoring any pre_weight
            unweighted_counts = (
                self.df.groupBy(var)
                .count()
                .withColumnRenamed("count", "Unweighted_Count")
            )

            # 2. Weighted counts
            weighted_counts = (
                self.df.groupBy(var)
                .agg(
                    F.sum(self.weight_col_name).alias("Weighted_Count"),
                    F.min(self.weight_col_name).alias("Min_Weight"),
                    F.max(self.weight_col_name).alias("Max_Weight")
                )
            )

            # We'll collect the sums so we can compute Weighted %
            total_weighted = weighted_counts.agg(F.sum("Weighted_Count")).collect()[0][0]

            # 3. Combine unweighted + weighted
            summary_df = (
                unweighted_counts.join(weighted_counts, on=var, how="outer")
                .fillna(0, subset=["Unweighted_Count", "Weighted_Count", "Min_Weight", "Max_Weight"])
                .withColumn(
                    "Unweighted_Percent",
                    F.col("Unweighted_Count") * 100.0 / self.total_sample
                )
                .withColumn(
                    "Weighted_Percent",
                    F.when(F.lit(total_weighted) != 0,
                           F.col("Weighted_Count") * 100.0 / F.lit(total_weighted))
                     .otherwise(F.lit(0.0))
                )
            )

            # Collect to Python for printing
            rows = summary_df.collect()
            # Print nicely
            print(f"{var:20s} | Unweighted_Count | Unweighted_% | Weighted_Count | Weighted_% | Min_Weight | Max_Weight")
            for r in rows:
                print(
                    f"{str(r[var]):20s} | "
                    f"{r['Unweighted_Count']:15.0f} | "
                    f"{r['Unweighted_Percent']:11.2f} | "
                    f"{r['Weighted_Count']:13.2f} | "
                    f"{r['Weighted_Percent']:10.2f} | "
                    f"{r['Min_Weight']:10.4f} | "
                    f"{r['Max_Weight']:10.4f}"
                )
            print()  # blank line
