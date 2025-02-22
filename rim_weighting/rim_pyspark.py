from typing import Dict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, sum as spark_sum, count, sqrt

class RIMWeightingPySpark:
    def __init__(
        self,
        spark: SparkSession,
        data: DataFrame,
        spec: Dict,
        id_col: str,
        pre_weight: str = None,
        tolerance: float = 0.005,
        weight_col_name: str = 'rim_weight',
        target: float = None
    ):
        self.spark = spark
        self.data = data
        self.spec = spec
        self.id_col = id_col
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = data.count()
        self.target = target if target is not None else self.total_sample
        
        if id_col not in data.columns:
            raise ValueError(f"❌ ID column '{id_col}' does not exist in the dataset.")
        
        self.validate_spec()
        self.convert_targets_to_counts()
        
        if pre_weight is None or pre_weight not in data.columns:
            self.pre_weight_col_name = "pre_weight"
            self.data = data.withColumn("pre_weight", lit(1.0))
        else:
            self.pre_weight_col_name = pre_weight

        self.data = self.data.withColumn(self.weight_col_name, col(self.pre_weight_col_name))

    def get_weighted_factors(self) -> DataFrame:
        """
        Returns a DataFrame with the ID column as the index and the weight column.
        """
        return self.data.select(self.id_col, self.weight_col_name)

    def validate_spec(self):
        for var, targets in self.spec.items():
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")

    def convert_targets_to_counts(self):
        for var, targets in self.spec.items():
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=12, min_weight=0.6, max_weight=1.4) -> DataFrame:
        for iteration in range(max_iterations):
            for var, targets in self.spec.items():
                current_totals = (
                    self.data.groupBy(var)
                    .agg(spark_sum(self.weight_col_name).alias("observed"))
                ).collect()
                
                adjustment_factors = {row[var]: targets[row[var]] / row["observed"] if row["observed"] > 0 else 1.0 
                                      for row in current_totals}
                
                mapping_expr = when(col(var).isNull(), 1.0)
                for cat, factor in adjustment_factors.items():
                    mapping_expr = mapping_expr.when(col(var) == cat, factor)
                
                self.data = self.data.withColumn(self.weight_col_name, col(self.weight_col_name) * mapping_expr)
            
            total_weight_sum = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            scale_factor = self.target / total_weight_sum if total_weight_sum > 0 else 1.0
            self.data = self.data.withColumn(self.weight_col_name, col(self.weight_col_name) * scale_factor)
            self.data = self.data.withColumn(
                self.weight_col_name,
                when(col(self.weight_col_name) < min_weight, min_weight)
                .when(col(self.weight_col_name) > max_weight, max_weight)
                .otherwise(col(self.weight_col_name))
            )
            
            rms_error = 0.0
            for var, targets in self.spec.items():
                weighted_totals = (
                    self.data.groupBy(var)
                    .agg(spark_sum(self.weight_col_name).alias("weighted"))
                ).collect()
                for row in weighted_totals:
                    cat, observed = row[var], row["weighted"]
                    target_value = targets.get(cat, 0)
                    rms_error += (target_value - observed) ** 2
            
            rms_error = sqrt(rms_error / len(self.spec))
            print(f"Iteration {iteration + 1}: RMS Error = {rms_error:.6f}")
            if rms_error < self.tolerance:
                print(f"✅ Converged in {iteration + 1} iterations.")
                break

        return self.data
