from typing import Dict
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, sum as spark_sum, when

class RIMWeightingPySpark:
    
    def __init__(self, data: DataFrame, spec: Dict, pre_weight: str = None, tolerance: float = 0.001, weight_col_name: str = 'rim_weight'):
        """
        Initialize the RIM weighting class for PySpark.

        Parameters:
        - data: PySpark DataFrame with survey data.
        - spec: Dictionary with variable names as keys and target distributions as values.
        - pre_weight: Column name containing existing weights (if None, defaults to 1.0 for all).
        - tolerance: Convergence threshold.
        - weight_col_name: Name of the weight column in the DataFrame.
        """
        self.spark = data.sql_ctx.sparkSession
        self.data = data
        self.spec = spec
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = data.count()
        
        # Validate specification
        self.validate_spec()
        
        # Set pre-weight column
        if pre_weight is None or pre_weight not in data.columns:
            self.data = self.data.withColumn("pre_weight", lit(1.0))
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight
        
        # Initialize rim weight column based on pre-weight
        self.data = self.data.withColumn(self.weight_col_name, col(self.pre_weight_col_name))
    
    def validate_spec(self):
        """
        Validates the specification dictionary (spec) against the dataset.
        """
        for var, targets in self.spec.items():
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")
            
            unique_categories = [row[var] for row in self.data.select(var).distinct().collect()]
            missing_categories = [cat for cat in targets.keys() if cat not in unique_categories]
            if missing_categories:
                raise ValueError(f"❌ Categories {missing_categories} in spec for variable '{var}' do not exist in the data.")
            
            total_target = sum(targets.values())
            if not abs(total_target - 1.0) < 1e-6:
                raise ValueError(f"❌ Target proportions for '{var}' sum to {total_target:.6f}, but must sum to exactly 1.0.")
    
    def apply_weights(self, max_iterations=30, min_weight=0.5, max_weight=1.5) -> DataFrame:
        """
        Applies RIM weighting using an iterative approach with convergence check.
        """
        for iteration in range(max_iterations):
            rms_error = 0  
            for var, targets in self.spec.items():
                # Compute current weighted totals
                current_totals = self.data.groupBy(var).agg(spark_sum(self.weight_col_name).alias("weighted_sum"))
                
                # Convert proportions to actual target counts
                targets_actual = {k: v * self.total_sample for k, v in targets.items()}
                
                # Compute adjustment factors
                adjustment_factors = {k: targets_actual[k] / v if v > 0 else 1 for k, v in current_totals.collect() if k in targets_actual}
                
                # Apply adjustments
                self.data = self.data.withColumn(self.weight_col_name, 
                    when(col(var).isin(adjustment_factors.keys()), col(self.weight_col_name) * adjustment_factors[col(var)])
                    .otherwise(col(self.weight_col_name))
                )
            
            # Normalize weights
            total_weight = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            scale_factor = self.total_sample / total_weight
            self.data = self.data.withColumn(self.weight_col_name, col(self.weight_col_name) * scale_factor)
            
            # Apply weight capping
            self.data = self.data.withColumn(self.weight_col_name, 
                when(col(self.weight_col_name) < min_weight, min_weight)
                .when(col(self.weight_col_name) > max_weight, max_weight)
                .otherwise(col(self.weight_col_name))
            )
            
            # Check for convergence
            max_weight_value = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            min_weight_value = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            if min_weight_value >= min_weight and max_weight_value <= max_weight:
                print(f"✅ Converged in {iteration + 1} iterations: All weights within limits.")
                break
        
        return self.data

    def generate_summary(self):
        """
        Generates a summary of unweighted and weighted counts per variable.
        """
        for var in self.spec.keys():
            unweighted_counts = self.data.groupBy(var).count()
            unweighted_counts = unweighted_counts.withColumn("Unweighted %", (col("count") / self.total_sample) * 100)
            
            weighted_counts = self.data.groupBy(var).agg(
                spark_sum(self.weight_col_name).alias("Weighted_Count"),
                spark_sum(self.weight_col_name).alias("Weighted %")
            )
            total_weight = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            weighted_counts = weighted_counts.withColumn("Weighted %", (col("Weighted_Count") / total_weight) * 100)
            
            summary_df = unweighted_counts.join(weighted_counts, var, "outer").fillna(0)
            summary_df.show()

