from typing import Dict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lit, sum as spark_sum, expr

class RIMWeightingPySpark:
    def __init__(
        self,
        data: DataFrame,
        spec: Dict,
        id: str,
        pre_weight: str = None,
        tolerance: float = 0.005,
        weight_col_name: str = "rim_weight",
        target: float = None
    ):
        """
        Initialize the RIM weighting class for PySpark.
        """
        self.spark = SparkSession.builder.getOrCreate()
        self.data = data.cache()
        self.spec = spec
        self.id = id
        self.tolerance = tolerance
        self.weight_col_name = weight_col_name
        self.total_sample = self.data.count()
        self.target = target if target is not None else self.total_sample
        
        if id not in data.columns:
            raise ValueError(f"❌ ID column '{id}' does not exist in the dataset.")
        
        # Validate the specification
        self.validate_spec()
        
        # Convert any proportions in spec to absolute counts
        self.convert_targets_to_counts()
        
        # Set pre-weight column
        if pre_weight is None or pre_weight not in self.data.columns:
            self.data = self.data.withColumn("pre_weight", lit(1.0))
            self.pre_weight_col_name = "pre_weight"
        else:
            self.pre_weight_col_name = pre_weight
        
        # Initialize rim weight column
        if pre_weight:
            self.data = self.data.withColumn(self.weight_col_name, col(self.pre_weight_col_name))
        else:
            self.data = self.data.withColumn(self.weight_col_name, lit(1.0))

    def validate_spec(self):
        """Validates the specification dictionary against the dataset."""
        for var, targets in self.spec.items():
            if var not in self.data.columns:
                raise ValueError(f"❌ Variable '{var}' in spec does not exist in the data.")

    def convert_targets_to_counts(self):
        """Convert proportions in spec to absolute counts."""
        for var, targets in self.spec.items():
            first_val = next(iter(targets.values()))
            if 0 <= first_val <= 1:
                self.spec[var] = {k: v * self.total_sample for k, v in targets.items()}

    def apply_weights(self, max_iterations=12, min_weight=0.6, max_weight=1.4) -> DataFrame:
        """Applies RIM weighting using an iterative approach."""
        for iteration in range(max_iterations):
            for var, targets in self.spec.items():
                weight_sums = self.data.groupBy(var).agg(spark_sum(self.weight_col_name).alias("total_weight"))
                adjustment_factors = {k: lit(v) / col("total_weight") for k, v in targets.items()}
                
                adjustment_expr = expr("CASE " + " ".join(
                    [f"WHEN {var} = '{k}' THEN {v}" for k, v in adjustment_factors.items()]
                ) + " ELSE 1.0 END")
                
                self.data = self.data.join(weight_sums, var, "left").withColumn(
                    self.weight_col_name, col(self.weight_col_name) * adjustment_expr
                ).drop("total_weight")
            
            # Normalize weights
            total_weight_sum = self.data.agg(spark_sum(self.weight_col_name)).collect()[0][0]
            scale_factor = self.target / total_weight_sum if total_weight_sum > 0 else 1.0
            self.data = self.data.withColumn(self.weight_col_name, col(self.weight_col_name) * lit(scale_factor))
            
            # Clip weights
            self.data = self.data.withColumn(
                self.weight_col_name, when(col(self.weight_col_name) < min_weight, min_weight)
                .when(col(self.weight_col_name) > max_weight, max_weight)
                .otherwise(col(self.weight_col_name))
            )
        
        return self.data

    def get_weighted_factors(self) -> DataFrame:
        """Returns a DataFrame with the ID column and the weight column."""
        return self.data.select(self.id, self.weight_col_name)
