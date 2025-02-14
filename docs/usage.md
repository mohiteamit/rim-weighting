# 📖 Usage Guide

This guide explains how to use `RIMWeightingPandas` and `RIMWeightingPySpark` in your projects.

---

## 1️⃣ **Using RIMWeightingPandas**
### **📌 Example: Applying Weights**
```python
import pandas as pd
from rim_weighting.rim_pandas import RIMWeightingPandas

# Sample data
data = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'age': ['18-24', '25-34', '35-44', '45+', '18-24'],
    'pre_weight': [1.1, 0.9, 1.2, 1.0, 1.05]
})

# RIM specification
spec = {
    'gender': {'M': 0.5, 'F': 0.5},
    'age': {'18-24': 0.3, '25-34': 0.4, '35-44': 0.2, '45+': 0.1}
}

# Apply weighting
rim = RIMWeightingPandas(data, spec, pre_weight='pre_weight')
weighted_data = rim.apply_weights()
print(weighted_data)
```

---

## 2️⃣ **Using RIMWeightingPySpark**
If processing large datasets, use the PySpark implementation:
```python
from rim_weighting.rim_pyspark import RIMWeightingPySpark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RIMWeighting").getOrCreate()
data = spark.createDataFrame([
    ("M", "18-24", 1.1),
    ("F", "25-34", 0.9),
    ("M", "35-44", 1.2),
    ("F", "45+", 1.0)
], ["gender", "age", "pre_weight"])

spec = {
    "gender": {"M": 0.5, "F": 0.5},
    "age": {"18-24": 0.3, "25-34": 0.4, "35-44": 0.2, "45+": 0.1}
}

rim = RIMWeightingPySpark(data, spec, pre_weight="pre_weight")
weighted_data = rim.apply_weights()
weighted_data.show()
```

---

## 📊 **Generating a Summary Report**
After applying weights, you can generate a summary report:
```python
rim.generate_summary()
```
