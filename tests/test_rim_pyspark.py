import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from rim_weighting.rim_pyspark import RIMWeightingPySpark  # Assuming your implementation is in this module

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local").appName("pytest-spark").getOrCreate()

@pytest.fixture
def sample_data(spark):
    data = [
        ("A", "X", 1.0),
        ("A", "Y", 1.0),
        ("B", "X", 1.0),
        ("B", "Y", 1.0),
        ("C", "X", 1.0),
        ("C", "Y", 1.0),
    ]
    columns = ["group", "subgroup", "pre_weight"]
    return spark.createDataFrame(data, columns)

@pytest.fixture
def spec():
    return {
        "group": {"A": 0.3, "B": 0.5, "C": 0.2},
        "subgroup": {"X": 0.6, "Y": 0.4}
    }

@pytest.fixture
def rim_weighting(sample_data, spec):
    return RIMWeightingPySpark(sample_data, spec, pre_weight="pre_weight")


def test_initialization(rim_weighting):
    assert rim_weighting.data is not None
    assert rim_weighting.spec is not None
    assert rim_weighting.total_sample > 0
    assert rim_weighting.weight_col_name == "rim_weight"


def test_validate_spec(rim_weighting):
    try:
        rim_weighting.validate_spec()
    except Exception as e:
        pytest.fail(f"validate_spec raised an exception: {e}")


def test_apply_weights(rim_weighting):
    weighted_data = rim_weighting.apply_weights(max_iterations=5)
    assert weighted_data is not None
    assert rim_weighting.data.select(col("rim_weight")).count() == rim_weighting.total_sample
    

def test_generate_summary(rim_weighting):
    try:
        rim_weighting.generate_summary()
    except Exception as e:
        pytest.fail(f"generate_summary raised an exception: {e}")