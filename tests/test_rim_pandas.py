import pytest
import pandas as pd
import numpy as np
from rim_weighting.rim_pandas import RIMWeightingPandas

# Mock data for testing
def mock_data():
    data = pd.DataFrame({
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'age': ['18-24', '18-24', '25-34', '25-34', '35-44', '35-44', '45+', '45+', '18-24', '25-34'],
        'pre_weight': [1.1, 0.9, 1.2, 1.0, 1.1, 0.95, 1.0, 1.05, 1.0, 1.2]
    })
    return data

# Valid specification
def valid_spec():
    return {
        'gender': {'M': 0.5, 'F': 0.5},
        'age': {'18-24': 0.3, '25-34': 0.4, '35-44': 0.2, '45+': 0.1}
    }

# Invalid specification (wrong sum)
def invalid_spec():
    return {
        'gender': {'M': 0.6, 'F': 0.3},  # Sum is not 1.0
    }

# Test initialization
def test_initialization():
    data = mock_data()
    spec = valid_spec()
    rim = RIMWeightingPandas(data, spec, pre_weight='pre_weight')
    assert rim.weight_col_name in rim.data.columns
    assert rim.pre_weight_col_name == 'pre_weight'

# Test validation failure due to incorrect spec
def test_invalid_spec():
    data = mock_data()
    with pytest.raises(ValueError, match="Target proportions for 'gender' sum to 0.900000, but must sum to exactly 1.0"):
        RIMWeightingPandas(data, invalid_spec(), pre_weight='pre_weight')

# Test apply_weights function
def test_apply_weights():
    data = mock_data()
    spec = valid_spec()
    rim = RIMWeightingPandas(data, spec, pre_weight='pre_weight')
    weighted_data = rim.apply_weights()
    assert rim.weight_col_name in weighted_data.columns
    assert weighted_data[rim.weight_col_name].between(0.5, 1.5).all()  # Check weight bounds

# Test weighting efficiency
def test_weighting_efficiency():
    data = mock_data()
    spec = valid_spec()
    rim = RIMWeightingPandas(data, spec, pre_weight='pre_weight')
    rim.apply_weights()
    efficiency = rim.weighting_efficiency()
    assert efficiency > 0 and efficiency <= 100  # Efficiency should be within 0-100%

# Test summary generation (ensures no error)
def test_generate_summary():
    data = mock_data()
    spec = valid_spec()
    rim = RIMWeightingPandas(data, spec, pre_weight='pre_weight')
    rim.apply_weights()
    rim.generate_summary()  # No assertion, just ensuring it runs without error

if __name__ == "__main__":
    pytest.main()
