# RIMWeightingPySpark
- This work is pending

# RIMWeightingPandas

**RIMWeightingPandas** is a Python implementation of the **Random Iterative Method (RIM)** weighting algorithm, as described in the paper:

- Deming, W. E., & Stephan, F. F. (1940). *On a least squares adjustment of a sampled frequency table when the expected marginal totals are known*. Annals of Mathematical Statistics, 11, 427-444.

This method is used to adjust the weights of survey data so that they align with known marginal totals, helping to improve the accuracy of weighted statistical analysis.

## Key Features:
- Adjusts survey data to match known target distributions (marginal totals).
- Implements the **Random Iterative Method (RIM)** as described by Deming and Stephan.
- Provides convergence checks to ensure that weights stabilize within specified bounds.
- Includes functionality to handle pre-existing weights and apply iterative adjustments.

## Installation

To install **RIMWeightingPandas**, use the following command:

```bash
pip install RIMWeightingPandas
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/mohiteamit/rim-weighting.git
cd rim-weighting
pip install .
```

## Usage

### Importing and Initializing the Class

First, import the class and prepare your dataset.

```python
import pandas as pd
from RIMWeightingPandas import RIMWeightingPandas

# Example DataFrame (survey data)
data = pd.DataFrame({
    'age_group': ['18-25', '26-35', '36-45', '46-60', '60+'],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'weight': [1, 1, 1, 1, 1]  # Pre-existing weights (optional)
})

# Define the specification (target proportions for each category)
spec = {
    'age_group': {'18-25': 0.2, '26-35': 0.2, '36-45': 0.2, '46-60': 0.2, '60+': 0.2},
    'gender': {'M': 0.5, 'F': 0.5}
}

# Initialize the RIMWeightingPandas object
rim_weighting = RIMWeightingPandas(data, spec, pre_weight='weight')

# Apply RIM weighting
weighted_data = rim_weighting.apply_weights()
```

### Detailed Explanation of Methods

- **`__init__(data, spec, pre_weight=None, tolerance=0.001, weight_col_name='rim_weight')`**: Initializes the RIMWeightingPandas object.
  - `data`: The survey data as a Pandas DataFrame.
  - `spec`: A dictionary of target proportions for each category (marginal totals).
  - `pre_weight`: Optional column name containing existing weights in the data. Defaults to `None`.
  - `tolerance`: Convergence threshold. Defaults to `0.001`.
  - `weight_col_name`: The name of the weight column in the DataFrame. Defaults to `'rim_weight'`.

- **`apply_weights(max_iterations=30, min_weight=0.5, max_weight=1.5)`**: Applies the RIM weighting algorithm.
  - `max_iterations`: Maximum number of iterations allowed.
  - `min_weight`: Minimum allowable weight for each observation. Adjusted weights that fall below this value are clipped.
  - `max_weight`: Maximum allowable weight for each observation. Adjusted weights that exceed this value are clipped.
  - **Returns**: The DataFrame with the adjusted `rim_weight` column.

- **`weighting_efficiency()`**: Computes the RIM weighting efficiency as a percentage, based on the formula:
  
  \[
  \text{Efficiency (\%)} = \frac{ \left( \sum (P_j \times R_j) \right)^2 }{ \sum P_j \times \sum P_j \times (R_j^2) }
  \]

  where \( P_j \) are the pre-weights and \( R_j \) are the adjusted rim weights.

- **`generate_summary()`**: Generates a summary of the unweighted and weighted counts per variable, including:
  - Unweighted counts & percentages.
  - Weighted counts & percentages.
  - Min/Max weights per category.

## Example Output

After applying the RIM weighting, you can generate a summary of your data:

```python
rim_weighting.generate_summary()
```

This will display a formatted summary showing the unweighted and weighted counts for each variable.

## Contributing

I welcome contributions! If you'd like to help improve this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the **MIT License**. You can freely use, modify, and distribute this code. However, **you must provide appropriate credit** by mentioning this repository in any usage of the code. For more details, see the `LICENSE` file.
