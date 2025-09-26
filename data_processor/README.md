# Data Processor Module

## Overview

The `data_processor` module provides robust preprocessing and postprocessing utilities for tabular data, including handling of continuous, categorical, and binary columns, as well as outlier detection, imputation, scaling, encoding, and PCA.

## Features
- **Automatic column classification**: Detects continuous, categorical, binary, and long text columns.
- **Imputation**: Handles missing values with various strategies.
- **Encoding**: Supports multiple encoding strategies for categorical columns (binary columns are NOT encoded):
  - `label` (integer)
  - `onehot`
  - `binary`
  - `embedding` (use label encoding, model should use embeddings)
  - `target` (requires target variable)
  - `frequency`
  - `hash`
- **Scaling**: Multiple scaling strategies for continuous columns.
- **PCA**: Optional dimensionality reduction.
- **Outlier detection**: Flags outliers in continuous columns.
- **Seamless postprocessing**: Inverse transforms and reconstructs original data (where possible).

## Supported Encoding Strategies

| Encoding Type      | Suitable for GANs? | Pros                        | Cons                        |
|--------------------|--------------------|-----------------------------|-----------------------------|
| Label/Integer      | Yes                | Simple, compact             | Implies order               |
| One-Hot            | Yes (with care)    | No order, easy to interpret | High dimensionality         |
| Binary             | Yes                | Lower dimension             | Harder to interpret         |
| Embedding          | Yes (if supported) | Efficient, powerful         | Needs model support         |
| Target/Mean        | Rarely             | Can boost performance       | Risk of leakage, supervised |
| Frequency          | Rarely             | Simple                      | May lose info               |
| Hash               | Yes                | Handles many categories     | Collisions, less clear      |

### When to Use Each Encoding
- **Label/Integer**: Most recommended for custom GANs, especially if you use embedding layers.
- **One-hot**: If your model is designed to handle one-hot vectors and you ensure data integrity.
- **Binary**: When you have high-cardinality categorical variables.
- **Embedding**: If your CTGAN supports embeddings (common in deep learning).
- **Target/Mean**: Only for supervised tasks, not typical for GANs.
- **Frequency**: Rare for GANs, but possible.
- **Hash**: When you have many unique categories.

## Binary Columns Handling
- **Binary columns** (numeric columns with only 2 unique values) are detected automatically.
- **No encoding is applied to binary columns**. They are left as-is during preprocessing and postprocessing.
- Only categorical columns are encoded.

## Usage

### Preprocessing
```python
from data_preprocess import TabularPreprocessor
import pandas as pd

df = pd.DataFrame({
    'A': [0, 1, 0, 1],  # binary
    'B': ['x', 'y', 'x', 'z'],  # categorical
    'C': [1.2, 3.4, 2.2, 4.5],  # continuous
})

# Example: label encoding
pre = TabularPreprocessor(encoding_strategy='label')
result = pre.fit_transform(df)
preprocessed = result['preprocessed']

# Example: one-hot encoding
pre = TabularPreprocessor(encoding_strategy='onehot')
result = pre.fit_transform(df)

# Example: binary encoding
pre = TabularPreprocessor(encoding_strategy='binary')
result = pre.fit_transform(df)

# Example: hash encoding
pre = TabularPreprocessor(encoding_strategy='hash')
result = pre.fit_transform(df)

# Example: frequency encoding
pre = TabularPreprocessor(encoding_strategy='frequency')
result = pre.fit_transform(df)

# Example: target encoding (requires target)
pre = TabularPreprocessor(encoding_strategy='target')
pre.target = [0, 1, 0, 1]  # set your target variable here
result = pre.fit_transform(df)
```

### Postprocessing
```python
from data_postprocess import TabularPostprocessor

post = TabularPostprocessor(preprocessor_object=pre)
reconstructed = post.inverse_transform_modified_data(preprocessed)
```

## API
- `TabularPreprocessor`: Main class for preprocessing.
  - `fit_transform(df)`: Fits and transforms the DataFrame.
  - `save_preprocessor(filepath)`: Saves the preprocessor state.
  - `load_preprocessor(filepath)`: Loads a preprocessor state.
- `TabularPostprocessor`: Main class for postprocessing.
  - `inverse_transform_modified_data(df)`: Reconstructs original data from processed data (where possible).
  - `transform_new_data(df)`: Applies preprocessing to new data.

## Notes
- Binary columns are not encoded.
- Some encodings (binary, hash, frequency, target) are not invertible; postprocessing will copy encoded columns as-is and warn you.
- For more details, see the demo notebook.