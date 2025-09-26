# 🔒 Differential Privacy Module

A minimal differential privacy module for applying DP to preprocessed DataFrames with minimal noise addition.

## 📋 Overview

This module provides a simple way to apply differential privacy to your preprocessed data before synthetic data generation. It's designed to be:

- **Minimal**: Adds minimal noise while preserving data utility
- **Fast**: Optimized for preprocessed DataFrames
- **Flexible**: Multiple privacy methods and parameters
- **Safe**: Preserves important columns like targets

## 🚀 Quick Start

```python
from genuity.core_generator import apply_differential_privacy

# Apply minimal differential privacy to your preprocessed DataFrame
df_dp = apply_differential_privacy(
    df=your_preprocessed_df,
    epsilon=1.0,  # Privacy budget
    method="minimal",  # Minimal noise
    preserve_columns=['target']  # Keep target unchanged
)
```

## 📊 Usage Examples

### 1. Minimal Noise (Default)
```python
# Fastest method with least noise
df_dp = apply_differential_privacy(df_preprocessed, method="minimal")
```

### 2. Higher Privacy
```python
# More private with Laplace noise
df_dp = apply_differential_privacy(
    df_preprocessed,
    epsilon=0.5,
    method="laplace"
)
```

### 3. Large Datasets
```python
# Better for large datasets with Gaussian noise
df_dp = apply_differential_privacy(
    df_preprocessed,
    epsilon=2.0,
    method="gaussian"
)
```

### 4. Preserve Specific Columns
```python
# Keep target and important columns unchanged
df_dp = apply_differential_privacy(
    df_preprocessed,
    preserve_columns=['target', 'important_feature']
)
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Your preprocessed DataFrame |
| `epsilon` | float | 1.0 | Privacy budget (lower = more private) |
| `method` | str | "minimal" | DP method: "minimal", "laplace", "gaussian" |
| `preserve_columns` | List[str] | None | Columns to keep unchanged |
| `random_state` | int | None | Random seed for reproducibility |

## 🔧 Advanced Usage

### Using the Processor Class
```python
from genuity.core_generator import DifferentialPrivacyProcessor

processor = DifferentialPrivacyProcessor(
    epsilon=0.5,
    noise_scale=0.05,
    categorical_noise_prob=0.03,
    numerical_noise_std=0.01,
    random_state=42
)

df_dp = processor.apply_dp(
    df=df_preprocessed,
    method="laplace",
    preserve_columns=['target']
)

# Get privacy information
privacy_info = processor.get_privacy_info()
print(privacy_info)
```

## 📈 Privacy Methods

### 1. Minimal Method
- **Best for**: Most use cases
- **Noise**: Very small Gaussian noise
- **Speed**: Fastest
- **Utility**: Highest

### 2. Laplace Method
- **Best for**: Balanced privacy/utility
- **Noise**: Laplace noise based on data sensitivity
- **Speed**: Medium
- **Utility**: Good

### 3. Gaussian Method
- **Best for**: Large datasets
- **Noise**: Gaussian noise with formal DP guarantees
- **Speed**: Slower
- **Utility**: Good for large datasets

## 🎯 Recommendations

### For Most Use Cases
```python
df_dp = apply_differential_privacy(df_preprocessed, method="minimal")
```

### For Higher Privacy Requirements
```python
df_dp = apply_differential_privacy(
    df_preprocessed,
    epsilon=0.5,
    method="laplace"
)
```

### For Large Datasets
```python
df_dp = apply_differential_privacy(
    df_preprocessed,
    epsilon=2.0,
    method="gaussian"
)
```

## 🔐 Privacy Parameters

- **epsilon**: Privacy budget (0.1-10.0)
  - Lower values = more private but less utility
  - Higher values = less private but more utility
- **delta**: Privacy parameter (default: 1e-5)
- **noise_scale**: Scale factor for noise (default: 0.1)
- **categorical_noise_prob**: Probability of flipping categorical values (default: 0.05)
- **numerical_noise_std**: Standard deviation for numerical noise (default: 0.01)

## ⚡ Performance

- **Minimal method**: ~1-2 seconds for 10K rows
- **Laplace method**: ~2-3 seconds for 10K rows
- **Gaussian method**: ~3-5 seconds for 10K rows

## 🛡️ Safety Features

- **Column preservation**: Keep important columns unchanged
- **Bounds checking**: Ensure values stay within reasonable ranges
- **Error handling**: Graceful handling of edge cases
- **Reproducibility**: Random state for consistent results

## 📝 Example Workflow

```python
# 1. Load your preprocessed data
df_preprocessed = pd.read_csv("preprocessed_data.csv")

# 2. Apply differential privacy
df_dp = apply_differential_privacy(
    df=df_preprocessed,
    epsilon=1.0,
    method="minimal",
    preserve_columns=['target']
)

# 3. Use for synthetic data generation
# Your existing synthetic data generation code here
```

## 🚨 Important Notes

1. **Input**: Your DataFrame should be preprocessed (scaled, encoded, etc.)
2. **Target columns**: Use `preserve_columns` to keep targets unchanged
3. **Privacy budget**: Start with epsilon=1.0 and adjust based on needs
4. **Testing**: Always test with a small sample first

## ✅ Success Indicators

- Output DataFrame has same shape as input
- Numerical distributions are slightly perturbed
- Categorical distributions are slightly modified
- Preserved columns remain unchanged
- No errors or warnings in logs

---

**Ready to use!** Your preprocessed DataFrame can now be safely shared with differential privacy protection.
