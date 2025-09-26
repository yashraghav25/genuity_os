# CTGAN Documentation

## Overview

CTGAN (Conditional Tabular GAN) is a clean, optimized implementation for generating high-quality synthetic tabular data. This package provides a stable and efficient approach to synthetic data generation with improved accuracy and training stability.

## Features

- **Stable Training**: Enhanced with spectral normalization and gradient penalty
- **Improved Architecture**: Residual blocks with better gradient flow
- **Flexible Configuration**: Easy-to-use configuration system
- **High-Level API**: Simple interface for quick implementation
- **Model Persistence**: Save and load trained models

## Installation

### Requirements

```bash
torch>=1.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

### Installation

```bash
pip install -r ctgan/requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from ctgan import CTGANAPI

# Prepare your data
data = np.random.randn(1000, 10)  # Your tabular data
continuous_cols = [0, 1, 2, 3]    # Indices of continuous columns
categorical_cols = [4, 5, 6, 7, 8, 9]  # Indices of categorical columns

# Create and train the model
model = CTGANAPI()
losses = model.fit(data, continuous_cols, categorical_cols, epochs=1000)

# Generate synthetic data
synthetic_data = model.generate(500)

# Save the model
model.save("ctgan_model.pth")

# Load a saved model
model.load("ctgan_model.pth")
```

### Advanced Configuration

```python
from ctgan import CTGANConfig, CTGANFactory

# Custom configuration
config = CTGANConfig(
    latent_dim=256,
    hidden_dim=512,
    num_layers=4,
    learning_rate=2e-4,
    batch_size=256,
    use_spectral_norm=True,
    use_gradient_penalty=True,
    gradient_penalty_weight=10.0
)

# Create trainer with custom config
trainer = CTGANFactory.create_model(
    continuous_dims=[0, 1, 2],
    categorical_dims=[3, 4, 5],
    config=config
)

# Train the model
losses = trainer.fit(data, epochs=1000, verbose=True)

# Generate samples
synthetic_data = trainer.generate(1000)
```

## API Reference

### CTGANAPI

High-level API for easy model training and generation.

#### Methods

##### `fit(data, continuous_cols, categorical_cols, epochs=1000, **kwargs)`

Train the CTGAN model on provided data.

**Parameters:**
- `data` (np.ndarray): Training data
- `continuous_cols` (list): List of continuous column indices
- `categorical_cols` (list): List of categorical column indices
- `epochs` (int): Number of training epochs
- `**kwargs`: Additional configuration parameters

**Returns:**
- `dict`: Training losses

##### `generate(n_samples)`

Generate synthetic samples.

**Parameters:**
- `n_samples` (int): Number of samples to generate

**Returns:**
- `np.ndarray`: Generated synthetic data

##### `save(filepath)`

Save the trained model to disk.

**Parameters:**
- `filepath` (str): Path to save the model

##### `load(filepath)`

Load a previously saved model.

**Parameters:**
- `filepath` (str): Path to the saved model

### CTGANConfig

Configuration class for customizing model behavior.

#### Parameters

**Core Architecture:**
- `latent_dim` (int): Dimension of latent space (default: 128)
- `hidden_dim` (int): Hidden layer dimension (default: 256)
- `discriminator_hidden_dim` (int): Discriminator hidden dimension (default: 256)
- `num_layers` (int): Number of layers (default: 3)

**Training Settings:**
- `learning_rate` (float): Learning rate (default: 1e-4)
- `batch_size` (int): Batch size (default: 128)
- `discriminator_dropout` (float): Dropout rate (default: 0.3)

**Stability Improvements:**
- `use_spectral_norm` (bool): Enable spectral normalization (default: True)
- `use_gradient_penalty` (bool): Enable gradient penalty (default: True)
- `gradient_penalty_weight` (float): Gradient penalty weight (default: 10.0)

**Loss Balancing:**
- `generator_loss_weight` (float): Generator loss weight (default: 1.0)
- `discriminator_loss_weight` (float): Discriminator loss weight (default: 1.0)
- `diversity_loss_weight` (float): Diversity loss weight (default: 0.01)

### CTGANFactory

Factory class for creating CTGAN models with different configurations.

#### Methods

##### `create_model(continuous_dims, categorical_dims, **kwargs)`

Create a CTGAN model instance.

**Parameters:**
- `continuous_dims` (list): List of continuous dimension indices
- `categorical_dims` (list): List of categorical dimension indices
- `**kwargs`: Additional configuration parameters

**Returns:**
- `CTGANTrainer`: Configured trainer instance

## Architecture Details

### Generator

The generator uses a residual architecture with the following improvements:

- **Residual Blocks**: Improved gradient flow with skip connections
- **Spectral Normalization**: Stabilizes training by constraining weight norms
- **Layer Normalization**: Better training stability
- **Conditional Generation**: Supports both continuous and categorical features

### Discriminator

The discriminator features:

- **Spectral Normalization**: Prevents mode collapse
- **Gradient Penalty**: Enforces Lipschitz constraint
- **Dropout Regularization**: Prevents overfitting
- **Balanced Architecture**: Matches generator capacity

### Training Stability

The implementation includes several stability improvements:

1. **Spectral Normalization**: Applied to all linear layers
2. **Gradient Penalty**: WGAN-GP loss for stable training
3. **Learning Rate Scheduling**: Adaptive learning rate adjustment
4. **Loss Balancing**: Careful weighting of different loss components

## Best Practices

### Data Preparation

1. **Normalize Continuous Features**: Use StandardScaler or MinMaxScaler
2. **Encode Categorical Features**: Use LabelEncoder or OneHotEncoder
3. **Handle Missing Values**: Impute or remove missing data before training
4. **Balance Classes**: Ensure balanced representation of categorical values

### Training Tips

1. **Start with Default Settings**: The default configuration works well for most datasets
2. **Monitor Losses**: Watch for convergence and stability
3. **Adjust Batch Size**: Larger batches often improve stability
4. **Use Early Stopping**: Stop training when losses plateau

### Model Selection

1. **Cross-Validation**: Use k-fold validation to assess model quality
2. **Statistical Tests**: Compare distributions using KS tests or other metrics
3. **Downstream Tasks**: Evaluate synthetic data on actual use cases

## Troubleshooting

### Common Issues

**Training Instability:**
- Reduce learning rate
- Increase gradient penalty weight
- Use smaller batch sizes

**Mode Collapse:**
- Increase diversity loss weight
- Check data preprocessing
- Verify categorical encoding

**Poor Quality Samples:**
- Increase training epochs
- Adjust architecture (more layers/hidden units)
- Improve data preprocessing

### Performance Optimization

1. **GPU Acceleration**: Use CUDA-compatible PyTorch
2. **Batch Size**: Optimize for your GPU memory
3. **Data Loading**: Use DataLoader for large datasets
4. **Mixed Precision**: Consider using automatic mixed precision

## Examples

### Financial Data Generation

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ctgan import CTGANAPI

# Load financial data
df = pd.read_csv("financial_data.csv")

# Prepare data
continuous_features = ['income', 'age', 'credit_score']
categorical_features = ['education', 'marital_status', 'employment']

# Encode categorical features
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Normalize continuous features
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Get column indices
continuous_cols = [df.columns.get_loc(col) for col in continuous_features]
categorical_cols = [df.columns.get_loc(col) for col in categorical_features]

# Train model
model = CTGANAPI()
model.fit(df.values, continuous_cols, categorical_cols, epochs=2000)

# Generate synthetic data
synthetic_df = pd.DataFrame(
    model.generate(1000),
    columns=df.columns
)

# Inverse transform
synthetic_df[continuous_features] = scaler.inverse_transform(
    synthetic_df[continuous_features]
)
for i, col in enumerate(categorical_features):
    synthetic_df[col] = le.inverse_transform(
        synthetic_df[col].astype(int)
    )
```

### Healthcare Data Generation

```python
# Similar approach for healthcare data
# Focus on privacy-preserving synthetic data generation
# Ensure HIPAA compliance considerations
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Documentation updates
- Issue reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use CTGAN in your research, please cite:

```bibtex
@software{ctgan2024,
  title={CTGAN: Conditional Tabular GAN for Synthetic Data Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ctgan}
}
```

## Support

For questions, issues, or contributions:

- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]
- Email: [Contact email]
