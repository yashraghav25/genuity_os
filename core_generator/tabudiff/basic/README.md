# TabuDiff Basic

**TabuDiff Basic** is a simplified implementation of diffusion models for tabular data generation. It provides a clean, easy-to-use interface for training diffusion models on structured data and generating synthetic samples that preserve the statistical properties of the original dataset.

## 🚀 What is TabuDiff?

TabuDiff (Tabular Diffusion) applies the powerful diffusion model framework to structured/tabular data. Unlike traditional generative models that struggle with mixed data types and complex relationships in tabular data, TabuDiff uses a denoising diffusion probabilistic model (DDPM) approach to learn the underlying data distribution and generate high-quality synthetic samples.

## ✨ Key Features

- **Simple API**: Train and generate synthetic data with just a few lines of code
- **Pandas Integration**: Seamlessly work with DataFrames and CSV files
- **Flexible Configuration**: Customizable model architecture and training parameters
- **Lightweight**: Minimal dependencies and fast training on CPU/GPU
- **Production Ready**: Built-in model saving/loading for deployment

## 📦 Installation

```bash
pip install torch pandas numpy
```

## 🎯 Quick Start

### Basic Usage

```python
import pandas as pd
from tabudiff.basic.utils import TabuDiffAPI

# Load your data
df = pd.read_csv("your_data.csv")

# Create and train the model
api = TabuDiffAPI()
api.fit_dataframe(df)

# Generate synthetic data
synthetic_df = api.generate_dataframe(num_samples=1000)
print(synthetic_df.head())
```

### One-Line CSV Processing

```python
from tabudiff.basic.utils import TabuDiffAPI

api = TabuDiffAPI()
# Train on input.csv and save synthetic data to output.csv
api.fit_csv_and_generate_csv("input.csv", "output.csv", num_samples=5000)
```

## 🏗️ Architecture

TabuDiff Basic consists of three core components:

### 1. Score Network (`ScoreNetwork`)
- **Purpose**: Predicts the noise added to data at each diffusion timestep
- **Architecture**: Multi-layer perceptron (MLP) with GELU activation
- **Input**: Noisy data + timestep embedding
- **Output**: Predicted noise

### 2. Variance Scheduler (`VarianceScheduler`)
- **Purpose**: Manages the noise schedule during diffusion process
- **Schedule**: Linear beta schedule from `beta_start` to `beta_end`
- **Timesteps**: Configurable number of diffusion steps (default: 200)

### 3. Diffusion Sampler (`DiffusionSampler`)
- **Purpose**: Generates new samples by reversing the diffusion process
- **Method**: DDPM sampling with denoising steps
- **Output**: Clean synthetic samples

## ⚙️ Configuration

Customize TabuDiff Basic using the `TabuDiffConfig` class:

```python
from tabudiff.basic.config import TabuDiffConfig

config = TabuDiffConfig(
    # Training parameters
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=5,

    # Diffusion parameters
    num_diffusion_steps=200,
    beta_start=1e-4,
    beta_end=0.02,

    # Model architecture
    hidden_dim=256,
    num_hidden_layers=3,
    dropout=0.0,

    # Generation
    num_samples=1000,
    device="cpu"  # or "cuda" for GPU
)

api = TabuDiffAPI(config)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | Learning rate for Adam optimizer |
| `batch_size` | 256 | Training batch size |
| `num_epochs` | 5 | Number of training epochs |
| `num_diffusion_steps` | 200 | Number of diffusion timesteps |
| `beta_start` | 1e-4 | Starting noise level |
| `beta_end` | 0.02 | Ending noise level |
| `hidden_dim` | 256 | Hidden layer dimension |
| `num_hidden_layers` | 3 | Number of hidden layers |
| `dropout` | 0.0 | Dropout rate |
| `num_samples` | 1000 | Default number of samples to generate |
| `device` | "cpu" | Device for computation |

## 📊 Advanced Usage

### Custom Data Preprocessing

```python
import pandas as pd
import numpy as np
from tabudiff.basic.utils import TabuDiffAPI

# Load and preprocess your data
df = pd.read_csv("data.csv")

# Handle missing values
df = df.fillna(df.mean())

# Normalize if needed
df = (df - df.mean()) / df.std()

# Train the model
api = TabuDiffAPI()
api.fit_dataframe(df)

# Generate synthetic data
synthetic_df = api.generate_dataframe(num_samples=5000)
```

### Model Persistence

```python
# Save trained model
api.save("my_model.pt")

# Load pre-trained model
api.load(feature_dim=10, path="my_model.pt")
synthetic_df = api.generate_dataframe(num_samples=1000)
```

### Batch Processing

```python
# Process multiple CSV files
csv_files = ["data1.csv", "data2.csv", "data3.csv"]
api = TabuDiffAPI()

for i, csv_file in enumerate(csv_files):
    output_file = f"synthetic_data_{i+1}.csv"
    api.fit_csv_and_generate_csv(csv_file, output_file, num_samples=2000)
```

## 🔧 API Reference

### TabuDiffAPI

The main API class for TabuDiff Basic.

#### Methods

- `fit_dataframe(df: pd.DataFrame) -> nn.Module`: Train the model on a DataFrame
- `fit_csv(csv_path: str) -> nn.Module`: Train the model on a CSV file
- `generate_dataframe(num_samples: int = None) -> pd.DataFrame`: Generate synthetic data
- `save(path: str = None)`: Save the trained model
- `load(feature_dim: int, path: str = None) -> nn.Module`: Load a pre-trained model
- `fit_csv_and_generate_csv(input_csv: str, output_csv: str, num_samples: int = None) -> str`: Complete pipeline

## 🎯 Use Cases

TabuDiff Basic is perfect for:

- **Data Augmentation**: Generate additional training samples for machine learning
- **Privacy Protection**: Create synthetic data that preserves statistical properties
- **Testing & Development**: Generate test data for application development
- **Research**: Explore data distributions and relationships
- **Data Sharing**: Share synthetic versions of sensitive datasets

## 🚀 Performance Tips

1. **GPU Acceleration**: Set `device="cuda"` for faster training on GPU
2. **Batch Size**: Increase `batch_size` for better GPU utilization
3. **Epochs**: More epochs generally improve quality but increase training time
4. **Diffusion Steps**: More steps improve quality but slow down generation
5. **Model Size**: Larger `hidden_dim` and `num_hidden_layers` for complex data

## 🔍 Example: Complete Workflow

```python
import pandas as pd
import numpy as np
from tabudiff.basic.utils import TabuDiffAPI
from tabudiff.basic.config import TabuDiffConfig

# Create sample data
np.random.seed(42)
data = {
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'score': np.random.uniform(0, 100, 1000)
}
df = pd.DataFrame(data)

# Configure the model
config = TabuDiffConfig(
    learning_rate=5e-4,
    batch_size=128,
    num_epochs=10,
    hidden_dim=512,
    num_hidden_layers=4,
    device="cpu"
)

# Train and generate
api = TabuDiffAPI(config)
api.fit_dataframe(df)

# Generate synthetic data
synthetic_df = api.generate_dataframe(num_samples=2000)

# Compare statistics
print("Original data statistics:")
print(df.describe())
print("\nSynthetic data statistics:")
print(synthetic_df.describe())

# Save synthetic data
synthetic_df.to_csv("synthetic_data.csv", index=False)
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `batch_size` or `hidden_dim`
2. **Slow Training**: Use GPU (`device="cuda"`) or reduce `num_diffusion_steps`
3. **Poor Quality**: Increase `num_epochs` or model size
4. **Convergence Issues**: Adjust `learning_rate` or add `dropout`

### Error Messages

- `"Model not trained"`: Call `fit_dataframe()` or `fit_csv()` before generating
- `"Nothing to save"`: Train the model before saving
- `CUDA out of memory`: Reduce batch size or use CPU

## 📈 Limitations

TabuDiff Basic is designed for simplicity and includes some limitations:

- **Data Types**: Currently supports only numerical data
- **Categorical Variables**: Not natively supported (requires encoding)
- **Large Datasets**: May require memory optimization for very large datasets
- **Complex Relationships**: Basic MLP may not capture all complex interactions

For advanced features like categorical support, attention mechanisms, and enterprise-grade capabilities, consider upgrading to **TabuDiff Premium**.

## 🔗 Next Steps

- Explore **TabuDiff Premium** for advanced features
- Check out the examples in `basic/examples/`
- Read the full API documentation
- Join the community for support and feature requests

---

**TabuDiff Basic** - Simple, powerful tabular data generation with diffusion models.
