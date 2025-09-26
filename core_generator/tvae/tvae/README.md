# TVAE (Basic)

Basic implementation of TVAE (Tabular Variational AutoEncoder) for generating synthetic tabular data. For advanced features (VampPrior, β-divergence, transformer attention, GMM sampling, etc.), use the `tvae_premium` package in this repository.

## 🚀 Features

### Core Features
- **Tabular Variational AutoEncoder**: High-fidelity synthetic data for mixed-type tabular datasets
- **Mixed Data Types**: Support for continuous and categorical features
- **Production-Ready**: Modular, robust, and extensible

### Premium Features (moved)
- Advanced features are available in `tvae_premium`.

## 📁 Project Structure

```
tvae/
├── config/
│   ├── __init__.py
│   └── config.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   └── components.py
├── samplers/
│   ├── __init__.py
│   ├── sampler.py
│   └── sampling_strategies.py
├── trainers/
│   ├── __init__.py
│   ├── trainer.py
│   └── training_strategies.py
├── utils/
│   ├── __init__.py
│   ├── factory.py
│   └── api.py
├── examples/
│   └── basic_usage.py
├── tests/
│   └── test_basic.py
├── requirements.txt
├── README.md
└── __init__.py
```

## 📖 Quick Start

```python
import numpy as np
from tvae import TVAEAPI

data = np.random.randn(1000, 5)
continuous_cols = [0, 1, 2]
categorical_cols = [3, 4]

tvae = TVAEAPI(model_type="basic")
losses = tvae.fit(
    data=data,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    epochs=1000,
)
samples = tvae.generate(1000)
```

To use premium features:

```python
from tvae_premium import TVAEAPI

tvae = TVAEAPI(model_type="premium")
```