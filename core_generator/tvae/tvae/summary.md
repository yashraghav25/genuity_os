# TVAE Library Summary

## Overview
TVAE (Basic) provides a lightweight TVAE suitable for standard use cases. Advanced 2024-2025 research features have been moved to `tvae_premium`.

## Architecture
- **Encoder**: Maps tabular data to a latent distribution. Supports standard Gaussian and VampPrior.
- **Decoder**: Reconstructs data from latent vectors. Supports multi-head (continuous/categorical) and transformer attention.
- **Sampler**: Handles latent space sampling, including GMM and random strategies.
- **Trainer**: Manages training, loss computation, and advanced feature integration.
- **Quality Gating**: Multi-metric system for filtering and validating synthetic data.

## Premium Features
See `tvae_premium` for VampPrior, β-divergence, transformer attention, GMM clustering/sampling, cyclical KL, gradient noise, Wasserstein loss, and quality gating.

## Usage
- **API**: `TVAEAPI` with `fit`, `generate`, `save`, `load`, and feature importance methods.
- **Premium**: Import from `tvae_premium` for advanced features.
- **Input**: Preprocessed tabular data (continuous/categorical, normalized/encoded).
- **Output**: High-fidelity synthetic data, with robust categorical assignment.

## Example
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

## Directory Structure
```
tvae/
├── config/
├── models/
├── samplers/
├── trainers/
├── utils/
├── examples/
├── tests/
├── requirements.txt
├── README.md
├── summary.md
└── __init__.py
```

## Production-Ready
- Modular, extensible, and robust
- Comprehensive error handling
- Ready for enterprise deployment and research