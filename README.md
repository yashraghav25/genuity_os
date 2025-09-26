# Genuity Synthetic

Convert real CSV data (numeric + categorical) into synthetic tabular datasets using CTGAN, TVAE, or TabuDiff.

## Install (local)
```bash
pip install -e .
```

## Quick start
```python
from genuity import SyntheticTabular

api = SyntheticTabular()  # defaults to CTGAN
api.fit_csv("data.csv")
synthetic_df = api.generate_dataframe(1000)
synthetic_df.to_csv("synthetic.csv", index=False)
```

## CLI
```bash
genuity-synth data.csv synthetic.csv --backend ctgan --epochs 500 --encoding onehot
```

## Backends
- ctgan: conditional GAN
- tvae: variational autoencoder (basic)
- tabudiff: diffusion for tabular data (basic)

## License
Apache-2.0
