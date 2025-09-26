import numpy as np
from typing import List, Dict
from .factory import CTGANFactory


class CTGANAPI:
    """High-level API for CTGAN synthetic tabular data generation"""

    def __init__(self):
        self.trainer = None
        self.is_fitted = False

    def fit(
        self,
        data: np.ndarray,
        continuous_cols: list,
        categorical_cols: list,
        epochs: int = 1000,
        **kwargs,
    ) -> dict:
        """Fit the CTGAN model to data"""

        # Convert column indices to counts
        n_cont = len(continuous_cols)
        n_cat = len(categorical_cols)

        # Extract verbose parameter
        verbose = kwargs.pop("verbose", True)

        # Create model
        self.trainer = CTGANFactory.create_model(
            continuous_dims=list(range(n_cont)),
            categorical_dims=list(range(n_cat)),
            **kwargs,
        )

        # Fit the model
        losses = self.trainer.fit(data, epochs=epochs, verbose=verbose)
        self.is_fitted = True

        return losses

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")

        return self.trainer.generate(n_samples)

    def save(self, filepath: str):
        """Save the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        self.trainer.save_model(filepath)

    def load(self, filepath: str):
        """Load a saved model"""
        if self.trainer is None:
            # Create a basic trainer if none exists
            # We need to infer the dimensions from the saved model
            import torch

            checkpoint = torch.load(filepath, map_location="cpu")
            config = checkpoint.get("config")
            if config is None:
                raise ValueError("Cannot load model: config not found in saved file")

            self.trainer = CTGANFactory.create_model(
                continuous_dims=config.continuous_dims,
                categorical_dims=config.categorical_dims,
            )

        self.trainer.load_model(filepath)
        self.is_fitted = True
