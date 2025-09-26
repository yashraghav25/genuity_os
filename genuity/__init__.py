"""
Genuity Synthetic - Unified API to convert real CSV data (numeric + categorical)
into synthetic tabular datasets using CTGAN, TVAE, or TabuDiff backends.

Public API:
- SyntheticTabular: high-level class to fit/generate synthetic data
- convert_csv_to_synthetic: one-shot helper to read CSV and write synthetic CSV
"""

from .api import SyntheticTabular, convert_csv_to_synthetic

__all__ = [
    "SyntheticTabular",
    "convert_csv_to_synthetic",
]

__version__ = "0.1.0"
