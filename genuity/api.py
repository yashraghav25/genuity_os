from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from data_processor.data_preprocess import TabularPreprocessor
from data_processor.data_postprocess import TabularPostprocessor

# Backends
from core_generator.ctgan.ctgan.utils.api import CTGANAPI
from core_generator.tvae.tvae.utils.api import TVAEAPI
from core_generator.tabudiff.basic.utils.api import TabuDiffAPI


BackendName = Literal["ctgan", "tvae", "tabudiff"]


@dataclass
class BackendConfig:
    name: BackendName = "ctgan"
    epochs: int = 1000
    # Extra kwargs forwarded to backend factory/trainers
    extra: Optional[dict] = None


class SyntheticTabular:
    """
    High-level API that: (1) preprocesses a DataFrame/CSV, (2) fits a chosen
    generative model backend, and (3) generates synthetic data and reconstructs
    columns back to the original feature space when possible.
    """

    def __init__(
        self,
        backend: BackendConfig | None = None,
        encoding_strategy: str = "onehot",
        scaler_type: Optional[str] = "standard",
        n_pca_components: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.backend = backend or BackendConfig()
        self.verbose = verbose
        # Pre/post components
        self.pre = TabularPreprocessor(
            encoding_strategy=encoding_strategy,
            scaler_type=scaler_type,
            n_pca_components=n_pca_components,
            verbose=verbose,
        )
        self.post: Optional[TabularPostprocessor] = None
        # Model API
        self.model = None
        self.is_fitted = False

    def _init_backend(self, n_cont: int, n_cat: int) -> None:
        name = self.backend.name.lower()
        if name == "ctgan":
            self.model = CTGANAPI()
        elif name == "tvae":
            self.model = TVAEAPI(model_type="basic")
        elif name == "tabudiff":
            self.model = TabuDiffAPI()
        else:
            raise ValueError(f"Unknown backend: {self.backend.name}")

    def fit_dataframe(self, df: pd.DataFrame) -> None:
        result = self.pre.fit_transform(df)
        pre_df = result["preprocessed"]
        cont_cols = result["continuous"].columns.tolist()
        enc_cols = result["categorical"].columns.tolist()

        self._init_backend(n_cont=len(cont_cols), n_cat=len(enc_cols))

        data = pre_df.values.astype(np.float32)
        extra = (self.backend.extra or {}).copy()
        epochs = self.backend.epochs

        if isinstance(self.model, CTGANAPI):
            self.model.fit(
                data=data,
                continuous_cols=list(range(len(cont_cols))),
                categorical_cols=list(range(len(enc_cols))),
                epochs=epochs,
                **extra,
            )
        elif isinstance(self.model, TVAEAPI):
            self.model.fit(
                data=data,
                continuous_cols=list(range(len(cont_cols))),
                categorical_cols=list(range(len(enc_cols))),
                epochs=epochs,
                **extra,
            )
        elif isinstance(self.model, TabuDiffAPI):
            # TabuDiffAPI operates on DataFrames directly
            self.model.fit_dataframe(pre_df)
        else:
            raise RuntimeError("Model backend not initialized")

        self.post = TabularPostprocessor(
            preprocessor_object=self.pre, verbose=self.verbose
        )
        self.is_fitted = True

    def generate_dataframe(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Must fit before generating")

        if isinstance(self.model, TabuDiffAPI):
            syn_pre = self.model.generate_dataframe(num_samples=n_samples)
        else:
            arr = self.model.generate(n_samples)
            syn_pre = pd.DataFrame(arr, columns=self.pre.feature_names_)

        recon = self.post.inverse_transform_modified_data(syn_pre)
        return recon

    def fit_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        self.fit_dataframe(df)

    def fit_csv_and_generate_csv(
        self, input_csv: str, output_csv: str, num_samples: Optional[int] = None
    ) -> str:
        df = pd.read_csv(input_csv)
        self.fit_dataframe(df)
        num = num_samples or len(df)
        syn = self.generate_dataframe(num)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        syn.to_csv(output_csv, index=False)
        return output_csv


def convert_csv_to_synthetic(
    input_csv: str,
    output_csv: str,
    backend: BackendName = "ctgan",
    epochs: int = 1000,
    encoding_strategy: str = "onehot",
    scaler_type: Optional[str] = "standard",
    n_pca_components: Optional[int] = None,
    num_samples: Optional[int] = None,
    verbose: bool = True,
) -> str:
    api = SyntheticTabular(
        backend=BackendConfig(name=backend, epochs=epochs),
        encoding_strategy=encoding_strategy,
        scaler_type=scaler_type,
        n_pca_components=n_pca_components,
        verbose=verbose,
    )
    return api.fit_csv_and_generate_csv(input_csv, output_csv, num_samples=num_samples)
