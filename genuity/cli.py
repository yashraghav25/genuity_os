import argparse

from .api import convert_csv_to_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="genuity-synth",
        description="Generate synthetic tabular data from a real CSV using Genuity Synthetic.",
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to write synthetic CSV file")
    parser.add_argument(
        "--backend",
        choices=["ctgan", "tvae", "tabudiff"],
        default="ctgan",
        help="Model backend to use",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument(
        "--encoding",
        default="onehot",
        choices=[
            "onehot",
            "ordinal",
            "label",
            "binary",
            "embedding",
            "frequency",
            "hash",
            "target",
        ],
        help="Categorical encoding strategy",
    )
    parser.add_argument(
        "--scaler",
        default="standard",
        choices=["standard", "minmax", "robust", "maxabs", "none"],
        help="Scaling strategy for continuous features",
    )
    parser.add_argument(
        "--pca", type=int, default=None, help="Number of PCA components (optional)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")

    args = parser.parse_args()

    scaler = None if args.scaler == "none" else args.scaler
    verbose = not args.quiet

    convert_csv_to_synthetic(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        backend=args.backend,
        epochs=args.epochs,
        encoding_strategy=args.encoding,
        scaler_type=scaler,
        n_pca_components=args.pca,
        num_samples=args.num_samples,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
