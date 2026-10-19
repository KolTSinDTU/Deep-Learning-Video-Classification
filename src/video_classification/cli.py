"""Configuration preflight; this command does not train a model."""

import argparse
import json
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-data", action="store_true", help="Also require the dataset root directory to exist")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    root = Path(config["dataset"]["root"])
    if args.require_data and not root.is_dir():
        parser.error(f"Dataset directory does not exist: {root}. See data/README.md")
    print(json.dumps(config, indent=2))
    print("Configuration valid. This command checks configuration only; it does not train or evaluate.")
    print(f"Dataset root {'exists (contents not validated)' if root.is_dir() else 'is not present yet'}: {root}")


if __name__ == "__main__":
    main()
