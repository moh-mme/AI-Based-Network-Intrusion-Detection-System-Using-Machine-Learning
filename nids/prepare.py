"""Step 1: clean the raw CSV files.

    python -m nids.prepare [--config CONFIG]
"""

from __future__ import annotations

import argparse

from . import setup_logging
from .config import load_config
from .data import build_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean the raw CIC-IDS2017 CSV files into one dataset.")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    profile = build_dataset(cfg)

    print()
    print(f"Data source        : {profile['data_source']}")
    print(f"Rows read          : {profile['rows_read']:,}")
    print(f"Empty rows removed : {profile['rows_empty']:,}")
    print(f"NaN/inf removed    : {profile['rows_invalid']:,}")
    print(f"Duplicates removed : {profile['duplicates_removed']:,}")
    print(f"Final rows         : {profile['rows_final']:,}")
    print(f"Features           : {len(profile['features'])} "
          f"({len(profile['constant_features'])} constant)")
    print("\nRows per category:")
    for category, count in profile["category_counts"].items():
        print(f"  {category:<14}{count:>12,}")
    print(f"\nSaved to {cfg.paths.processed_dir}")


if __name__ == "__main__":
    main()
