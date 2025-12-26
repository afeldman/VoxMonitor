"""Command-line interface for VoxMonitor.

Provides utilities for dataset download and training.
"""

from pathlib import Path
import argparse
from loguru import logger

from voxmonitor.zenodo_download import download_dataset


def download_dataset_cli():
    """CLI command to download Soundwell dataset."""
    parser = argparse.ArgumentParser(description="Download Soundwell dataset from Zenodo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification",
    )

    args = parser.parse_args()

    logger.info("Soundwell Dataset Download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Source: https://zenodo.org/records/8252482")

    dataset_dir = download_dataset(
        "soundwell",
        output_dir=args.output_dir,
        checksum_verify=not args.no_verify,
    )

    logger.info("=" * 60)
    logger.info(f"✅ Download complete: {dataset_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Run: uv run voxmonitor-train-soundwell")
    logger.info("  2. Or: voxmonitor-train-soundwell --epochs 100 --batch-size 16")
    logger.info("=" * 60)
