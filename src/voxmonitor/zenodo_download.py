"""Zenodo dataset download utility.

Provides functions to download datasets from Zenodo with progress tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import hashlib

import requests
from loguru import logger


def get_zenodo_record_info(record_id: str | int) -> dict:
    """Fetch metadata for a Zenodo record.

    Args:
        record_id: Zenodo record ID (e.g., "8252482").

    Returns:
        dict: Record metadata including files and download URLs.
    """
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def download_zenodo_record(
    record_id: str | int,
    output_dir: str | Path = "datasets",
    files: Optional[list[str]] = None,
    checksum_verify: bool = True,
) -> Path:
    """Download files from a Zenodo record.

    Args:
        record_id: Zenodo record ID.
        output_dir: Directory to save downloaded files.
        files: Optional list of specific filenames to download.
               If None, downloads all files.
        checksum_verify: Whether to verify checksums.

    Returns:
        Path: Directory containing downloaded files.

    Raises:
        requests.HTTPError: If download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching Zenodo record {record_id}...")
    record = get_zenodo_record_info(record_id)

    # Extract file information
    files_to_download = record["files"]
    
    if files:
        files_to_download = [f for f in files_to_download if f["key"] in files]

    total_size = sum(f["size"] for f in files_to_download)
    logger.info(
        f"Downloading {len(files_to_download)} files "
        f"({total_size / 1e9:.2f} GB) from record {record_id}..."
    )

    for file_info in files_to_download:
        filename = file_info["key"]
        file_url = file_info["links"]["self"]
        file_size = file_info["size"]
        file_path = output_dir / filename

        # Skip if already exists and verified
        if file_path.exists():
            if checksum_verify and "checksum" in file_info:
                logger.info(f"Verifying {filename}...")
                if _verify_checksum(file_path, file_info["checksum"]):
                    logger.info(f"✅ {filename} already exists (verified)")
                    continue
            else:
                logger.info(f"✅ {filename} already exists (skipping)")
                continue

        logger.info(f"Downloading {filename} ({file_size / 1e9:.2f} GB)...")
        _download_file(file_url, file_path, file_size)

        if checksum_verify and "checksum" in file_info:
            if _verify_checksum(file_path, file_info["checksum"]):
                logger.info(f"✅ {filename} verified")
            else:
                logger.warning(f"⚠️  Checksum mismatch for {filename}")

    logger.info(f"✅ Download complete: {output_dir}")
    return output_dir


def _download_file(url: str, output_path: Path, total_size: int) -> None:
    """Download a single file with progress bar.

    Args:
        url: File URL.
        output_path: Path to save file.
        total_size: Expected file size in bytes.
    """
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    downloaded = 0
    chunk_size = 8192

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                print(
                    f"\r  Progress: {percent:.1f}% ({downloaded / 1e9:.2f} GB / {total_size / 1e9:.2f} GB)",
                    end="",
                    flush=True,
                )
    print()  # newline after progress


def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify file checksum.

    Args:
        file_path: Path to file.
        expected_checksum: Expected checksum (format: "algorithm:hash").

    Returns:
        bool: True if checksum matches.
    """
    if ":" in expected_checksum:
        algorithm, expected_hash = expected_checksum.split(":", 1)
    else:
        # Assume MD5 if no algorithm specified
        algorithm = "md5"
        expected_hash = expected_checksum

    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)

    computed_hash = hash_obj.hexdigest()
    return computed_hash == expected_hash


# Zenodo Record IDs for datasets
ZENODO_RECORDS = {
    "soundwell": "8252482",  # Soundwell pig vocalization dataset
}


def download_dataset(
    dataset_name: str,
    output_dir: str | Path = "datasets",
    **kwargs,
) -> Path:
    """Download a known dataset from Zenodo.

    Args:
        dataset_name: Name of dataset (e.g., "soundwell").
        output_dir: Directory to save files.
        **kwargs: Additional arguments to pass to download_zenodo_record.

    Returns:
        Path: Directory containing downloaded files.

    Raises:
        KeyError: If dataset_name not found.
    """
    if dataset_name not in ZENODO_RECORDS:
        raise KeyError(
            f"Unknown dataset: {dataset_name}. Available: {list(ZENODO_RECORDS.keys())}"
        )

    record_id = ZENODO_RECORDS[dataset_name]
    return download_zenodo_record(record_id, output_dir, **kwargs)
