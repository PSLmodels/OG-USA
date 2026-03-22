"""
Data retrieval helpers for the OG-USA app.

Adapted from cs-config/cs_config/helpers.py with Compute Studio
dependencies removed.
"""
import os
import warnings
from pathlib import Path

import pandas as pd

try:
    from s3fs import S3FileSystem

    _S3_AVAILABLE = True
except ImportError:
    _S3_AVAILABLE = False

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
PUF_S3_FILE_LOCATION = os.environ.get(
    "PUF_S3_LOCATION", "s3://ospc-data-files/puf.20210720.csv.gz"
)
TMD_S3_FILE_LOCATION = os.environ.get("TMD_S3_LOCATION", "")


def retrieve_puf(
    puf_s3_file_location=PUF_S3_FILE_LOCATION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
):
    """
    Retrieve the PUF from S3 or a local file.

    Returns a DataFrame or None (caller should fall back to CPS).
    """
    has_credentials = bool(aws_access_key_id and aws_secret_access_key)
    if puf_s3_file_location and has_credentials and _S3_AVAILABLE:
        print(f"Reading PUF from S3: {puf_s3_file_location}")
        fs = S3FileSystem(
            key=aws_access_key_id, secret=aws_secret_access_key
        )
        with fs.open(puf_s3_file_location) as f:
            return pd.read_csv(f, compression="gzip")
    elif Path("puf.csv.gz").exists():
        print("Reading PUF from puf.csv.gz")
        return pd.read_csv("puf.csv.gz", compression="gzip")
    elif Path("puf.csv").exists():
        print("Reading PUF from puf.csv")
        return pd.read_csv("puf.csv")
    else:
        warnings.warn(
            f"PUF file not available "
            f"(has_credentials={has_credentials}, "
            f"s3_available={_S3_AVAILABLE}). Falling back to CPS."
        )
        return None


def retrieve_tmd(
    tmd_s3_file_location=TMD_S3_FILE_LOCATION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
):
    """
    Retrieve the TMD from S3 or a local file.

    Returns a DataFrame or None (caller should fall back to CPS).
    """
    has_credentials = bool(aws_access_key_id and aws_secret_access_key)
    if tmd_s3_file_location and has_credentials and _S3_AVAILABLE:
        print(f"Reading TMD from S3: {tmd_s3_file_location}")
        fs = S3FileSystem(
            key=aws_access_key_id, secret=aws_secret_access_key
        )
        with fs.open(tmd_s3_file_location) as f:
            return pd.read_csv(f)
    elif Path("tmd.csv.gz").exists():
        print("Reading TMD from tmd.csv.gz")
        return pd.read_csv("tmd.csv.gz", compression="gzip")
    elif Path("tmd.csv").exists():
        print("Reading TMD from tmd.csv")
        return pd.read_csv("tmd.csv")
    else:
        warnings.warn(
            "TMD file not available. Falling back to CPS."
        )
        return None
