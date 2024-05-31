"""
Functions used to help OG-USA configure to COMP
"""

try:
    import boto3
except ImportError:
    boto3 = None
import gzip
import os
import pandas as pd
from taxcalc import Policy
from collections import defaultdict
from pathlib import Path
import warnings

try:
    from s3fs import S3FileSystem
except ImportError as ie:
    S3FileSystem = None

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
PUF_S3_FILE_LOCATION = os.environ.get(
    "PUF_S3_LOCATION", "s3://ospc-data-files/puf.20210720.csv.gz"
)
TMD_S3_FILE_LOCATION = os.environ.get(
    "TMD_S3_LOCATION", "s3://ospc-data-files/puf.20210720.csv.gz"
)
TC_LAST_YEAR = Policy.LAST_BUDGET_YEAR

POLICY_SCHEMA = {
    "labels": {
        "year": {
            "type": "int",
            "validators": {
                "choice": {
                    "choices": [yr for yr in range(2013, TC_LAST_YEAR + 1)]
                }
            },
        },
        "MARS": {
            "type": "str",
            "validators": {
                "choice": {
                    "choices": [
                        "single",
                        "mjoint",
                        "mseparate",
                        "headhh",
                        "widow",
                    ]
                }
            },
        },
        "idedtype": {
            "type": "str",
            "validators": {
                "choice": {
                    "choices": [
                        "med",
                        "sltx",
                        "retx",
                        "cas",
                        "misc",
                        "int",
                        "char",
                    ]
                }
            },
        },
        "EIC": {
            "type": "str",
            "validators": {
                "choice": {"choices": ["0kids", "1kid", "2kids", "3+kids"]}
            },
        },
        "data_source": {
            "type": "str",
            "validators": {"choice": {"choices": ["PUF", "CPS", "other"]}},
        },
    },
    "additional_members": {
        "section_1": {"type": "str"},
        "section_2": {"type": "str"},
        "start_year": {"type": "int"},
        "checkbox": {"type": "bool"},
    },
}


def retrieve_puf(
    puf_s3_file_location=PUF_S3_FILE_LOCATION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
):
    """
    Function for retrieving the PUF from the OSPC S3 bucket
    """
    s3_reader_installed = S3FileSystem is not None
    has_credentials = (
        aws_access_key_id is not None and aws_secret_access_key is not None
    )
    if puf_s3_file_location and has_credentials and s3_reader_installed:
        print("Reading puf from S3 bucket.", puf_s3_file_location)
        fs = S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
        )
        with fs.open(PUF_S3_FILE_LOCATION) as f:
            # Skips over header from top of file.
            puf_df = pd.read_csv(f, compression="gzip")
        return puf_df
    elif Path("puf.csv.gz").exists():
        print("Reading puf from puf.csv.gz.")
        return pd.read_csv("puf.csv.gz", compression="gzip")
    elif Path("puf.csv").exists():
        print("Reading puf from puf.csv.")
        return pd.read_csv("puf.csv")
    else:
        warnings.warn(
            f"PUF file not available (has_credentials={has_credentials}, "
            f"s3_reader_installed={s3_reader_installed})"
        )
        return None


def retrieve_tmd(
    tmd_s3_file_location=TMD_S3_FILE_LOCATION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
):
    """
    Function for retrieving the TMD from the S3 bucket
    """
    s3_reader_installed = S3FileSystem is not None
    has_credentials = (
        aws_access_key_id is not None and aws_secret_access_key is not None
    )
    if tmd_s3_file_location and has_credentials and s3_reader_installed:
        print("Reading tmd from S3 bucket.", tmd_s3_file_location)
        fs = S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
        )
        with fs.open(tmd_s3_file_location) as f:
            # Skips over header from top of file.
            tmd_df = pd.read_csv(f)
        return tmd_df
    elif Path("tmd.csv.gz").exists():
        print("Reading tmd from tmd.csv.gz.")
        return pd.read_csv("tmd.csv.gz", compression="gzip")
    elif Path("tmd.csv").exists():
        print("Reading tmd from tmd.csv.")
        return pd.read_csv("tmd.csv")
    else:
        warnings.warn(
            f"TMD file not available (tmd_location={tmd_s3_file_location}, "
            f"has_credentials={has_credentials}, "
            f"s3_reader_installed={s3_reader_installed})"
        )
        return None
