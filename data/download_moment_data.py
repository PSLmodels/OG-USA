"""
Download and trim survey data used by ogusa.compute_moments.

This script creates local CSV files in ogusa/data/CPS and ogusa/data/SCF so
moments can be computed without repeatedly downloading raw CPS ASEC and SCF
files.
"""

import io
import os
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ogusa", "data")
)
CPS_DIR = os.path.join(DATA_DIR, "CPS")
SCF_DIR = os.path.join(DATA_DIR, "SCF")

CPS_ASEC_URLS = {
    2022: "https://data.nber.org/cps_supp_1/raw/2022/march/asecpub22csv.zip",
    2023: "https://data.nber.org/cps_supp_1/raw/2023/march/asecpub23csv.zip",
}

SCF_YEARS = [2019, 2016, 2013, 2010, 2007]
# Total pre-tax household income and its components from the SCF summary
# extract. The components let moments use income net of transfers, which is
# the model's before-tax income concept (capital plus labor income).
SCF_INCOME_COLUMNS = [
    "income",
    "wageinc",
    "bussefarminc",
    "intdivinc",
    "kginc",
    "ssretinc",
    "transfothinc",
]
SCF_CPI_2019 = {
    2019: 100.000,
    2016: 94.06403464,
    2013: 88.83067929,
    2010: 84.09125952,
    2007: 80.05995867,
}


def _download_zip(url):
    """
    Download a zip archive and return a ZipFile object.
    """
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        zip_bytes = response.read()
    return zipfile.ZipFile(io.BytesIO(zip_bytes))


def _file_size_mb(path):
    """
    Return the file size in MB.
    """
    return os.path.getsize(path) / (1024**2)


def _report_file(path):
    """
    Print output file size and whether it exceeds 50MB.
    """
    size_mb = _file_size_mb(path)
    suffix = " exceeds 50MB" if size_mb > 50 else ""
    print(f"Wrote {path} ({size_mb:.2f}MB){suffix}")


def download_cps():
    """
    Download and trim CPS ASEC person files.
    """
    os.makedirs(CPS_DIR, exist_ok=True)
    for year, url in CPS_ASEC_URLS.items():
        print(f"Downloading CPS ASEC {year}...")
        with _download_zip(url) as zip_file:
            person_files = [
                name
                for name in zip_file.namelist()
                if name.lower().startswith("pppub")
                and name.lower().endswith(".csv")
            ]
            if len(person_files) != 1:
                raise ValueError(
                    f"Expected one CPS person file for {year}, found "
                    f"{len(person_files)}."
                )
            with zip_file.open(person_files[0]) as person_file:
                cps = pd.read_csv(
                    person_file,
                    usecols=["A_AGE", "HRSWK", "WKSWORK", "A_FNLWGT"],
                )

        cps.rename(
            columns={
                "A_AGE": "age",
                "HRSWK": "hours_per_week",
                "WKSWORK": "weeks_worked",
                "A_FNLWGT": "weight",
            },
            inplace=True,
        )
        cps["year"] = year
        cps = cps[["year", "age", "hours_per_week", "weeks_worked", "weight"]]
        cps = cps[pd.to_numeric(cps["age"], errors="coerce") >= 20].copy()

        output_path = os.path.join(CPS_DIR, f"cps_asec_hours_{year}.csv")
        cps.to_csv(output_path, index=False)
        _report_file(output_path)


def download_scf():
    """
    Download and trim SCF summary files.
    """
    os.makedirs(SCF_DIR, exist_ok=True)
    for year in SCF_YEARS:
        print(f"Downloading SCF {year}...")
        url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
        with _download_zip(url) as zip_file:
            dta_files = [
                name
                for name in zip_file.namelist()
                if os.path.basename(name).lower().startswith("rscfp")
                and str(year) in os.path.basename(name)
                and name.lower().endswith(".dta")
            ]
            if len(dta_files) != 1:
                raise ValueError(
                    f"Expected one SCF summary file for {year}, found "
                    f"{len(dta_files)}."
                )
            with zip_file.open(dta_files[0]) as dta_file:
                scf = pd.read_stata(
                    dta_file,
                    columns=["age", "networth", "wgt"] + SCF_INCOME_COLUMNS,
                )

        scf["year"] = year
        deflator = 100.0 / SCF_CPI_2019[year]
        scf["networth_infadj"] = scf["networth"] * deflator
        scf["income_infadj"] = scf["income"] * deflator
        scf = scf[
            ["year", "age", "networth", "networth_infadj", "wgt"]
            + SCF_INCOME_COLUMNS
            + ["income_infadj"]
        ]

        output_path = os.path.join(SCF_DIR, f"scf_wealth_{year}.csv")
        scf.to_csv(output_path, index=False)
        _report_file(output_path)


def main():
    """
    Download all local moment data.
    """
    download_cps()
    download_scf()


if __name__ == "__main__":
    main()
