import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    longdesc = fh.read()

setuptools.setup(
    name="ogusa",
    version="0.1.0",
    author="Jason DeBacker and Richard W. Evans",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="USA calibration for OG-Core",
    long_description_content_type="text/markdown",
    long_description=longdesc,
    url="https://github.com/PSLmodels/OG-USA/",
    download_url="https://github.com/PSLmodels/OG-USA/",
    project_urls={
        "Issue Tracker": "https://github.com/PSLmodels/OG-USA/issues",
    },
    packages=["ogusa"],
    package_data={
        "ogusa": [
            "ogusa_default_parameters.json",
            "data/PSID/*"
        ]
    },
    include_packages=True,
    python_requires=">=3.7.7, <3.11",
    install_requires=[
        "numpy",
        "psutil",
        "scipy>=1.5.0",
        "pandas>=1.2.5",
        "matplotlib",
        "dask>=2.30.0",
        "distributed>=2.30.1",
        "paramtools>=0.15.0",
        "taxcalc>=3.0.0",
        "requests",
        "rpy2",
        "pandas-datareader",
        "xlwt",
        "openpyxl>=3.1.2",
        "statsmodels",
        "linearmodels",
        "ogcore"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: CC0 1.0 Universal public domain dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tests_require=["pytest"],
)
