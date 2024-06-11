import setuptools

with open("README.md", "r") as readme_file:
    longdesc = readme_file.read()

setuptools.setup(
    name="ogusa",
    version="0.1.10",
    author="Jason DeBacker and Richard W. Evans",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="USA calibration for OG-Core",
    long_description_content_type="text/markdown",
    long_description=longdesc,
    keywords="USA calibration of large scale overlapping generations model of fiscal policy",
    url="https://github.com/PSLmodels/OG-USA/",
    download_url="https://github.com/PSLmodels/OG-USA/",
    project_urls={
        "Issue Tracker": "https://github.com/PSLmodels/OG-USA/issues",
    },
    packages=["ogusa"],
    package_data={
        "ogusa": [
            "ogusa_default_parameters.json",
            "psid_lifetime_income.csv.gz",
        ]
    },
    include_packages=True,
    python_requires=">=3.7.7, <3.12",
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
        "pandas-datareader",
        "xlwt",
        "openpyxl>=3.1.2",
        "statsmodels",
        "linearmodels",
        "wheel",
        "ogcore",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Common Public License",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tests_require=["pytest"],
)
