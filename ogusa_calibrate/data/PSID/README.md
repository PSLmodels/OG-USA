# Extracting PSID data with `psidR`

Steps:

1. Load the R environment by running these commands from the repo's root directory:
```
conda env create -f environment-r.yml
conda activate ogusa-calibrate-dev-r
```

2. Open R by typing `R` in the shell.

3. Install `psidR` with the command `install.packages("psidR")`. If you need a new feature not yet on CRAN, run `library(devtools)` then `install_github("floswald/psidR")` (though this can create issues).

4. Executive the R script by running `Rscript psid_download.R` from terminal in this folder.
