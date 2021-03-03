# Extracting PSID data with `psidR`

Steps for `conda` users:

1. Ensure you don't have R installed in your `conda` environment, e.g. by running `conda remove r-base`.

2. Install R with `sudo apt install -y r-base`. *Linux users should jump to the `Troubleshooting Linux` section below.*

3. Open R with `R`.

4. Install the `devtools` package with the R command `install.packages("devtools")`.

5. Install the `psidR` R package with the command `library(devtools); install_github("floswald/psidR")`.

6. Exit R with `q()`.

7. Run the script with the shell command `Rscript psid_download.R` from this folder.

## Troubleshooting Linux

If you use `conda` on Linux, these steps can avoid issues when trying to install `devtools`:

1. Install other necessary packages with `sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev`

2. Look for the location of the file `libicui18n.so.58` via the command `sudo find / -name "libicui18n.so.58"`. You should see some location of the pattern `*/anaconda3/lib/libicui18n.so.58`.

3. Add the location to your `LD_LIBRARY_PATH` variable with something like this command: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mghenis/anaconda3/lib`.

4. Move to step 3 in the main instructions above.
