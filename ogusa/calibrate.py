from ogusa import estimate_beta_j, bequest_transmission
from ogusa import macro_params, transfer_distribution, income
from ogusa import get_micro_data
import os
import numpy as np
from taxcalc import Records
from ogcore import txfunc, demographics
from ogcore.utils import safe_read_pickle, mkdirs
import pkg_resources


class Calibration:
    """OG-USA calibration class"""

    def __init__(
        self,
        p,
        estimate_tax_functions=False,
        estimate_beta=False,
        estimate_chi_n=False,
        estimate_pop=False,
        tax_func_path=None,
        iit_baseline=None,
        iit_reform={},
        guid="",
        data="cps",
        gfactors=None,
        weights=None,
        records_start_year=Records.CPSCSV_YEAR,
        client=None,
        num_workers=1,
        demographic_data_path=None,
        output_path=None,
    ):
        """
        Constructor for the Calibration class.  This class is used to find
        parameter values for the OG-USA model.

        Args:
            p (OG-USA Parameters object): parameters object
            estimate_tax_functions (bool): whether to estimate tax functions
            estimate_beta (bool): whether to estimate beta
            estimate_chi_n (bool): whether to estimate chi_n
            estimate_pop (bool): whether to estimate population
            tax_func_path (str): path to tax function parameters
            iit_baseline (dict): baseline policy to use
            iit_reform (dict): reform tax parameters
            guid (str): id for tax function parameters
            data (str or Pandas DataFrame): path or DataFrame with
                data for Tax-Calculator model
            gfactors (str or Pandas DataFrame ): path or DataFrame with
                growth factors for Tax-Calculator model
            weights (str or Pandas DataFrame): path or DataFrame with
                weights for Tax-Calculator model
            records_start_year (int): year micro data begins
            client (Dask client object): client
            num_workers (int): number of workers for Dask client
            output_path (str): path to save output to

        Returns:
            Calibration class object instance
        """
        # Create output_path if it doesn't exist
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        self.estimate_tax_functions = estimate_tax_functions
        self.estimate_beta = estimate_beta
        self.estimate_chi_n = estimate_chi_n
        self.estimate_pop = estimate_pop
        if estimate_tax_functions:
            if tax_func_path is not None:
                run_micro = False
            else:
                run_micro = True
            self.tax_function_params = self.get_tax_function_parameters(
                p,
                iit_baseline,
                iit_reform,
                guid,
                data,
                gfactors,
                weights,
                records_start_year,
                client,
                num_workers,
                run_micro=run_micro,
                tax_func_path=tax_func_path,
            )
        if self.estimate_beta:
            self.beta_j = estimate_beta_j.beta_estimate(self)
        # if estimate_chi_n:
        #     chi_n = self.get_chi_n()

        # Macro estimation
        self.macro_params = macro_params.get_macro_params()

        # eta estimation
        self.eta = transfer_distribution.get_transfer_matrix(
            p.J, p.lambdas, output_path=output_path
        )

        # zeta estimation
        self.zeta = bequest_transmission.get_bequest_matrix(
            p.J, p.lambdas, output_path=output_path
        )

        # demographics
        if estimate_pop:
            self.demographic_params = demographics.get_pop_objs(
                p.E,
                p.S,
                p.T,
                0,
                99,
                initial_data_year=p.start_year - 1,
                final_data_year=p.start_year,
                GraphDiag=False,
                download_path=demographic_data_path,
            )

            # demographics for 80 period lives (needed for getting e below)
            demog80 = demographics.get_pop_objs(
                20,
                80,
                p.T,
                0,
                99,
                initial_data_year=p.start_year - 1,
                final_data_year=p.start_year,
                GraphDiag=False,
            )

            # earnings profiles
            self.e = income.get_e_interp(
                p.S,
                self.demographic_params["omega_SS"],
                demog80["omega_SS"],
                p.lambdas,
                plot_path=output_path,
            )
        else:
            self.e = income.get_e_interp(
                p.S,
                p.omega_SS,
                p.omega_SS,
                p.lambdas,
                plot_path=output_path,
            )

    # Tax Functions
    def get_tax_function_parameters(
        self,
        p,
        iit_baseline=None,
        iit_reform={},
        guid="",
        data="",
        gfactors=None,
        weights=None,
        records_start_year=Records.CPSCSV_YEAR,
        client=None,
        num_workers=1,
        run_micro=False,
        tax_func_path=None,
    ):
        """
        Reads pickle file of tax function parameters or estimates the
        parameters from microsimulation model output.

        Args:
            p (OG-Core Parameters object): parameters object
            iit_baseline (dict): baseline policy to use
            iit_reform (dict): reform tax parameters
            guid (string): id for tax function parameters
            data (str or Pandas DataFrame): path or DataFrame with
                data for Tax-Calculator model
            gfactors (str or Pandas DataFrame ): path or DataFrame with
                growth factors for Tax-Calculator model
            weights (str or Pandas DataFrame): path or DataFrame with
                weights for Tax-Calculator model
            records_start_year (int): year micro data begins
            client (Dask client object): client
            num_workers (int): number of workers for Dask client
            run_micro (bool): whether to estimate parameters from
                microsimulation model
            tax_func_path (string): path where find or save tax
                function parameter estimates

        Returns:
            None

        """
        # set paths if none given
        if tax_func_path is None:
            if p.baseline:
                pckl = "TxFuncEst_baseline{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
                print("Using baseline tax parameters from ", tax_func_path)
            else:
                pckl = "TxFuncEst_policy{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
                print(
                    "Using reform policy tax parameters from ", tax_func_path
                )
        # create directory for tax function pickles to be saved to
        mkdirs(os.path.split(tax_func_path)[0])
        # If run_micro is false, check to see if parameters file exists
        # and if it is consistent with Specifications instance
        if not run_micro:
            dict_params, run_micro = self.read_tax_func_estimate(
                p, tax_func_path
            )
            taxcalc_version = "Cached tax parameters, no taxcalc version"
        if run_micro:
            micro_data, taxcalc_version = get_micro_data.get_data(
                baseline=p.baseline,
                start_year=p.start_year,
                iit_baseline=iit_baseline,
                iit_reform=iit_reform,
                data=data,
                path=p.output_base,
                client=client,
                num_workers=num_workers,
            )
            p.BW = len(micro_data)
            dict_params = txfunc.tax_func_estimate(  # pragma: no cover
                micro_data,
                p.BW,
                p.S,
                p.starting_age,
                p.ending_age,
                start_year=p.start_year,
                analytical_mtrs=p.analytical_mtrs,
                tax_func_type=p.tax_func_type,
                age_specific=p.age_specific,
                client=client,
                num_workers=num_workers,
                tax_func_path=tax_func_path,
            )
        mean_income_data = dict_params["tfunc_avginc"][0]
        frac_tax_payroll = np.append(
            dict_params["tfunc_frac_tax_payroll"],
            np.ones(p.T + p.S - p.BW)
            * dict_params["tfunc_frac_tax_payroll"][-1],
        )
        # Conduct checks to be sure tax function params are consistent
        # with the model run
        params_list = ["etr", "mtrx", "mtry"]
        BW_in_tax_params = dict_params["BW"]
        start_year_in_tax_params = dict_params["start_year"]
        S_in_tax_params = len(dict_params["tfunc_etr_params_S"][0])
        # Check that start years are consistent in model and cached tax functions
        if p.start_year != start_year_in_tax_params:
            print(
                "Input Error: There is a discrepancy between the start"
                + " year of the model and that of the tax functions!!"
            )
            assert False
        # Check that S is consistent in model and cached tax functions
        # Note: even if p.age_specific = False, the arrays coming from
        # ogcore.txfunc_est should be of length S
        if p.S != S_in_tax_params:
            print(
                "Input Error: There is a discrepancy between the ages"
                + " used in the model and those in the tax functions!!"
            )
            assert False

        # Extrapolate tax function parameters for years after budget window
        # list of list: BW x S - either an array of function at that element...
        etr_params = [[None] * p.S] * p.T
        mtrx_params = [[None] * p.S] * p.T
        mtry_params = [[None] * p.S] * p.T
        for s in range(p.S):
            for t in range(p.T):
                if t < p.BW:
                    etr_params[t][s] = dict_params["tfunc_etr_params_S"][t][s]
                    mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][t][
                        s
                    ]
                    mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][t][
                        s
                    ]
                else:
                    etr_params[t][s] = dict_params["tfunc_etr_params_S"][-1][s]
                    mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][-1][
                        s
                    ]
                    mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][-1][
                        s
                    ]

        if p.constant_rates:
            print("Using constant rates!")
            # Make all tax rates equal the average
            p.tax_func_type = "linear"
            etr_params = [[None] * p.S] * p.T
            mtrx_params = [[None] * p.S] * p.T
            mtry_params = [[None] * p.S] * p.T
            for s in range(p.S):
                for t in range(p.T):
                    if t < p.BW:
                        etr_params[t][s] = dict_params["tfunc_avg_etr"][t]
                        mtrx_params[t][s] = dict_params["tfunc_avg_mtrx"][t]
                        mtry_params[t][s] = dict_params["tfunc_avg_mtry"][t]
                    else:
                        etr_params[t][s] = dict_params["tfunc_avg_etr"][-1]
                        mtrx_params[t][s] = dict_params["tfunc_avg_mtrx"][-1]
                        mtry_params[t][s] = dict_params["tfunc_avg_mtry"][-1]
        if p.zero_taxes:
            print("Zero taxes!")
            etr_params = [[0] * p.S] * p.T
            mtrx_params = [[0] * p.S] * p.T
            mtry_params = [[0] * p.S] * p.T
        tax_param_dict = {
            "etr_params": etr_params,
            "mtrx_params": mtrx_params,
            "mtry_params": mtry_params,
            "taxcalc_version": taxcalc_version,
            "mean_income_data": mean_income_data,
            "frac_tax_payroll": frac_tax_payroll,
        }

        return tax_param_dict

    def read_tax_func_estimate(self, p, tax_func_path):
        """
        This function reads in tax function parameters from pickle
        files.

        Args:
            tax_func_path (str): path to pickle with tax function
                parameter estimates

        Returns:
            dict_params (dict): dictionary containing arrays of tax
                function parameters
            run_micro (bool): whether to estimate tax function parameters

        """
        flag = 0
        if os.path.exists(tax_func_path):
            print("Tax Function Path Exists")
            dict_params = safe_read_pickle(tax_func_path)
            # check to see if tax_functions compatible
            try:
                if p.start_year != dict_params["start_year"]:
                    print(
                        "Model start year not consistent with tax "
                        + "function parameter estimates"
                    )
                    flag = 1
            except KeyError:
                pass
            try:
                p.BW = dict_params["BW"]  # QUICK FIX
                if p.BW != dict_params["BW"]:
                    print(
                        "Model budget window length is "
                        + str(p.BW)
                        + " but the tax function parameter "
                        + "estimates have a budget window length of "
                        + str(dict_params["BW"])
                    )
                    flag = 1
            except KeyError:
                pass
            try:
                if p.tax_func_type != dict_params["tax_func_type"]:
                    print(
                        "Model tax function type is not "
                        + "consistent with tax function parameter "
                        + "estimates"
                    )
                    flag = 1
            except KeyError:
                pass
            if flag >= 1:
                raise RuntimeError(
                    "Tax function parameter estimates at given path"
                    + " are not consistent with model parameters"
                    + " specified."
                )
        else:
            flag = 1
            print(
                "Tax function parameter estimates do not exist at"
                + " given path. Running new estimation."
            )
        if flag >= 1:
            dict_params = None
            run_micro = True
        else:
            run_micro = False

        return dict_params, run_micro

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        dict = {}
        if self.estimate_tax_functions:
            dict.update(self.tax_function_params)
        if self.estimate_beta:
            dict["beta_annual"] = self.beta
        if self.estimate_chi_n:
            dict["chi_n"] = self.chi_n
        dict["eta"] = self.eta
        dict["zeta"] = self.zeta
        dict.update(self.macro_params)
        dict["e"] = self.e
        if self.estimate_pop:
            dict.update(self.demographic_params)

        return dict
