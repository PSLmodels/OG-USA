from ogusa import estimate_beta_j, bequest_transmission
from ogusa import macro_params, transfer_distribution, income
from ogusa import get_micro_data
from ogusa import calibrate_lifecycle
import copy
import json
import os
import warnings
import numpy as np
from taxcalc import Records
from ogcore import txfunc, demographics
from ogcore.utils import safe_read_pickle, mkdirs


class Calibration:
    """OG-USA calibration class"""

    def __init__(
        self,
        p,
        estimate_tax_functions=False,
        estimate_beta=False,
        estimate_chi_n=False,
        estimate_lifecycle_prefs=False,
        estimate_pop=False,
        get_macro_params=False,
        tax_func_path=None,
        lifecycle_params_path=None,
        lifecycle_config=None,
        lifecycle_options=None,
        lifecycle_initial_ss=None,
        lifecycle_kwargs=None,
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
            estimate_beta (bool): whether to estimate beta with the
                legacy wealth-moment SMM (`estimate_beta_j`)
            estimate_chi_n (bool): deprecated alias for
                `estimate_lifecycle_prefs`; `chi_n` is only calibrated
                jointly with `beta_annual` and `chi_b`
            estimate_lifecycle_prefs (bool): whether to calibrate
                `beta_annual` by type, `chi_b`, and the `chi_n` age
                profile with the nested general-equilibrium routine in
                `calibrate_lifecycle.calibrate_lifecycle_preferences`
                (20 to 40 minutes serially; see
                `LIFECYCLE_CALIBRATION_PLAN.md`)
            estimate_pop (bool): whether to estimate population
            get_macro_params (bool): whether to get macro parameters
            tax_func_path (str): path to tax function parameters
            lifecycle_params_path (str): JSON file with calibrated
                lifecycle preference parameters. If it exists and is
                consistent with `p`, the parameters are read instead of
                re-calibrated; otherwise the calibration runs and the
                result is written there. When None the calibration
                always runs and is saved to `p.output_base`.
            lifecycle_config (LifecycleCalibrationConfig): moment
                configuration for the lifecycle calibration
            lifecycle_options (PreferenceCalibrationOptions): options
                for the beta / chi_b step
            lifecycle_initial_ss (dict): OG-Core steady-state output to
                start the outer loop from, instead of a cold solve
            lifecycle_kwargs (dict): further keyword arguments for
                `calibrate_lifecycle_preferences` (for example
                `max_outer`, `param_tol`, `ge_client`)
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
            demographic_data_path (str): path to save or find
                downloaded UN demographic data
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
        if estimate_chi_n and not estimate_lifecycle_prefs:
            warnings.warn(
                "estimate_chi_n is deprecated; chi_n is calibrated jointly "
                "with beta_annual and chi_b. Use estimate_lifecycle_prefs.",
                DeprecationWarning,
                stacklevel=2,
            )
            estimate_lifecycle_prefs = True
        self.estimate_chi_n = estimate_chi_n
        self.estimate_lifecycle_prefs = estimate_lifecycle_prefs
        self.estimate_pop = estimate_pop
        self.get_macro_params = get_macro_params
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
            self.beta_j, self.beta_j_se = estimate_beta_j.beta_estimate(
                np.asarray(p.beta_annual, dtype=float), client=client
            )

        # Macro estimation
        if self.get_macro_params:
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
                income_percentiles=p.lambdas.flatten(),
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
                income_percentiles=p.lambdas.flatten(),
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

        # Lifecycle preference parameters: beta by type, chi_b, chi_n
        self.lifecycle_outcome = None
        if self.estimate_lifecycle_prefs:
            self.lifecycle_params = self.get_lifecycle_parameters(
                p,
                lifecycle_params_path=lifecycle_params_path,
                config=lifecycle_config,
                options=lifecycle_options,
                initial_ss=lifecycle_initial_ss,
                client=client,
                **(lifecycle_kwargs or {}),
            )

    # Lifecycle preference parameters
    def get_lifecycle_parameters(
        self,
        p,
        lifecycle_params_path=None,
        config=None,
        options=None,
        initial_ss=None,
        client=None,
        **kwargs,
    ):
        """
        Reads calibrated lifecycle preference parameters from a JSON file
        or calibrates them with the nested general-equilibrium routine.

        The calibration runs on a copy of ``p`` that already carries every
        other parameter this class has produced so far (tax functions,
        transfer and bequest matrices, earnings profiles, demographics,
        macro parameters), so the calibrated preferences are consistent
        with the rest of ``get_dict()``.  ``p`` itself is not modified.

        Args:
            p (OG-Core Specifications object): parameters object
            lifecycle_params_path (str): JSON file to read from or write
                to; None writes to ``p.output_base``
            config (LifecycleCalibrationConfig): moment configuration
            options (PreferenceCalibrationOptions): beta / chi_b options
            initial_ss (dict): steady state to start the outer loop from
            client (Dask client object): client for household solves
            kwargs: passed to ``calibrate_lifecycle_preferences``

        Returns:
            dict: ``beta_annual``, ``chi_b``, ``chi_n`` as lists

        """
        run_calibration = True
        if lifecycle_params_path is None:
            lifecycle_params_path = os.path.join(
                p.output_base, "LifecyclePrefEst.json"
            )
        else:
            params, run_calibration = self.read_lifecycle_parameters(
                p, lifecycle_params_path
            )
        mkdirs(os.path.split(lifecycle_params_path)[0])
        if run_calibration:
            p_calib = copy.deepcopy(p)
            updates = {
                k: v
                for k, v in self._parameter_updates().items()
                if k in p_calib._data
            }
            if updates:
                p_calib.update_specifications(updates)
            outcome = calibrate_lifecycle.calibrate_lifecycle_preferences(
                p_calib,
                config=config,
                options=options,
                initial_ss=initial_ss,
                client=client,
                **kwargs,
            )
            self.lifecycle_outcome = outcome
            params = outcome.parameter_dict
            record = dict(params)
            record["_meta"] = {
                "S": int(p.S),
                "J": int(p.J),
                "start_year": int(p.start_year),
                "converged": bool(outcome.converged),
                "iterations": int(outcome.iterations),
            }
            with open(lifecycle_params_path, "w", encoding="utf-8") as file:
                json.dump(record, file, indent=1)
            print(
                "Saved lifecycle preference parameters to ",
                lifecycle_params_path,
            )
        return params

    def read_lifecycle_parameters(self, p, lifecycle_params_path):
        """
        Reads calibrated lifecycle preference parameters from a JSON
        file and checks that they fit the model dimensions.

        Args:
            p (OG-Core Specifications object): parameters object
            lifecycle_params_path (str): path to the JSON file

        Returns:
            params (dict or None): ``beta_annual``, ``chi_b``, ``chi_n``
            run_calibration (bool): whether the calibration must run

        """
        keys = ("beta_annual", "chi_b", "chi_n")
        if not os.path.exists(lifecycle_params_path):
            print(
                "Lifecycle preference parameters do not exist at given "
                "path. Running new calibration."
            )
            return None, True
        with open(lifecycle_params_path, "r", encoding="utf-8") as file:
            record = json.load(file)
        if not all(k in record for k in keys):
            raise RuntimeError(
                "Lifecycle preference parameter file at given path is "
                "missing one of " + ", ".join(keys)
            )
        params = {k: list(record[k]) for k in keys}
        consistent = (
            len(params["beta_annual"]) == p.J
            and len(params["chi_b"]) == p.J
            and len(params["chi_n"]) == p.S
        )
        if not consistent:
            print(
                "Lifecycle preference parameters at given path do not "
                "match the model's S and J. Running new calibration."
            )
            return None, True
        print(
            "Using lifecycle preference parameters from ",
            lifecycle_params_path,
        )
        return params, False

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
                gfactors=gfactors,
                weights=weights,
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

    def _parameter_updates(self):
        """
        Collects the calibrated parameters other than the lifecycle
        preferences (tax functions, legacy beta, eta, zeta, macro
        parameters, e, demographics).

        Returns:
            dict (dict): parameter updates in `update_specifications`
                format

        """
        dict = {}
        if self.estimate_tax_functions:
            dict.update(self.tax_function_params)
        if self.estimate_beta:
            dict["beta_annual"] = self.beta_j
        dict["eta"] = self.eta
        dict["zeta"] = self.zeta
        if self.get_macro_params:
            dict.update(self.macro_params)
        dict["e"] = self.e
        if self.estimate_pop:
            dict.update(self.demographic_params)

        return dict

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        """
        Returns all newly calibrated parameters in a dictionary.

        Returns:
            dict (dict): parameter updates in `update_specifications`
                format, including `beta_annual`, `chi_b`, and `chi_n`
                when the lifecycle preferences were calibrated

        """
        dict = self._parameter_updates()
        if self.estimate_lifecycle_prefs:
            dict.update(self.lifecycle_params)

        return dict
