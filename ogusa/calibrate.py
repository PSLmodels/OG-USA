from ogusa import estimate_beta_j, bequest_transmission, demographics
from ogusa import macro_params, transfer_distribution, income
from ogusa import get_micro_data, psid_data_setup
import os
import numpy as np
from ogcore import txfunc
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
        tax_func_path=None,
        iit_reform={},
        guid="",
        data="cps",
        client=None,
        num_workers=1,
    ):

        self.estimate_tax_functions = estimate_tax_functions
        self.estimate_beta = estimate_beta
        self.estimate_chi_n = estimate_chi_n
        if estimate_tax_functions:
            self.tax_function_params = self.get_tax_function_parameters(
                p,
                iit_reform,
                guid,
                data,
                client,
                num_workers,
                run_micro=True,
                tax_func_path=tax_func_path,
            )
        if estimate_beta:
            self.beta_j = estimate_beta_j.beta_estimate(self)
        # if estimate_chi_n:
        #     chi_n = self.get_chi_n()

        # Macro estimation
        self.macro_params = macro_params.get_macro_params()

        # eta estimation
        self.eta = transfer_distribution.get_transfer_matrix()

        # zeta estimation
        self.zeta = bequest_transmission.get_bequest_matrix()

        # demographics
        self.demographic_params = demographics.get_pop_objs(
            p.E, p.S, p.T, 1, 100, p.start_year
        )
        # demographics for 80 period lives (needed for getting e below)
        demog80 = demographics.get_pop_objs(20, 80, p.T, 1, 100, p.start_year)

        # earnings profiles
        self.e = income.get_e_interp(
            p.S,
            self.demographic_params["omega_SS"],
            demog80["omega_SS"],
            p.lambdas,
            plot=False,
        )

    # Tax Functions
    def get_tax_function_parameters(
        self,
        p,
        iit_reform={},
        guid="",
        data="",
        client=None,
        num_workers=1,
        run_micro=False,
        tax_func_path=None,
    ):
        """
        Reads pickle file of tax function parameters or estimates the
        parameters from microsimulation model output.

        Args:
            client (Dask client object): client
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
        if run_micro:
            micro_data, _ = get_micro_data.get_data(
                baseline=p.baseline,
                start_year=p.start_year,
                reform=iit_reform,
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
                baseline=p.baseline,
                analytical_mtrs=p.analytical_mtrs,
                tax_func_type=p.tax_func_type,
                age_specific=p.age_specific,
                reform=iit_reform,
                data=data,
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

        # Reorder indices of tax function and tile for all years after
        # budget window ends
        num_etr_params = dict_params["tfunc_etr_params_S"].shape[2]
        num_mtrx_params = dict_params["tfunc_mtrx_params_S"].shape[2]
        num_mtry_params = dict_params["tfunc_mtry_params_S"].shape[2]
        # First check to see if tax parameters that are used were
        # estimated with a budget window and ages that are as long as
        # the those implied based on the start year and model age.
        # N.B. the tax parameters dictionary does not save the years
        # that correspond to the parameter estimates, so the start year
        # used there may name match what is used in a run that reads in
        # some cached tax function parameters.  Likewise for age.
        params_list = ["etr", "mtrx", "mtry"]
        BW_in_tax_params = dict_params["tfunc_etr_params_S"].shape[1]
        S_in_tax_params = dict_params["tfunc_etr_params_S"].shape[0]
        if p.BW != BW_in_tax_params:
            print(
                "Warning: There is a discrepency between the start"
                + " year of the model and that of the tax functions!!"
            )
        # After printing warning, make it work by tiling
        if p.BW > BW_in_tax_params:
            for item in params_list:
                dict_params["tfunc_" + item + "_params_S"] = np.concatenate(
                    (
                        dict_params["tfunc_" + item + "_params_S"],
                        np.tile(
                            dict_params["tfunc_" + item + "_params_S"][
                                :, -1, :
                            ].reshape(S_in_tax_params, 1, num_etr_params),
                            (1, p.BW - BW_in_tax_params, 1),
                        ),
                    ),
                    axis=1,
                )
                dict_params["tfunc_avg_" + item] = np.append(
                    dict_params["tfunc_avg_" + item],
                    np.tile(
                        dict_params["tfunc_avg_" + item][-1],
                        (p.BW - BW_in_tax_params),
                    ),
                )
        if p.S != S_in_tax_params:
            print(
                "Warning: There is a discrepency between the ages"
                + " used in the model and those in the tax functions!!"
            )
        # After printing warning, make it work by tiling
        if p.S > S_in_tax_params:
            for item in params_list:
                dict_params["tfunc_" + item + "_params_S"] = np.concatenate(
                    (
                        dict_params["tfunc_" + item + "_params_S"],
                        np.tile(
                            dict_params["tfunc_" + item + "_params_S"][
                                -1, :, :
                            ].reshape(1, p.BW, num_etr_params),
                            (p.S - S_in_tax_params, 1, 1),
                        ),
                    ),
                    axis=0,
                )
        etr_params = np.empty((p.T, p.S, num_etr_params))
        mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
        mtry_params = np.empty((p.T, p.S, num_mtry_params))
        etr_params[: p.BW, :, :] = np.transpose(
            dict_params["tfunc_etr_params_S"][: p.S, : p.BW, :], axes=[1, 0, 2]
        )
        etr_params[p.BW :, :, :] = np.tile(
            np.transpose(
                dict_params["tfunc_etr_params_S"][: p.S, -1, :].reshape(
                    p.S, 1, num_etr_params
                ),
                axes=[1, 0, 2],
            ),
            (p.T - p.BW, 1, 1),
        )
        mtrx_params[: p.BW, :, :] = np.transpose(
            dict_params["tfunc_mtrx_params_S"][: p.S, : p.BW, :],
            axes=[1, 0, 2],
        )
        mtrx_params[p.BW :, :, :] = np.transpose(
            dict_params["tfunc_mtrx_params_S"][: p.S, -1, :].reshape(
                p.S, 1, num_mtrx_params
            ),
            axes=[1, 0, 2],
        )
        mtry_params[: p.BW, :, :] = np.transpose(
            dict_params["tfunc_mtry_params_S"][: p.S, : p.BW, :],
            axes=[1, 0, 2],
        )
        mtry_params[p.BW :, :, :] = np.transpose(
            dict_params["tfunc_mtry_params_S"][: p.S, -1, :].reshape(
                p.S, 1, num_mtry_params
            ),
            axes=[1, 0, 2],
        )

        if p.constant_rates:
            print("Using constant rates!")
            # Make all ETRs equal the average
            etr_params = np.zeros(etr_params.shape)
            # set shift to average rate
            etr_params[: p.BW, :, 10] = np.tile(
                dict_params["tfunc_avg_etr"].reshape(p.BW, 1), (1, p.S)
            )
            etr_params[p.BW :, :, 10] = dict_params["tfunc_avg_etr"][-1]

            # # Make all MTRx equal the average
            mtrx_params = np.zeros(mtrx_params.shape)
            # set shift to average rate
            mtrx_params[: p.BW, :, 10] = np.tile(
                dict_params["tfunc_avg_mtrx"].reshape(p.BW, 1), (1, p.S)
            )
            mtrx_params[p.BW :, :, 10] = dict_params["tfunc_avg_mtrx"][-1]

            # # Make all MTRy equal the average
            mtry_params = np.zeros(mtry_params.shape)
            # set shift to average rate
            mtry_params[: p.BW, :, 10] = np.tile(
                dict_params["tfunc_avg_mtry"].reshape(p.BW, 1), (1, p.S)
            )
            mtry_params[p.BW :, :, 10] = dict_params["tfunc_avg_mtry"][-1]
        if p.zero_taxes:
            print("Zero taxes!")
            etr_params = np.zeros(etr_params.shape)
            mtrx_params = np.zeros(mtrx_params.shape)
            mtry_params = np.zeros(mtry_params.shape)
        tax_param_dict = {
            "etr_params": etr_params,
            "mtrx_params": mtrx_params,
            "mtry_params": mtry_params,
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
        dict.update(self.demographic_params)

        return dict
