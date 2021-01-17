from ogusa-calibrate import (estimate_beta_j, bequest_transmission,
                             demographics, deterministic_profiles,
                             macro_params, transfer_distribution)

class ogusa_calibrate():
    ''' OG-USA calibration class
    '''
    def __init__(self,
                 estimate_tax_functions=False, estimate_beta=False,
                 estimate_chi_n=False, baseline_dir=BASELINE_DIR,
                 iit_reform={}, guid='', data='cps',
                 client=None, num_workers=1):

        if estimate_tax_functions:
            get_tax_function_parameters(self, client, run_micro=False,
                                    tax_func_path=None)
        if estimate_beta:
            beta_j = self.get_beta_j(self)
        if estimate_chi_n:
            chi_n = self.get_chi_n()

        self.macro_params = self.get_macro_params()
        self.eta = self.get_eta()
        self.zeta = self.get_zeta()

    # Tax Functions
    def get_tax_function_parameters(self, client, run_micro=False,
                                    tax_func_path=None):
        '''
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
        '''
        # set paths if none given
        if tax_func_path is None:
            if self.baseline:
                pckl = "TxFuncEst_baseline{}.pkl".format(self.guid)
                tax_func_path = os.path.join(self.output_base, pckl)
                print('Using baseline tax parameters from ',
                        tax_func_path)
            else:
                pckl = "TxFuncEst_policy{}.pkl".format(self.guid)
                tax_func_path = os.path.join(self.output_base, pckl)
                print('Using reform policy tax parameters from ',
                        tax_func_path)
        # If run_micro is false, check to see if parameters file exists
        # and if it is consistent with Specifications instance
        if not run_micro:
            dict_params, run_micro = self.read_tax_func_estimate(
                tax_func_path)
        if run_micro:
            txfunc.get_tax_func_estimate(  # pragma: no cover
                self.BW, self.S, self.starting_age, self.ending_age,
                self.baseline, self.analytical_mtrs, self.tax_func_type,
                self.age_specific, self.start_year, self.iit_reform,
                self.guid, tax_func_path, self.data, client,
                self.num_workers)
            dict_params, _ = self.read_tax_func_estimate(tax_func_path)
        self.mean_income_data = dict_params['tfunc_avginc'][0]
        try:
            self.frac_tax_payroll = np.append(
                dict_params['tfunc_frac_tax_payroll'],
                np.ones(self.T + self.S - self.BW) *
                dict_params['tfunc_frac_tax_payroll'][-1])
        except KeyError:
            pass
        try:
            self.taxcalc_version = dict_params['taxcalc_version']
        except KeyError:
            self.taxcalc_version = 'No version recorded'

        # Reorder indices of tax function and tile for all years after
        # budget window ends
        num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
        num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
        num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
        # First check to see if tax parameters that are used were
        # estimated with a budget window and ages that are as long as
        # the those implied based on the start year and model age.
        # N.B. the tax parameters dictionary does not save the years
        # that correspond to the parameter estimates, so the start year
        # used there may name match what is used in a run that reads in
        # some cached tax function parameters.  Likewise for age.
        params_list = ['etr', 'mtrx', 'mtry']
        BW_in_tax_params = dict_params['tfunc_etr_params_S'].shape[1]
        S_in_tax_params = dict_params['tfunc_etr_params_S'].shape[0]
        if self.BW != BW_in_tax_params:
            print('Warning: There is a discrepency between the start' +
                    ' year of the model and that of the tax functions!!')
        # After printing warning, make it work by tiling
        if self.BW > BW_in_tax_params:
            for item in params_list:
                dict_params['tfunc_' + item + '_params_S'] =\
                    np.concatenate(
                        (dict_params['tfunc_' + item + '_params_S'],
                            np.tile(dict_params['tfunc_' + item +
                                                '_params_S'][:, -1, :].
                                    reshape(S_in_tax_params, 1, num_etr_params),
                                    (1, self.BW - BW_in_tax_params, 1))),
                        axis=1)
                dict_params['tfunc_avg_' + item] =\
                    np.append(dict_params['tfunc_avg_' + item],
                                np.tile(dict_params['tfunc_avg_' + item][-1],
                                        (self.BW - BW_in_tax_params)))
        if self.S != S_in_tax_params:
            print('Warning: There is a discrepency between the ages' +
                    ' used in the model and those in the tax functions!!')
        # After printing warning, make it work by tiling
        if self.S > S_in_tax_params:
            for item in params_list:
                dict_params['tfunc_' + item + '_params_S'] =\
                    np.concatenate(
                        (dict_params['tfunc_' + item + '_params_S'],
                            np.tile(dict_params['tfunc_' + item +
                                                '_params_S'][-1, :, :].
                                    reshape(1, self.BW, num_etr_params),
                                    (self.S - S_in_tax_params, 1, 1))),
                        axis=0)
        self.etr_params = np.empty((self.T, self.S, num_etr_params))
        self.mtrx_params = np.empty((self.T, self.S, num_mtrx_params))
        self.mtry_params = np.empty((self.T, self.S, num_mtry_params))
        self.etr_params[:self.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_etr_params_S'][:self.S, :self.BW, :],
                axes=[1, 0, 2])
        self.etr_params[self.BW:, :, :] =\
            np.tile(np.transpose(
                dict_params['tfunc_etr_params_S'][:self.S, -1, :].reshape(
                    self.S, 1, num_etr_params), axes=[1, 0, 2]),
                    (self.T - self.BW, 1, 1))
        self.mtrx_params[:self.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_mtrx_params_S'][:self.S, :self.BW, :],
                axes=[1, 0, 2])
        self.mtrx_params[self.BW:, :, :] =\
            np.transpose(
                dict_params['tfunc_mtrx_params_S'][:self.S, -1, :].reshape(
                    self.S, 1, num_mtrx_params), axes=[1, 0, 2])
        self.mtry_params[:self.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_mtry_params_S'][:self.S, :self.BW, :],
                axes=[1, 0, 2])
        self.mtry_params[self.BW:, :, :] =\
            np.transpose(
                dict_params['tfunc_mtry_params_S'][:self.S, -1, :].reshape(
                    self.S, 1, num_mtry_params), axes=[1, 0, 2])

        if self.constant_rates:
            print('Using constant rates!')
            # # Make all ETRs equal the average
            self.etr_params = np.zeros(self.etr_params.shape)
            # set shift to average rate
            self.etr_params[:self.BW, :, 10] = np.tile(
                dict_params['tfunc_avg_etr'].reshape(self.BW, 1),
                (1, self.S))
            self.etr_params[self.BW:, :, 10] =\
                dict_params['tfunc_avg_etr'][-1]

            # # Make all MTRx equal the average
            self.mtrx_params = np.zeros(self.mtrx_params.shape)
            # set shift to average rate
            self.mtrx_params[:self.BW, :, 10] = np.tile(
                dict_params['tfunc_avg_mtrx'].reshape(self.BW, 1),
                (1, self.S))
            self.mtrx_params[self.BW:, :, 10] =\
                dict_params['tfunc_avg_mtrx'][-1]

            # # Make all MTRy equal the average
            self.mtry_params = np.zeros(self.mtry_params.shape)
            # set shift to average rate
            self.mtry_params[:self.BW, :, 10] = np.tile(
                dict_params['tfunc_avg_mtry'].reshape(self.BW, 1),
                (1, self.S))
            self.mtry_params[self.BW:, :, 10] =\
                dict_params['tfunc_avg_mtry'][-1]
        if self.zero_taxes:
            print('Zero taxes!')
            self.etr_params = np.zeros(self.etr_params.shape)
            self.mtrx_params = np.zeros(self.mtrx_params.shape)
            self.mtry_params = np.zeros(self.mtry_params.shape)

    def read_tax_func_estimate(self, tax_func_path):
        '''
        This function reads in tax function parameters from pickle
        files.
        Args:
            tax_func_path (str): path to pickle with tax function
                parameter estimates
        Returns:
            dict_params (dict): dictionary containing arrays of tax
                function parameters
            run_micro (bool): whether to estimate tax function parameters
        '''
        flag = 0
        if os.path.exists(tax_func_path):
            print('Tax Function Path Exists')
            dict_params = safe_read_pickle(tax_func_path)
            # check to see if tax_functions compatible
            current_taxcalc =\
                pkg_resources.get_distribution("taxcalc").version
            try:
                if current_taxcalc != dict_params['tax_calc_version']:
                    print('WARNING: Tax function parameters estimated' +
                            ' from Tax Calculator version that is not ' +
                            ' the one currently installed on this machine.')
                    print('Current TC version is ', current_taxcalc,
                            ', Estimated tax functions from version ',
                            dict_params.get('tax_calc_version', None))
                    flag = 1
            except KeyError:
                pass
            try:
                if self.start_year != dict_params['start_year']:
                    print('Model start year not consistent with tax ' +
                            'function parameter estimates')
                    flag = 1
            except KeyError:
                pass
            try:
                if self.BW != dict_params['BW']:
                    print('Model budget window length is not ' +
                            'consistent with tax function parameter ' +
                            'estimates')
                    flag = 1
            except KeyError:
                pass
            try:
                if self.tax_func_type != dict_params['tax_func_type']:
                    print('Model tax function type is not ' +
                            'consistent with tax function parameter ' +
                            'estimates')
                    flag = 1
            except KeyError:
                pass
            if flag >= 1:
                raise RuntimeError(
                    'Tax function parameter estimates at given path' +
                    ' are not consistent with model parameters' +
                    ' specified.')
        else:
            flag = 1
            print('Tax function parameter estimates do not exist at' +
                    ' given path. Running new estimation.')
        if flag >= 1:
            dict_params = None
            run_micro = True
        else:
            run_micro = False

        return dict_params, run_micro

    # Macro estimation
    self.macro_params = macro_params.get_macro_params()

    # eta estimation
    self.eta = transfer_distribution.get_transfer_matrix()

    # zeta estimation
    self.zeta = bequest_transmission.get_bequest_matrix()

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        dict = {}
        if estimate_tax_functions:
            dict['tax_func_params'] = self.taxfunctions
        if estimate_beta:
            dict['beta_annual'] = self.beta
        if estimate_chi_n:
            dict['chi_n'] = self.chi_n
        dict['eta'] = self.eta
        dict['zeta'] = self.zeta
        dict.update(self.macro_params)


        return dict
