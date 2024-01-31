"""
This script uses simulated method of moments estimator to estimate the
beta_j parameters for OG-USA.
"""

import numpy as np
import os
import scipy.optimize as opt
from ogcore.parameters import Specifications
from ogusa import wealth
from ogcore import SS
from ogcore.utils import Inequality


# Don't print output along the way of SS solution
SS.VERBOSE = False

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]


def beta_estimate(
    beta_initial_guesses, og_spec={}, two_step=False, client=None
):
    """
    This function estimates the beta_j parameters using a simulated
    method of moments estimator that targets moments of the wealth
    distribution.

    Args:
        beta_initial_guesses (array-like): array of initial guesses for the
            beta_j parameters
        og_spec (dict): any updates to default model parameters
        two_step (boolean): whether to use a two-step estimator
        client (Dask Client object): dask client for multiprocessing

    Returns:
        beta_hat (array-like): estimates of the beta_j
        beta_se (array-like): standard errors on the beta_j estimates

    """

    # initialize parameters object
    tax_func_path = os.path.join(
        CUR_PATH,
        "..",
        "data",
        "tax_functions",
        "TxFuncEst_baseline_PUF.pkl",
    )
    p = Specifications(baseline=True)
    p.update_specifications(og_spec)
    p.get_tax_function_parameters(client, False, tax_func_path)

    # Compute wealth moments from the data
    scf = wealth.get_wealth_data(scf_yrs_list=[2019], web=True, directory=None)
    data_moments = wealth.compute_wealth_moments(scf, p.lambdas)

    # Get weighting matrix
    W = compute_weighting_matrix(p, optimal_weight=False)

    # call minimizer
    # set bounds on beta estimates (need to be between 0 and 1)
    bnds = np.tile(np.array([1e-12, 1]), (p.J, 1))  # Need (1e-12, 1) J times
    # pack arguments in a tuple
    min_args = (data_moments, W, p, client)
    # NOTE: may want to try some global optimization routing like
    # simulated annealing (aka basin hopping) or differential
    # evolution
    est_output = opt.minimize(
        minstat,
        beta_initial_guesses,
        args=(min_args),
        method="L-BFGS-B",
        bounds=bnds,
        tol=1e-15,
        options={"maxfun": 1, "maxiter": 1, "maxls": 1},
    )
    beta_hat = est_output["x"]

    # calculate std errors
    K = len(data_moments)
    beta_se, VCV_params = compute_se(beta_hat, W, K, p, h=0.01, client=client)

    if two_step:
        W = VCV_params
        min_args = (data_moments, W, p, client)
        est_output = opt.minimize(
            minstat,
            beta_initial_guesses,
            args=(min_args),
            method="L-BFGS-B",
            bounds=bnds,
            tol=1e-15,
            options={"maxfun": 1, "maxiter": 1, "maxls": 1},
        )
        beta_hat = est_output["x"]
        beta_se, VCV_params = compute_se(
            beta_hat, W, K, p, h=0.01, client=client
        )

    return beta_hat, beta_se


def minstat(beta_guesses, *args):
    """
    This function generates the weighted sum of squared differences
    between the model and data moments.

    Args:
        beta_guesses (array-like): a vector of length J with the betas
        args (tuple): length 6 tuple, variables needed for minimizer

    Returns:
        distance (scalar): weighted, squared deviation between data and
            model moments

    """
    # unpack args tuple
    data_moments, W, p, client = args

    # Update beta in parameters object with beta guesses
    p.beta = beta_guesses

    # Solve model SS
    print("Baseline = ", p.baseline)
    ss_output = SS.run_SS(p, client=client)

    # Compute moments from model SS
    model_moments = calc_moments(ss_output, p)

    # distance with levels
    distance = np.dot(
        np.dot((np.array(model_moments) - np.array(data_moments)).T, W),
        np.array(model_moments) - np.array(data_moments),
    )

    print("DATA and MODEL DISTANCE: ", distance)

    return distance


def calc_moments(ss_output, p):
    """
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    Args:
        ss_output = dictionary, variables from SS of model
        p (OG-USA Specifications object): model parameters

    Returns:
        model_moments (array-like): Array of model moments

    """
    # Create Inequality object
    wealth_ineq = Inequality(
        ss_output["bssmat_splus1"], p.omega_SS, p.lambdas, p.S, p.J
    )

    # wealth moments
    # moments are: bottom 25% of wealth, next 25% share of wealth
    #  (25-50 pctile), next 20% share of wealth (50-70 pctile),
    #  next 10% share (70-80 pctile), next 10% share (80-90 pctile),
    #  next 9% share (90-99 pctile), top 1% share,
    # gini coefficient, variance of log wealth
    model_moments = np.array(
        [
            1 - wealth_ineq.top_share(0.75),
            wealth_ineq.top_share(0.75) - wealth_ineq.top_share(0.50),
            wealth_ineq.top_share(0.50) - wealth_ineq.top_share(0.30),
            wealth_ineq.top_share(0.30) - wealth_ineq.top_share(0.20),
            wealth_ineq.top_share(0.20) - wealth_ineq.top_share(0.10),
            wealth_ineq.top_share(0.10) - wealth_ineq.top_share(0.01),
            wealth_ineq.top_share(0.01),
            wealth_ineq.gini(),
            wealth_ineq.var_of_logs(),
        ]
    )

    return model_moments


def compute_weighting_matrix(p, optimal_weight=False):
    """
    Function to compute the weighting matrix for the GMM estimator.

    Args:
        p (OG-USA Specifications object): model parameters
        optimal_weight (boolean): whether to use an optimal
            weighting matrix or not

    Returns:
        W (Numpy array): Weighting matrix

    """
    # determine weighting matrix
    if optimal_weight:
        # This uses the inverse of the VCV matrix for the data moments
        # more precisely estimated moments get more weight
        # Reference: Gourieroux, Monfort, and Renault (1993,
        # Journal of Applied Econometrics)
        # read in SCF
        n = 1000  # number of bootstrap iterations
        scf = wealth.get_wealth_data(
            scf_yrs_list=[2019], web=True, directory=None
        )
        VCV_data_moments = wealth.VCV_moments(scf, n, p.lambdas, p.J)
        W = np.linalg.inv(VCV_data_moments)
    else:
        # Assumes use 2 more moments than there are parameters
        W = np.identity(p.J + 2)

    return W


def VCV_moments(scf, n, bin_weights, J):
    """
    Compute variance-covariance matrix for wealth moments by
    bootstrapping data.

    Args:
        scf (Pandas DataFrame): raw data from SCF
        n (int): number of bootstrap iterations to run
        bin_weights (array-like): ability weights (Jx1 array)
        J (int): number of ability groups

    Returns:
        VCV (Numpy array): variance-covariance matrix of wealth moments,
            (J+2xJ+2) array

    """
    wealth_moments_boot = np.zeros((n, J + 2))
    for i in range(n):
        sample = scf[np.random.randint(2, size=len(scf.index)).astype(bool)]
        # note that wealth moments from data are in array in same order
        # as model moments are computed in this module
        wealth_moments_boot[i, :] = wealth.compute_wealth_moments(
            sample, bin_weights
        )

    VCV = np.cov(wealth_moments_boot.T)

    return VCV


def compute_se(beta_hat, W, K, p, h=0.01, client=None):
    """
    Function to compute standard errors for the SMM estimator.

    Args:
        beta_hat (array-like): estimates of beta parameters
        W (Numpy array): weighting matrix
        K (int): number of moments
        p (OG-USA Specifications object): model parameters
        h (scalar): percentage to move parameters for numerical derivatives
        client (Dask Client object): Dask client

    Returns:
        beta_se (array-like): standard errors for beta estimates
        VCV_params (Numpy array): VCV matrix for parameter estimates

    """
    # compute numerical derivatives that will need for SE's
    model_moments_low = np.zeros((p.J, K))
    model_moments_high = np.zeros((p.J, K))
    beta_low = beta_hat
    beta_high = beta_hat
    for i in range(len(beta_hat)):
        # compute moments with downward change in param
        beta_low[i] = beta_hat[i] * (1 + h)
        p.beta = beta_low
        ss_output = ss_output = SS.run_SS(p, client=client)
        model_moments_low[i, :] = calc_moments(ss_output, p)
        # compute moments with upward change in param
        beta_high[i] = beta_hat[i] * (1 - h)
        p.beta = beta_low
        ss_output = ss_output = SS.run_SS(p, client=client)
        model_moments_high[i, :] = calc_moments(ss_output, p)

    deriv_moments = (model_moments_high - model_moments_low).T / (
        2 * h * beta_hat
    )
    VCV_params = np.linalg.inv(
        np.dot(np.dot(deriv_moments.T, W), deriv_moments)
    )
    beta_se = (np.diag(VCV_params)) ** (1 / 2)

    return beta_se, VCV_params
