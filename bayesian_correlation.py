import numpy as np
import theano.tensor as T
import pymc3 as pm


# These are internal theano functions used in the model
def _Tcov(sigma, rho):
    """Build a covariance matrix"""
    C = T.alloc(rho, 2, 2)
    C = T.fill_diagonal(C, 1.)
    S = T.diag(sigma)
    return T.nlinalg.matrix_dot(S, C, S)


def _Cov2Cor(Cov):
    """Build a correlation matrix from covariance matrix"""
    S = T.nlinalg.diag(T.nlinalg.diag(Cov))
    S = T.nlinalg.matrix_inverse(T.sqrt(S))
    return T.nlinalg.matrix_dot(S, Cov, S)


def bayesian_linear_model(data,
                          N_steps=20000,
                          step="Metropolis",
                          burnin=None,
                          njobs=1,
                          progressbar=True,
                          chain_start=0,
                          output_format="rho",
                          sample_params={}):

    """Docstring for bayesian linear model.
    :data: The data used, expects a 2-d array with one dimension having 2 columns/rows
    :N_steps: The number of steps in each chain. If using NUTS sampling, this can be smaller.
    :step: The sampling method. Either "Metropolis" (faster sampling, but needs more steps) or NUTS (slower, but fewer steps)
    :burnin: number of steps to discard at the beginning of each chain. If None, half of N_steps is discarded.
    :njobs: The number of parallel jobs.
    :chain_start: The number assigned to the chain. Can be useful when aiming to combine different chains
    :progressbar: Should a progressbar for the sampling?
    :output_format: What should be returned from the sampling? If "rho" only an numpy array with the correlation
    values is returned. If "full" the whole multitrace is returned, which is useful for convergence analysis.
    :sample_params: Additional parameters for pymc3's sample function.
    :returns: Either a multitrace or a numpy array, depending on output_format
    """

    # test the data for the right format and transform it if necessary/possible
    if isinstance(data, list):
        try:
            data = np.vstack(data)
        except ValueError:
            print("Error: Data dimensions do not match!")
            return None
    
    if isinstance(data, np.ndarray):
        if len(data.shape) != 2:
            if len(data) == 2 and len(data[0]) != len(data[1]):
                print("Error: Data dimensions do not match!")
                return None
            else:
                print("Error: Data not a two-dimensional array, don't know what to do!")
                return None
        else:
            if data.shape[1] != 2:
                if data.shape[0] != 2:
                    print("Error: No dimension with 2 variables present, don't know what to do!")
                else:
                    data = data.T

    # if no burnin is specified, use half of the step number
    if burnin is None:
        burnin = N_steps/2

    sample_params.update({"draws": N_steps,
                          "njobs": njobs,
                          "tune": burnin,
                          "progressbar": progressbar})

    # initialize model
    basic_model = pm.Model()

    with basic_model:
        
        # define model priors
        m_mu = pm.Normal('mu', mu=0., sd=10, shape=2)
        nu = pm.Uniform('nu', 0, 5)
        packed_L = pm.LKJCholeskyCov('packed_L', n=2, eta=nu, sd_dist=pm.HalfCauchy.dist(1.))
        chol = pm.expand_packed_triangular(2, packed_L)
        sigma = pm.Deterministic('sigma', _Cov2Cor(chol.dot(chol.T)))
        
        # model likelihood
        mult_n = pm.MvNormal('mult_n', mu=m_mu, chol=chol, observed=data)
        
        # which sampler to use
        if step == "Metropolis":
            step = pm.Metropolis()
        elif step == "NUTS":
            step = pm.NUTS()
        sample_params.update({"step": step})

        # MCMC sample
        trace = pm.sample(**sample_params)

    # Return the full pymc3 trace or only an array of the correlation?
    if output_format == "full":
        output = trace
    else:
        output = trace["sigma"][:, 0, 1]
    return output
