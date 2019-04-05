# bayesian_correlation
A small script to estimate Bayesian correlations (bivariate normal model), using pymc3.

Usage: The main command is called 
bayesian_linear_model(data, N_steps=20000, step="Metropolis", burnin=None, njobs=1, progressbar=True, chain_start=0,            output_format="rho", sample_params={}):

data: a 2D array or list with dim=2 at one axis

N_steps: the number of MCMC steps

step: MCMC step method

burnin: the steps discarded as burnin, if None, half of N_steps is added to the chain and discarded after sampling

njobs: number of parallel jobs

progressbar: show a progressbar for sampling

chain_start: pymc3 number of chains, useful for combining chains

output_format: the return format, if "rho" return only an numpy array of the sampled correlation values, if "full" return the full 
multitrace of the bivariate normal model

sample_params: other parameters that are given to the pymc3 sampling function
