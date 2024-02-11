# LM414A3Q1
!pip install wget
import os
import os.path
import matplotlib.pyplot as plt
import wget
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.io
import scipy.stats
import torch
import random
from torch import nn
from torch.distributions.normal import Normal
from functools import partial
from tqdm import trange, tqdm_notebook
import matplotlib.pyplot as plt

# Helper function
def diag_gaussian_log_density(x, mu, std):
    # axis=-1 means sum over the last dimension.
    m = Normal(mu, std)
    return torch.sum(m.log_prob(x), axis=-1)

def log_joint_prior(zs_array):
    return diag_gaussian_log_density(zs_array, torch.tensor([0.0]), torch.tensor([1.0]))

def logp_a_beats_b(z_a, z_b):
    return -torch.logaddexp(torch.tensor([0.0]), z_b - z_a)

def log_prior_over_2_players(z1, z2):
    m = Normal(torch.tensor([0.0]), torch.tensor([[1.0]]))
    return m.log_prob(z1) + m.log_prob(z2)

def prior_over_2_players(z1, z2):
    return torch.exp(log_prior_over_2_players(z1, z2))

def log_posterior_A_beat_B(z1, z2):
    return log_prior_over_2_players(z1, z2) + logp_a_beats_b(z1, z2)

def posterior_A_beat_B(z1, z2):
    return torch.exp(log_posterior_A_beat_B(z1, z2))

def log_posterior_A_beat_B_10_times(z1, z2):
    return log_prior_over_2_players(z1, z2) + 10.0 * logp_a_beats_b(z1, z2)

def posterior_A_beat_B_10_times(z1, z2):
    return torch.exp(log_posterior_A_beat_B_10_times(z1, z2))

def log_posterior_beat_each_other_10_times(z1, z2):
    return log_prior_over_2_players(z1, z2) \
        + 10.* logp_a_beats_b(z1, z2) \
        + 10.* logp_a_beats_b(z2, z1)

def posterior_beat_each_other_10_times(z1, z2):
    return torch.exp(log_posterior_beat_each_other_10_times(z1, z2))

def diag_gaussian_samples(mean, log_std, num_samples):
    # mean and log_std are (D) dimensional vectors
    # Return a (num_samples, D) matrix, where each sample is
    # from a diagonal multivariate Gaussian.
    # you must use the reparameterization trick. Also remember that
    # we are parameterizing the _log_ of the standard deviation.
    D = len(mean)
    torch.manual_seed(0)
    epsilon = torch.randn(num_samples, D)
    return epsilon * torch.exp(log_std) + mean

def diag_gaussian_logpdf(x, mean, log_std):
    # Evaluate the density of a batch of points on a
    # diagonal multivariate Gaussian. x is a (num_samples, D) matrix.
    # Return a tensor of shape (num_samples)
    return diag_gaussian_log_density(x, mean, torch.exp(log_std))

def batch_elbo(logprob, mean, log_std, num_samples):
    # Use simple Monte Carlo to estimate ELBO on a batch of size num_samples
    samples = diag_gaussian_samples(mean, log_std, num_samples)
    log_q = diag_gaussian_logpdf(samples, mean, log_std)
    logp_prob = torch.tensor(logprob(samples))
    return (logp_prob - log_q).mean()

# Hyperparameters
num_players = 2
n_iters = 800
stepsize = 0.0001
num_samples_per_iter = 50
## Approximating the joint probability of Player A wins 10 games using Evidence Lower Bound
def log_posterior_A_beat_B_10_times_1_arg(z1z2):
    return log_posterior_A_beat_B_10_times(z1z2[:,0], z1z2[:,1]).flatten()

def objective(params): # The loss function to be minimized.
    return -batch_elbo(log_posterior_A_beat_B_10_times_1_arg,params[0].clone().detach().requires_grad_(True),params[1].clone().detach().require

def callback(params, t):
    if t % 25 == 0:
        print("Iteration {} lower bound {}".format(t, objective(params)))

# Set up optimizer.
D = 2
init_log_std = torch.tensor(np.zeros(D), requires_grad=True)# TODO.
init_mean = torch.tensor(np.zeros(D), requires_grad=True)# TODO
params = (init_mean, init_log_std)
optimizer = torch.optim.SGD(params, lr=stepsize, momentum=0.9)
def update():
    optimizer.zero_grad()
    loss = objective(params)
    loss.backward()
    optimizer.step()

# Main loop.
print("Optimizing variational parameters...")
for t in trange(0, n_iters):
    update()
    callback(params, t)

def approx_posterior_2d(z1, z2):
    # The approximate posterior
    mean, logstd = params[0].detach(), params[1].detach()
    return torch.exp(diag_gaussian_logpdf(torch.stack([z1, z2], dim=2), mean, logstd))

plot_2d_fun(posterior_A_beat_B_10_times, "Player A Skill", "Player B Skill", f2=approx_posterior_2d)

################################################################################
## Implement Simple Monte Carlo to return a negative elbo estimate to approximate
## the joint distribution where players A and B each win 10 games
# Hyperparameters
n_iters = 100
stepsize = 0.0001
num_samples_per_iter = 50

def log_posterior_beat_each_other_10_times_1_arg(z1z2):
    # z1z2 is a tensor with shape (num_samples x 2)
    # Return a tensor with shape (num_samples)
    return log_posterior_beat_each_other_10_times(z1z2[:,0], z1z2[:,1]).flatten()

def objective(params):
    return -batch_elbo(log_posterior_beat_each_other_10_times_1_arg,
                   params[0].clone().detach().requires_grad_(True),
                   params[1].clone().detach().requires_grad_(True))


# Main loop.
init_mean = torch.tensor(np.zeros(D), requires_grad=True)
init_log_std = torch.tensor(np.zeros(D), requires_grad=True)
params = (init_mean, init_log_std)
optimizer torch optim SGD(params lr stepsize momentum 0 9)