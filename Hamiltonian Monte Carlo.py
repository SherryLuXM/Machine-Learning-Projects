# LM414A2
from tqdm import trange, tqdm_notebook # Progress meters

def leapfrog(params_t0, momentum_t0, stepsize, logprob_grad_fun):
    # Performs a reversible update of parameters and momentum
    # See https://en.wikipedia.org/wiki/Leapfrog_integration
    momentum_thalf = momentum_t0 + 0.5 * stepsize * logprob_grad_fun(params_t0)
    params_t1 = params_t0 + stepsize * momentum_thalf
    momentum_t1 = momentum_thalf + 0.5 * stepsize * logprob_grad_fun(params_t1)
    return params_t1, momentum_t1

def iterate_leapfrogs(theta, v, stepsize, num_leapfrog_steps, grad_fun):
    for i in range(0, num_leapfrog_steps):
        theta, v = leapfrog(theta, v, stepsize, grad_fun)
    return theta, v

def metropolis_hastings(state1, state2, log_posterior):
    # Compares the log_posterior at two values of parameters,
    # and accepts the new values proportional to the ratio of the posterior
    # probabilities.
    accept_prob = torch.exp(log_posterior(state2) - log_posterior(state1))
    if random.random() < accept_prob:
        return state2 # Accept
    else:
        return state1 # Reject

def draw_samples(num_params, stepsize, num_leapfrog_steps, n_samples, log_posterior):
    theta = torch.zeros(num_params)

def log_joint_density_over_params_and_momentum(state):
    params, momentum = state
    return diag_gaussian_log_density(momentum, torch.zeros_like(momentum), torch.ones_like(momentum)) \
        + log_posterior(params)

def grad_fun(zs):
    zs = zs.detach().clone()
    zs.requires_grad_(True)
    y = log_posterior(zs)
    y.backward()
    return zs.grad

sampleslist = []
for i in trange(0, n_samples):
    sampleslist.append(theta)
    momentum = torch.normal(0, 1, size = np.shape(theta))
    theta_new, momentum_new = iterate_leapfrogs(theta, momentum, stepsize, num_leapfrog_steps, grad_fun)
    theta, momentum = metropolis_hastings((theta, momentum), (theta_new, momentum_new), log_joint_density_over_params_and_momentum)
return torch.stack((sampleslist))

# Hyperparameters
num_players = 2
num_leapfrog_steps = 20
n_samples = 2000
stepsize = 0.01

def log_posterior_a(zs):
    return log_posterior_A_beat_B(zs[0], zs[1])

samples_a = draw_samples(num_players, stepsize, num_leapfrog_steps, n_samples, log_posterior_a)
plot_2d_fun(posterior_A_beat_B, "Player A Skill", "Player B Skill", samples_a)

# Hyperparameters
num_players = 2
num_leapfrog_steps = 20
n_samples = 2000
stepsize = 0.01
key = 42

def log_posterior_b(zs):
    return log_posterior_A_beat_B_10_times(zs[0], zs[1])

samples_b = draw_samples(num_players, stepsize, num_leapfrog_steps, n_samples, log_posterior_b)
ax = plot_2d_fun(posterior_A_beat_B_10_times, "Player A Skill", "Player B Skill", samples_b)

# Hyperparameters
num_players = 2
num_leapfrog_steps = 20
n_samples = 2000
stepsize = 0.01

def log_posterior(zs):
    return log_posterior_beat_each_other_10_times(zs[0], zs[1])

samples_c = draw_samples(num_players, stepsize, num_leapfrog_steps, n_samples, log_posterior)
ax = plot_2d_fun(posterior_A_beat_B_10_times, "Player A Skill", "Player B Skill", samples_c)