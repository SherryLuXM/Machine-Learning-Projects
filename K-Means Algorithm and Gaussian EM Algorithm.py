# ML414A3Q2
%matplotlib inline
import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Generate 400 data points with dimension 2, for two clusters, with 200 data
# points each, based on mean_1, mean_2, and cov(covariance matrix)
num_samples = 200
cov = [[10, 7], [7, 10]]
mean_1 = [0.1, 0.1]
mean_2 = [6.0, 0.1]
x_class1 = np.random.multivariate_normal(mean_1, cov, size = 200)
x_class2 = np.random.multivariate_normal(mean_2, cov, size = 200)
xy_class1 = np.c_[np.zeros(200), x_class1]
xy_class2 = np.c_[np.ones(200), x_class2]
data_full = np.vstack((xy_class1, xy_class2))
np.random.shuffle(data_full)
data = data_full[:,1:]
labels = data_full[:, 0]

# scatterplot of the data to first assess the data through visualization
plt.plot(x_class1,'cx') # first class, x shape
plt.plot(x_class2,'mo') # second class, circle shape

################################################################################
############################################# implement K-Means Algorithm ######
## The algorithm is broken down into individual functions cost(), km_assignment_step(), km_refitting_step()
def cost(data, R, Mu):
  N, D = data.shape
  K = Mu.shape[1]
  dist = np.zeros(N)
  for i in range(N):
    indx = np.argmax(R[i,])
    dist[i] = scipy.spatial.distance.euclidean(Mu[:,indx], data[i,])
  J = (dist**2).sum()
  return J

def km_assignment_step(data, Mu):
  """ Compute K-Means assignment step
  Args:
  data: a NxD matrix for the data points
  Mu: a DxK matrix for the cluster means locations
  Returns:
  R_new: a NxK matrix of responsibilities
  """
  N, D = data.shape # Number of data points and dimension of datapoint
  K = Mu.shape[1] # number of clusters
  r = np.zeros((N,K))
  R_new = np.zeros((N,K))
  for i in range(N):
    for k in range(K):
      r[i, k] = scipy.spatial.distance.euclidean(Mu[:,k], data[i,:])
    R_new[i,np.argmin(r[i,])] = 1
  return R_new

def km_refitting_step(data, R, Mu):
  """ Compute K-Means refitting step.
  Args:
  data: a NxD matrix for the data points
  R: a NxK matrix of responsibilities
  Mu: a DxK matrix for the cluster means locations
  Returns:
  Mu_new: a DxK matrix for the new cluster means locations
  """
  N, D = data.shape # Number of data points and dimension of datapoint
  K = Mu.shape[1] # number of clusters
  Mu_new = np.zeros((D, K))
  for k in range(K):
    indx = np.nonzero(R[:,k])
    potential_mean = data[indx].mean(axis = 0)
    if np.isnan(potential_mean).any():
      Mu_new[:,k] = Mu[:,k]
    else:
      Mu_new[:,k] = data[indx].mean(axis = 0)
  return Mu_new

## Running the K-Means Algorithm
N, D = 400, 2
K = 2
max_iter = 100
class_init = np.array([0,1])
R = np.zeros([N, K])
for i in range(N):
  R[i, labels[i].astype(np.int)] = 1
np.random.shuffle(R)

Mu = np.zeros([D, K])
Mu[:, 1] = 1.
R.T.dot(data), np.sum(R, axis=0)

for it in range(max_iter):
  R = km_assignment_step(data, Mu)
  Mu = km_refitting_step(data, R, Mu)
  print(it, cost(data, R, Mu))

class_1 = data[np.nonzero(R[:,0])]
class_2 = data[np.nonzero(R[:,1])]

# visualize the result using scatterplot again
plt.plot(class_1,'cx') # first class, x shape
plt.plot(class_2,'mo') # second class, circle shape

################################################################################
######################### Implement EM algorithm for Gaussian mixtures #########
def normal_density(x, mu, Sigma):
  return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
    / np.sqrt(np.linalg.det(2 * np.pi * Sigma))

def log_likelihood(data, Mu, Sigma, Pi):
  """ Compute log likelihood on the data given the Gaussian Mixture Parameters.
  Args:
    data: a NxD matrix for the data points
    Mu: a DxK matrix for the means of the K Gaussian Mixtures
    Sigma: a list of size K with each element being DxD covariance matrix
    Pi: a vector of size K for the mixing coefficients
  Returns:
    L: a scalar denoting the log likelihood of the data given the Gaussian Mixture
  """
  N, D = data.shape # Number of datapoints and dimension of datapoint
  K = Mu.shape[1] # number of mixtures
  L, T = 0., 0.
  # given n, k, compute the likelihood from the k-th Gaussian weighted by the mixing coefficients
  multi_normal = [scipy.stats.multivariate_normal(Mu[:,j], Sigma[j]) for j in range(K)]
  for i in range(N):
    k_sum = 0
    for k in range(K):
      k_sum += Pi[k]*multi_normal[k].pdf(data[i,:])
    L += np.log(k_sum)
  return L

## Gaussian Mixture Expectation
def gm_e_step(data, Mu, Sigma, Pi):
  """ Gaussian Mixture Expectation Step.
  Args:
    data: a NxD matrix for the data points
    Mu: a DxK matrix for the means of the K Gaussian Mixtures
    Sigma: a list of size K with each element being DxD covariance matrix
    Pi: a vector of size K for the mixing coefficients
  Returns:
    Gamma: a NxK matrix of responsibilities
  """
  N, D = data.shape # Number of datapoints and dimension of datapoint
  K = Mu.shape[1] # number of mixtures
  Gamma = np.zeros((N, K)) # zeros of shape (N,K), matrix of responsibilities
  multi_normal = [scipy.stats.multivariate_normal(Mu[:,j], Sigma[j], allow_singular=True) for j in range(K)]
  # given n, k, normalize by sum across second dimension (mixtures)
  for i in range(N):
    joint = np.zeros(K)
    for k in range(K):
      joint[k] = Pi[k]*multi_normal[k].pdf(data[i,:])
    Gamma[i,:] = joint/joint.sum()
  return Gamma

## Gaussian Mixture Maximization
def gm_m_step(data, Gamma):
  """ Gaussian Mixture Maximization Step.
  Args:
    data: a NxD matrix for the data points
    Gamma: a NxK matrix of responsibilities
  Returns:
    Mu: a DxK matrix for the means of the K Gaussian Mixtures
    Sigma: a list of size K with each element being DxD covariance matrix
    Pi: a vector of size K for the mixing coefficients
  """
  N, D = data.shape # Number of datapoints and dimension of datapoint
  K = Gamma.shape[1] # number of mixtures
  Nk = Gamma.sum(axis = 0)
  Mu = np.zeros((D,K))
  Sigma = [np.zeros((D,D))]*K
  # find Mu
  for k in range(K):
    prod_sum = np.zeros(D)
    for d in range(D):
      prod_sum[d] = np.dot(Gamma[:,k],data[:,d])
    Mu[:,k] = prod_sum/Nk[k]
  # find Sigma
  for k in range(K):
    cov_mat = np.zeros((D,D))
    for i in range(N):
      mat = np.subtract(data[i,:], Mu[:,k])
      cov_mat += Gamma[i,k] * np.matmul(np.transpose(mat), mat)
    Sigma[k] = cov_mat/Nk[k]
  Pi = Nk/N
  return Mu, Sigma, Pi

## Running the Gaussian Mixture EM algorithm
N, D = 400, 2
K = 2
Mu = np.zeros([D, K])
Mu[:, 1] = 1.
Sigma = [np.eye(2), np.eye(2)]
Pi = np.ones(K) / K
Gamma = np.zeros([N, K]) # Gamma is the matrix of responsibilities

max_iter = 200

for it in range(max_iter):
  Gamma = gm_e_step(data, Mu, Sigma, Pi)
  Mu, Sigma, Pi = gm_m_step(data, Gamma)

class_type = np.array([np.argmin(Gamma[i,:]) for i in range(N)])
class_1 = data[np.where(class_type == 0)[0],:]
class_2 = data[np.where(class_type == 1)[0],:]

## Make scatterplot
plt.plot(class_1,'cx') # first class, x shape
plt.plot(class_2,'mo') # second class, circle shape