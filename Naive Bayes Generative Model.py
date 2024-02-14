#ML414A1
import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve
from math import log

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)

def fashion_mnist():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')
    # Remove the data point that cause log(0)
    remove = (14926, 20348, 36487, 45128, 50945, 51163, 55023)
    train_images = np.delete(train_images,remove, axis=0)
    train_labels = np.delete(train_labels, remove, axis=0)
    return train_images, train_labels, test_images[:1000], test_labels[:1000]

def load_fashion_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = fashion_mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels

def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images (N_samples x N_features), train_labels (N_samples x N_classes)
    Returns the MAP estimator theta_est (N_features x N_classes) and the MLE
    estimator pi_est (N_classes)"""
    N_samples, N_features = train_images.shape
    N_classes = train_labels.shape[1]
    N_1 = np.zeros((N_features, N_classes))
    for i in range(N_samples):
        for j in range(N_features):
            N_1[j,] = N_1[j,] + train_labels[i,]
    Total = train_labels.sum(axis=0)
    theta_est = (N_1 + 1)/(Total +2)
    pi_est = Total/N_samples
    return(theta_est, pi_est)

def log_likelihood(images, theta, pi):
    """ Inputs: images (N_samples x N_features), theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
        log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
        log_like is a matrix of (N_samples x N_classes)
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    N_samples, N_features = images.shape
    N_classes = theta.shape[1]
    log_like = np.zeros((N_samples, N_classes))
    for i in range(N_samples):
        for k in range(N_classes):
            product = np.zeros(N_classes)
            for j in range(N_features):
                product = product + log(pow(theta[j,k], images[i, j])*pow((1-theta[j,k]), (1- images[i, j])))
            log_like[i,k] = log(pi[k]) + product[k]
    sums = np.sum(log_like, axis = 0)
    log_like = log_like - sums
    return(log_like)

def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels (N_samples x N_classes)
    Returns the accuracy based on predictions from log likelihood values"""

    N_samples = log_like.shape[0]
    prediction = log_like*labels
    accurate = 0
    for j in range(N_samples):
        accurate += int(max(log_like[j,]) == max(prediction[j,]))
    return(accurate/N_samples)

N_data, train_images, train_labels, test_images, test_labels = load_fashion_mnist()
theta_est, pi_est = train_map_estimator(train_images, train_labels)

loglike_train = log_likelihood(train_images, theta_est, pi_est)
avg_loglike = np.sum(loglike_train * train_labels) / N_data
train_accuracy = accuracy(loglike_train, train_labels)
loglike_test = log_likelihood(test_images, theta_est, pi_est)
test_accuracy = accuracy(loglike_test, test_labels)

print(f"Average log-likelihood for MAP is {avg_loglike:.3f}")
print(f"Training accuracy for MAP is {train_accuracy:.3f}")
print(f"Test accuracy for MAP is {test_accuracy:.3f}")

# move beyond prediction to sample and plot 10 images using MAP estimates
def image_sampler(theta, pi, num_images):
    """ Inputs: parameters theta and pi, and number of images to sample
    Returns the sampled images (N_images x N_features)"""

    N_classes = len(pi)
    N_features = theta.shape[0]
    sample_images = np.zeros((num_images, N_features))
    cum_prob = np.cumsum(pi)
    dataset1 = np.random.rand(num_images)
    cond = [dataset1 <= limit for limit in cum_prob]
    class_matrix = [i*np.ones(num_images) for i in range(N_classes)]
    c_values = np.select(condlist = cond, choicelist = class_matrix) # getting c from categorical distribution
    for i in range(num_images):
        for j in range(N_features):
            c_ind = int(c_values[i])
            p = theta[j,c_ind]
            sample_images[i, j] = int(np.random.rand()<=p) # Bernoulli distribution
    return(sample_images)

def plot_images(images, ims_per_row=5, padding=5, image_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=0., vmax=1.):
    """Images should be a (N_images x pixels) matrix."""
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = vmin
    concat_images = np.full(((image_dimensions[0] + padding) * N_rows + padding,
                             (image_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], image_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + image_dimensions[0]) * row_ix
        col_start = padding + (padding + image_dimensions[1]) * col_ix
        concat_images[row_start: row_start + image_dimensions[0],
                        col_start: col_start + image_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    plt.plot()

sampled_images = image_sampler(theta_est, pi_est, 10)
plot_images(sampled_images)

# Use the generative model to fill in missing pixels(about 70% of the pixels are missing)
def probabilistic_imputer(theta, pi, original_images, is_observed):
    """Inputs: parameters theta and pi, original_images (N_images x N_features),
        and is_observed which has the same shape as original_images, with a value
        1. in every observed entry and 0. in every unobserved entry.
    Returns the new images where unobserved pixels are replaced by their
    conditional probability"""

    N_images, N_features = original_images.shape
    N_classes = theta.shape[1]
    log_like = np.zeros((N_images, N_classes))
    cl = np.zeros(N_images)
    for i in range(N_images):
        for k in range(N_classes):
            product = np.zeros(N_classes)
            for j in range(N_features):
                if original_images[i,j] == 0:
                    product = product + log(pow(theta[j,k], original_images[i, j])*pow((1-theta[j,k]), (1- original_images[i, j])))
            log_like[i,k] = log(pi[k]) + product[k]
        cl[i] = np.argmax(log_like[i,])
    for i in range(N_images):
        for j in range(N_features):
            if is_observed[i,j] == 1:
                ind = int(cl[i])
                original_images[i,j] = theta[j,ind]
    return(original_images)

num_features = train_images.shape[1]
is_observed = np.random.binomial(1, p=0.3, size=(20, num_features))
plot_images(train_images[:20] * is_observed)

imputed_images = probabilistic_imputer(theta_est, pi_est, train_images[:20], is_observed)
plot_images(imputed_images)