import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln, logsumexp
from numpy.random import multivariate_normal as npnormal
from numpy.random import normal
from scipy.stats import multivariate_normal
from generate_init_centroids import *

class GaussianMixtureHard():

    def __init__(self, K, init_param="random", seed=None, max_iter=10, eps=None, delta=None, deltai=None, pi0=None, mu0=None, sigma0=None):
        self.K = K
        self.init_param = init_param
        self.pi = pi0
        self.mu = mu0
        self.sigma = sigma0
        self.max_iter = max_iter
        self.eps = eps
        self.delta = delta
        self.deltai = deltai

    def _initialize_parameters_smart(self, X):
        n_samples, D = X.shape
        if self.sigma is None:
            self.sigma = np.array([np.eye(D) for k in range(self.K)])
        if self.mu is None:
            min_max = [[X.min(axis=0)[i], X.max(axis=0)[i]] for i in range(D)]
            self.mu, _ = batch_smartly_generate_init_centroids_binary_search(self.K, min_max)
            self.mu = np.array(self.mu)
            self.mu0 = self.mu
        if self.pi is None:
            self.pi = np.array([1/self.K for k in range(self.K)])

    def _initialize_parameters_random(self, X):
        n_samples, D = X.shape

        if self.sigma is None:
            self.sigma = np.array([np.eye(D) for k in range(self.K)])
        if self.mu is None:
            min_max = [[X.min(axis=0)[i], X.max(axis=0)[i]] for i in range(D)]
            self.mu = randomly_generate_init_centroids(self.K, min_max, D)
            self.mu = np.array(self.mu)
            self.mu0 = self.mu
        if self.pi is None:
            self.pi = np.array([1/self.K for k in range(self.K)])

    def fit_predict(self, X):

        _, D = X.shape

        assert self.init_param in ['random', 'smart']
        if self.init_param == 'random':
            self._initialize_parameters_random(X)
        else:
            self._initialize_parameters_smart(X)

        self.lll = np.empty(self.max_iter)
        for i in range(self.max_iter):
            resp = self._e_step(X)  
            self._m_step(X, resp)
            self.lll[i] = self._compute_log_likelihood(X)

        resp = self._e_step(X)
        return resp.argmax(axis=1)

    def _compute_statististics(self, X, resp):
        n_samples, D = X.shape
        N = resp.sum(axis=0)
        if self.eps:
            rate_sum = 1 + 3*D + 2*D**2
            epsi = np.sqrt(4*np.log(1.25/self.deltai)/self.max_iter/rate_sum*(self.eps+2*np.log(1/self.delta)-np.sqrt((self.eps+2*np.log(1/self.delta))**2-self.eps**2)))
        if self.eps:
            N_sigma = 2*np.log(1.25/self.deltai)/epsi**2
            N += normal(0, np.sqrt(N_sigma))
        N = np.maximum(1, N)
        pi = N/N.sum()
        x_bar = (resp.T @ X)  / N[:, np.newaxis]
        if self.eps:
            for k in range(self.K):
                xbar_eps = epsi*2*np.sqrt(D)
                xbar_sigma = 2*np.log(1.25/self.deltai)*4*D/xbar_eps**2
                x_bar[k] += npnormal(np.zeros(D), xbar_sigma*np.eye(D))/N[k]
                x_bar[k] = np.maximum(-1,np.minimum(1, x_bar[k]))
        S = np.zeros((self.K, D, D))
        for k in range(self.K):
            Xc = X - x_bar[k]
            S[k] = ((resp[:,k] * Xc.T) @ Xc) / N[k]   
            if self.eps:
                S_eps = epsi*np.sqrt(2*D**2-D)
                S_sigma = 2*np.log(1.25/self.deltai)*(2*D**2-D)/S_eps**2
                triu = np.triu(npnormal(np.zeros(D*D), S_sigma*np.eye(D*D)).reshape(D,D))/N[k]
                diag = triu+triu.T-np.diag(np.diag(triu))
                S[k] += diag 
            eigval, eigvec = np.linalg.eig(S[k])
            eigval = np.maximum(0.01, eigval)
            S[k] = (eigval*eigvec) @ eigvec.T
        return pi, x_bar, S
    
    def _e_step(self, X):
        n_samples, D = X.shape    
        E = np.zeros((n_samples, self.K))
        for k in range(self.K):
            E[:,k] = multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k])
        E *= self.pi.reshape(1,-1)
        resp = E/(np.sum(E, axis=1).reshape(-1,1))
        resp_max = np.zeros_like(resp)
        resp_max[np.arange(len(resp)), resp.argmax(1)] = 1
        return resp_max

    def _m_step(self, X, resp):
        n, D = X.shape
        self.pi, self.mu, self.sigma = self._compute_statististics(X, resp)   

    def _compute_log_likelihood(self, X):
        n_samples, D = X.shape
        E = np.zeros((n_samples, self.K))
        for k in range(self.K):
            E[:,k] = multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k])
        E *= self.pi.reshape(1,-1)
        ln_p_x = np.sum(np.log(np.sum(E, axis=1)))
        return ln_p_x

