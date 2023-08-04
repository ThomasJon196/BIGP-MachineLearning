import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def rbf(Xa, Xb, l=1.0, sigma_f=1.0):
    
    rbf = sigma_f**2 * np.exp(-0.5/(l**2) * ((Xa[:, np.newaxis] - Xb.T)**2))
    rbf_matrix = rbf.reshape(Xa.size, Xb.size)
    return rbf_matrix



def log_marginal_likelihood(x, y, length_scale, sig_noise):
    """
    Calculates the log marginal likelihood of the prior given the observations.
    """
    
    n_samples = x.shape[0]
    jitter = 1e-4
    K = rbf(x,x,length_scale,sig_noise) + jitter * np.eye(n_samples) # Added jitter
    K_inv = np.linalg.inv(K)
    log_marginal_like = -0.5 * y.T @ K_inv @ y - 0.5 * np.log(np.linalg.det(K)) - 0.5 * n_samples * np.log(2 * np.pi)
    return log_marginal_like


# Optimization of hyperparameters

def optimize_hyperparams(observations):
    
    x = observations.get("x")
    Y = observations.get("Y")
    def opt_goal(param):
        l = param[0]
        sig = param[1]
        log_marg = log_marginal_likelihood(x, Y, l, sig)
        return -log_marg

    h_param_start = np.array([1, 1])

    res = scipy.optimize.minimize(opt_goal, h_param_start, bounds=((0.01, None), (1e-8, None)), method='L-BFGS-B',
                                options={'gtol': 1e-12, 'disp': False})

    l_opt = res.x[0]
    sig_opt = res.x[1]
    # print(f"Optimized Lenghtscale & Variance of rbf-kernel: {res.x}")
    return l_opt, sig_opt



def mean_func(x):
    return np.zeros(x.size)

def calc_kernel_matrix(x_obs, x_pred, l=1., sig=1.):
    """
    Calculates kernel matrix for K, K_s, K_ss
    """
     
    x_pred = np.expand_dims(x_pred, 1)
    x_obs = np.expand_dims(x_obs, 1)
    K = rbf(x_obs, x_obs, l, sig)
    K_s = rbf(x_obs, x_pred, l, sig)
    K_ss = rbf(x_pred, x_pred, l, sig)
    
    return K, K_s, K_ss

def calc_predictive_posterior_distribution(K, K_s, K_ss, y_obs):
    
    K_inv = np.linalg.inv(K + np.eye(K.shape[0]) * 1e-4)
    K_ss_posterior = K_ss - K_s.T @ K_inv @ K_s
    mu_posterior = K_s.T @ K_inv @ y_obs
    
    return K_ss_posterior, mu_posterior
    
def sample_posterior(K_posterior, mu_posterior, n_draws):
    
    multivariate_sample = np.random.multivariate_normal(
            mean=mu_posterior,
            cov=K_posterior, 
            size=n_draws
        )
    
    return multivariate_sample


def gp_regression(observations, X_pred):
    X = observations.get("x")
    Y = observations.get("Y")
    
    l_opt, sig_opt = optimize_hyperparams(observations)

    K, K_s, K_ss = calc_kernel_matrix(X, X_pred, l_opt, sig_opt)

    K_ss_posterior, mu_posterior = calc_predictive_posterior_distribution(K, K_s, K_ss, Y)

    interval = 1.96 * np.sqrt(np.diag(K_ss_posterior))

    return mu_posterior, interval


if __name__ == "__main__":
    X_pred = np.linspace(-15, 15, 5)

    x = [-3, -1, 1, 3]
    y = [-2, 0, -3, 5]

    observations = {}
    observations["x"] = np.asarray(x)
    observations["Y"] = np.asarray(y)


    print(gp_regression(observations, X_pred))