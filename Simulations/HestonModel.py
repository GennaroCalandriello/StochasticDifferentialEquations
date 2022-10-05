import numpy as np
import matplotlib.pyplot as plt
from numba import njit

"""Simulation of Heston model stochastic volatility"""

# parameters to generate 2 sets of correlated Wiener processes (for the price and for the volatility)
rho = 0.6  # correlation parameter
N = 10000
mu = np.array([0.0, 0.0])
SigmaX = 1
SigmaY = 1
Sigma = np.array([[SigmaX ** 2, rho], [rho, SigmaY ** 2]])
kappa = 4
theta = 0.02
v_0 = 0.02
xi = 0.9
r = 0.2
S_0 = 100
paths = 5000
steps = 20000
T = 1
dt = T / steps


def correlatedWiener(steps):

    """Python correlation function"""

    ScalingFactorw1 = 1e-2  # integral solutions change drastically
    ScalingFactorw2 = 1e-2

    W = np.random.multivariate_normal(mu, Sigma, size=steps)

    w1 = W[:, 0] * ScalingFactorw1
    w2 = W[:, 1] * ScalingFactorw2

    covw1w2 = np.mean(w1 * w2) - np.mean(w1) * np.mean(w2)
    corr = covw1w2 / np.std(w1, ddof=1) / np.std(w2, ddof=1)

    return w1, w2, corr


@njit(fastmath=True, cache=True)
def CorrRandVariables(steps):

    """
    Correlation function for 2 Wiener processes
    """

    u1 = np.random.rand(steps, 1)
    u2 = np.random.rand(steps, 1)

    s1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    s2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    mu_x = 0.0
    mu_y = 0.0
    sigma_x = 0.01
    sigma_y = 0.03
    rho = 0.6
    X = mu_x + sigma_x * s1
    Y = mu_y + sigma_y * (rho * s1 + np.sqrt(1 - rho ** 2) * s2)

    E_X = np.mean(X)
    var_X = np.mean(X ** 2) - E_X ** 2
    sigma_X = np.sqrt(var_X)
    E_Y = np.mean(Y)
    var_Y = np.mean(Y ** 2) - E_Y ** 2
    sigma_Y = np.sqrt(var_Y)

    cov_X_Y = np.mean(X * Y) - E_X * E_Y
    corr_X_Y = cov_X_Y / sigma_X / sigma_Y

    return X, Y, corr_X_Y


w1, w2, _ = correlatedWiener(100000)
x, y, _ = CorrRandVariables(10000)
plt.scatter(w1, w2, s=0.3)
plt.scatter(x, y, s=0.4, c="blue")
plt.show()


def HestonPaths(S_0, T, r, steps, kappa, theta, v_0, rho, xi):

    """This function simulates the paths of the Heston model from the analytical solution"""

    dt = T / steps
    prices = np.zeros(steps)
    volatility = np.zeros(steps)

    S_t = S_0
    v_t = v_0

    for i in range(steps):

        Wt = np.random.multivariate_normal(mu, Sigma, size=1) * np.sqrt(dt)
        S_t = S_t * np.exp((r - v_t / 2) * dt + np.sqrt(v_t) * Wt[:, 0])
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * Wt[:, 1])
        prices[i] = S_t
        volatility[i] = v_t

    return prices, volatility


# @njit(fastmath=True, cache=True)
def EulerMaruyamaSchemeHeston(
    S_0, T, r, steps, kappa, theta, v_0, rho, xi, automultivariate=True
):

    """This function integrates the SDE of Heston Model and compare with analytical solution for the same set of random correlated process"""

    dt = T / steps
    S_sim = np.zeros(steps)
    S = S_0
    S_theor = np.zeros(steps)
    vol_sim = np.zeros(steps)  # volatility, initial volatility
    v = v_0

    if automultivariate:
        W1, W2, correlation = correlatedWiener(steps)
    else:
        W1, W2, correlation = CorrRandVariables(steps)  # correlated random variables

    print("The correlation between the 2 processes is: ", correlation)

    for j in range(steps):

        # Wt = np.random.multivariate_normal(mu, Sigma, size=1) * np.sqrt(dt) #this should create correlated rnd processes but it is done by hand in the function above

        S_theor[j] = S * np.exp((r - v / 2) * dt + np.sqrt(v) * W1[j])
        S += r * S * dt + np.sqrt(v) * S * W1[j]  # here is the integration step
        S_sim[j] = S
        v = np.abs(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * W2[j])
        vol_sim[j] = v

    W1, W2 = 0, 0

    return S_sim, S_theor, vol_sim


# pr, vol = HestonPaths(S_0, T, r, steps, kappa, theta, v_0, rho, xi)
# print(len(pr))
if __name__ == "__main__":

    exe = True

    if exe:

        for k in range(6):  # generating tot Heston paths
            S_sim, S_t, volatility = EulerMaruyamaSchemeHeston(
                S_0, T, r, steps, kappa, theta, v_0, rho, xi
            )
            plt.title(
                "Confronto tra soluzione teorica e numerica tramite Euler-Maruyama"
            )
            plt.plot(range(steps), S_sim, label=f"simulated path {k}")
            plt.plot(range(steps), S_t, label=f"Theoretical path {k}")
            plt.legend()

        plt.show()

        # plot volatility
        plt.title(r"$\sigma^2(t)$", fontsize=20)
        plt.plot(range(steps), volatility, label=f"simulated path {k}")
        plt.legend()
        plt.show()

