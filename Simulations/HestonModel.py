import numpy as np
import matplotlib.pyplot as plt
from numba import njit

"""Simulation of Heston model stochastic volatility"""

# parameters to generate 2 sets of correlated Wiener processes (for the price and for the volatility)
rho = -0.7
N = 10000
mu = np.array([0, 0])
Sigma = np.array([[1, rho], [rho, 1]])


def correlatedWiener():
    W = np.random.multivariate_normal(mu, Sigma, size=N)
    return W


# @njit(fastmath=True, cache=True)
def HestonPaths(S_0, T, r, kappa, theta, v_0, rho, xi, steps):

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


def EulerMaruyamaSchemeHeston(S_0, T, r, steps, kappa, theta, v_0, rho, xi):

    """This function integrates the SDE of Heston Model and compare with analytical solution for the same set of random correlated process"""

    dt = T / steps
    S_sim, S = np.zeros(steps), S_0
    S_theor = np.zeros(steps)
    vol_sim, v = np.zeros(steps), v_0  # volatility, initial volatility

    for j in range(steps):

        Wt = np.random.multivariate_normal(mu, Sigma, size=1) * np.sqrt(dt)

        S_theor[j] = S * np.exp((r - v / 2) * dt + np.sqrt(v) * Wt[:, 0])
        S += r * S * dt + np.sqrt(v) * S * Wt[:, 0]
        S_sim[j] = S
        v = np.abs(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * Wt[:, 1])
        vol_sim[j] = v

    return S_sim, S_theor, vol_sim


kappa = 4
theta = 0.02
v_0 = 0.02
xi = 0.9
r = 0.02
S_0 = 100
paths = 5000
steps = 2000
T = 1

# pr, vol = HestonPaths(S_0, T, r, kappa, theta, v_0, rho, xi, steps)
# print(len(pr))

for k in range(3):  # generating 6 Heston paths
    S_sim, S_t, volatility = EulerMaruyamaSchemeHeston(
        S_0, T, r, steps, kappa, theta, v_0, rho, xi
    )
    plt.title("Confronto tra soluzione teorica e numerica tramite Euler-Maruyama")
    plt.plot(range(steps), S_sim, label=f"simulated path {k}")
    plt.plot(range(steps), S_t, label=f"Theoretical path {k}")
    plt.legend()
plt.show()
