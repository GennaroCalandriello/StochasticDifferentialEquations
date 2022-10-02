import numpy as np
import matplotlib.pyplot as plt

"""Integration of Black & Scholes price options SDE via Euler-Maruyama integration scheme. For the theoretical part you can read this: https://math.gmu.edu/~tsauer/pre/sde.pdf"""

mu, sigma, x0 = 0.75, 0.30, 100

T = 1
N = 2 ** 10
dt = 1.0 / N
t = np.arange(0, 1, dt)

np.random.seed(1)
dW = np.sqrt(dt) * np.random.randn(
    N
)  # genero un array random z*sqrt(dt) per ottenere numeri estratti da N(0, sqrt(dt))
W = np.cumsum(dW)
Wt = np.cumsum(np.sqrt(dt) * np.random.randn(N))
x_theorethical = x0 * np.exp(
    (mu - 0.5 * sigma ** 2) * t + (sigma * Wt)
)  # array per soluzione analitica B&S


def EulerMaruyamaScheme():

    x_simulation, x = [], x0

    for j in range(N):
        x += mu * x * dt + sigma * x * dW[j]
        x_simulation.append(x)

    return x_simulation


x_sim = EulerMaruyamaScheme()
plt.title("Black & Scholes", fontsize=20)
plt.plot(t, x_sim, label="Simulation")
plt.plot(t, x_theorethical, label="Theorethical")
plt.legend()
plt.show()

