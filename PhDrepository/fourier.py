import numpy as np
import matplotlib.pyplot as plt


class FourierVolatilityEstimator:
    def __init__(self, p, sigma2, T=2 * np.pi, N=1000, mu_func=None, sigma_func=None):
        self.T = T
        self.N = N
        self.dt = T / N
        self.t = np.linspace(0, T, N + 1)
        self.p = p
        self.sigma2 = sigma2
        self.mu_func = mu_func if mu_func is not None else lambda t: 0.1
        self.sigma_func = sigma_func if sigma_func is not None else lambda t: 0.1
        self.a0_p = None
        self.a0_sigma2 = None
        self.a_p = None
        self.b_p = None
        self.a_sigma2 = None
        self.b_sigma2 = None

    def compute_fourier_coefficients(self):
        print("Computing Fourier coefficients...")
        N = self.N
        t = self.t
        p = self.p
        sigma = self.sigma_func

        dt = np.diff(t)
        d_p = np.diff(p)

        self.a0_p = (1 / (2 * np.pi)) * np.sum(d_p)
        self.a0_sigma2 = (1 / (2 * np.pi)) * np.sum(sigma(t[:-1]) ** 2 * dt)

        a_p = []
        b_p = []
        a_sigma2 = []
        b_sigma2 = []

        for k in range(1, N + 1):
            if k % 10 == 0:
                print(f"Computing Fourier coefficients for k={k}")
            a_p.append((1 / np.pi) * np.sum(np.cos(k * t[:-1]) * d_p))
            b_p.append((1 / np.pi) * np.sum(np.sin(k * t[:-1]) * d_p))
            a_sigma2.append(
                (1 / np.pi) * np.sum(np.cos(k * t[:-1]) * sigma(t[:-1]) ** 2 * dt)
            )
            b_sigma2.append(
                (1 / np.pi) * np.sum(np.sin(k * t[:-1]) * sigma(t[:-1]) ** 2 * dt)
            )

        self.a_p = np.array(a_p)
        self.b_p = np.array(b_p)
        self.a_sigma2 = np.array(a_sigma2)
        self.b_sigma2 = np.array(b_sigma2)

    def reconstruct_sigma2(self, M=None):
        if M is None:
            M = self.N // 2

        t = self.t
        a0 = self.a0_sigma2
        a_k = self.a_sigma2
        b_k = self.b_sigma2

        sigma2 = np.full_like(t, a0)
        for k in range(1, M + 1):
            if k % 10 == 0:
                print(f"Reconstructing sigma2 for k={k}")
            sigma2 += (1 - k / M) * (
                a_k[k - 1] * np.cos(k * t) + b_k[k - 1] * np.sin(k * t)
            )
        return sigma2

    def plot_results(self, sigma2_reconstructed):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.p, label="$p(t)$")
        plt.title("Simulated Log-Price $p(t)$")
        plt.xlabel("Time")
        plt.ylabel("$p(t)$")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.sigma2, label="True $\sigma^2(t)$", linestyle="--")
        plt.plot(self.t, sigma2_reconstructed, label="Reconstructed $\sigma^2(t)$")
        plt.title("Reconstructed $\sigma^2(t)$ using Fourier-Fejer formula")
        plt.xlabel("Time")
        plt.ylabel("$\sigma^2(t)$")
        plt.legend()

        plt.tight_layout()
        plt.show()


# # Example Usage
# def main():
#     # Parameters
#     theta = 0.035  # parameter estimated by Andersen and Bollerslev (1998a)
#     omega = 0.636  # parameter estimated by Andersen and Bollerslev (1998a)
#     lambda_ = 0.296  # parameter estimated by Andersen and Bollerslev (1998a)
#     T = 1.0  # time horizon (1 day)
#     N = 86400  # number of time steps (1 second intervals)
#     p0 = 0.0  # initial value of log-price p(t)
#     sigma0 = 0.2  # initial value of volatility sigma(t)
#     mean_duration = 14  # mean duration between quotes in seconds

#     # Custom drift function
#     def custom_mu(t):
#         return 0.05  # Example drift function

#     # Create an instance of the simulator
#     simulator = EulerMaruyamaSimulator(
#         T, N, theta, omega, lambda_, p0, sigma0, mu_func=custom_mu
#     )
#     t, p, sigma2 = simulator.simulate_paths()

#     # Create an instance of the FourierVolatilityEstimator
#     estimator = FourierVolatilityEstimator(p, sigma2, T, N)
#     estimator.compute_fourier_coefficients()
#     sigma2_reconstructed = estimator.reconstruct_sigma2()
#     estimator.plot_results(sigma2_reconstructed)


# if __name__ == "__main__":
#     main()
