import numpy as np
import matplotlib.pyplot as plt

# Model parameters
theta1, omega1, lambda1 = 0.035, 0.636, 0.296
theta2, omega2, lambda2 = 0.054, 0.476, 0.480
rho_values = [0.35, -0.35]

T = 1.0  # 1 day in time
N = 86400  # number of time steps (1 second intervals)
dt = T / N


# Initialize arrays
def simulate_paths(N, theta1, omega1, lambda1, theta2, omega2, lambda2, rho):
    t = np.linspace(0, T, N + 1)
    p1, p2 = np.zeros(N + 1), np.zeros(N + 1)
    sigma1_sq, sigma2_sq = np.zeros(N + 1), np.zeros(N + 1)
    sigma1_sq[0], sigma2_sq[0] = 0.2**2, 0.2**2

    # Generate correlated Brownian motions
    cov_matrix = np.array([[1, rho, 0, 0], [rho, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    L = np.linalg.cholesky(cov_matrix)
    dW = np.random.normal(size=(N, 4)) @ L.T * np.sqrt(dt)

    for i in range(1, N + 1):
        sigma1_sq[i] = (
            sigma1_sq[i - 1]
            + lambda1 * (omega1 - sigma1_sq[i - 1]) * dt
            + np.sqrt(2 * lambda1 * theta1 * sigma1_sq[i - 1]) * dW[i - 1, 2]
        )
        sigma2_sq[i] = (
            sigma2_sq[i - 1]
            + lambda2 * (omega2 - sigma2_sq[i - 1]) * dt
            + np.sqrt(2 * lambda2 * theta2 * sigma2_sq[i - 1]) * dW[i - 1, 3]
        )
        p1[i] = p1[i - 1] + np.sqrt(sigma1_sq[i - 1]) * dW[i - 1, 0]
        p2[i] = p2[i - 1] + np.sqrt(sigma2_sq[i - 1]) * dW[i - 1, 1]

    return t, p1, p2, sigma1_sq, sigma2_sq


def unevenly_sampled_times(T, mean_duration):
    times = [0]
    while times[-1] < T:
        times.append(times[-1] + np.random.exponential(mean_duration))
    return np.array(times)


def compute_realized_volatility(t, p1, p2, sampling_times):
    dp1 = np.diff(p1[np.searchsorted(t, sampling_times)])
    dp2 = np.diff(p2[np.searchsorted(t, sampling_times)])
    realized_vol = np.sqrt(np.sum(dp1**2) / len(dp1)), np.sqrt(
        np.sum(dp2**2) / len(dp2)
    )
    realized_cov = np.sum(dp1 * dp2) / len(dp1)
    return realized_vol, realized_cov


def compute_fourier_coefficients(p1, p2, dt):
    N = len(p1)
    dp1 = np.diff(p1)
    dp2 = np.diff(p2)

    freqs = np.fft.fftfreq(N - 1, d=dt)
    dft1 = np.fft.fft(dp1 / np.sqrt(dt))
    dft2 = np.fft.fft(dp2 / np.sqrt(dt))

    a0_1 = (1 / (N - 1)) * np.sum(dp1**2 / dt)
    a0_2 = (1 / (N - 1)) * np.sum(dp2**2 / dt)
    cov = (1 / (N - 1)) * np.sum(dp1 * dp2 / dt)

    return (
        2 * np.sum((dft1.real**2 + dft1.imag**2) / (N - 1)),
        2 * np.sum((dft2.real**2 + dft2.imag**2) / (N - 1)),
        cov,
    )


def main():
    mean_duration = 60  # mean duration between quotes in seconds

    for rho in rho_values:
        t, p1, p2, sigma1_sq, sigma2_sq = simulate_paths(
            N, theta1, omega1, lambda1, theta2, omega2, lambda2, rho
        )

        # Unevenly sampled times
        sampling_times = unevenly_sampled_times(T, mean_duration)

        # Compute realized volatility
        realized_vol, realized_cov = compute_realized_volatility(
            t, p1, p2, sampling_times
        )

        # Compute Fourier coefficients
        fourier_vol1, fourier_vol2, fourier_cov = compute_fourier_coefficients(
            p1, p2, dt
        )

        print(f"Results for rho = {rho}:")
        print(f"Realized Volatility: {realized_vol}")
        print(f"Realized Covariance: {realized_cov}")
        print(f"Fourier Volatility for p1: {fourier_vol1}")
        print(f"Fourier Volatility for p2: {fourier_vol2}")
        print(f"Fourier Covariance: {fourier_cov}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, p1, label="Asset 1 Price")
        plt.plot(t, p2, label="Asset 2 Price")
        plt.title(f"Simulated Prices for rho = {rho}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, sigma1_sq, label="Asset 1 Variance")
        plt.plot(t, sigma2_sq, label="Asset 2 Variance")
        plt.title(f"Simulated Variances for rho = {rho}")
        plt.xlabel("Time")
        plt.ylabel("Variance")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for multivariate GARCH process
# theta = np.array(
#     [0.035, 0.035]
# )  # parameter estimated by Andersen and Bollerslev (1998a)
# omega = np.array(
#     [0.636, 0.636]
# )  # parameter estimated by Andersen and Bollerslev (1998a)
# lambda_ = np.array(
#     [0.296, 0.296]
# )  # parameter estimated by Andersen and Bollerslev (1998a)
# T = 1.0  # time horizon (1 day)
# N = 3000  # number of time steps
# p0 = np.array([0.0, 0.0])  # initial value of log-prices p(t)
# sigma0 = np.array([0.2, 0.2])  # initial value of volatilities sigma(t)
# mean_duration = 14  # mean duration between quotes in seconds
# rho = 0.5  # correlation between the Brownian motions


# def simulate_multivariate_paths(T, N, theta, omega, lambda_, p0, sigma0, rho):
#     """This simulates a multivariate path for a GARCH model"""
#     dt = T / N
#     t = np.linspace(0, T, N)
#     p = np.zeros((N, 2))
#     sigma2 = np.zeros((N, 2))
#     p[0] = p0
#     sigma2[0] = sigma0**2
#     dW = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N) * np.sqrt(dt)
#     for i in range(1, N + 1):
#         for j in range(2):
#             sigma2[i, j] = (
#                 sigma2[i - 1, j]
#                 + theta[j] * (omega[j] - sigma2[i - 1, j]) * dt
#                 + np.sqrt(2 * lambda_[j] * theta[j]) * sigma2[i - 1, j] * dW[i - 1, j]
#             )
#             p[i, j] = p[i - 1, j] + np.sqrt(sigma2[i - 1, j]) * dW[i - 1, j]
#     return t, p, sigma2


# def unevenly_sampled_times(T, mean_duration):
#     times = [0]
#     while times[-1] < T:
#         times.append(times[-1] + np.random.exponential(mean_duration))
#     return np.array(times)


# def compute_fourier_coefficients(t, p, sigma):
#     print("Computing Fourier coefficients...")

#     dt = np.diff(t)
#     d_p = np.diff(p, axis=0)

#     a0_p = (1 / (2 * np.pi)) * np.sum(d_p, axis=0)
#     a0_sigma2 = (1 / (2 * np.pi)) * np.sum(sigma[:-1] ** 2 * dt[:, np.newaxis], axis=0)

#     a_p = []
#     b_p = []
#     a_sigma2 = []
#     b_sigma2 = []

#     for k in range(1, N + 1):
#         if k % 10 == 0:
#             print(f"Computing Fourier coefficients for k={k}")
#         cos_k_t = np.cos(k * t[:-1])[:, np.newaxis]
#         sin_k_t = np.sin(k * t[:-1])[:, np.newaxis]
#         a_p.append((1 / np.pi) * np.sum(cos_k_t * d_p, axis=0))
#         b_p.append((1 / np.pi) * np.sum(sin_k_t * d_p, axis=0))
#         a_sigma2.append(
#             (1 / np.pi) * np.sum(cos_k_t * sigma[:-1] ** 2 * dt[:, np.newaxis], axis=0)
#         )
#         b_sigma2.append(
#             (1 / np.pi) * np.sum(sin_k_t * sigma[:-1] ** 2 * dt[:, np.newaxis], axis=0)
#         )

#     a_p = np.array(a_p)
#     b_p = np.array(b_p)
#     a_sigma2 = np.array(a_sigma2)
#     b_sigma2 = np.array(b_sigma2)

#     return a0_p, a0_sigma2, a_p, b_p, a_sigma2, b_sigma2


# def reconstruct_sigma2(t, a0_sigma2, a_sigma2, b_sigma2, M=None):
#     if M is None:
#         M = len(t) // 2

#     sigma2 = np.full((len(t), 2), a0_sigma2)
#     for k in range(1, M + 1):
#         if k % 10 == 0:
#             print(f"Reconstructing sigma2 for k={k}")
#         sigma2 += (1 - k / M) * (
#             a_sigma2[k - 1] * np.cos(k * t)[:, np.newaxis]
#             + b_sigma2[k - 1] * np.sin(k * t)[:, np.newaxis]
#         )
#     return sigma2


# def plot_results(t, p, sigma2, sigma2_reconstructed):
#     plt.figure(figsize=(12, 6))

#     plt.subplot(2, 1, 1)
#     plt.plot(t, p[:, 0], label="$p_1(t)$")
#     plt.plot(t, p[:, 1], label="$p_2(t)$")
#     plt.title("Simulated Log-Prices $p_1(t)$ and $p_2(t)$")
#     plt.xlabel("Time")
#     plt.ylabel("$p(t)$")
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(t, sigma2[:, 0], label="True $\sigma^2_1(t)$", linestyle="--")
#     plt.plot(t, sigma2[:, 1], label="True $\sigma^2_2(t)$", linestyle="--")
#     plt.plot(t, sigma2_reconstructed[:, 0], label="Reconstructed $\sigma^2_1(t)$")
#     plt.plot(t, sigma2_reconstructed[:, 1], label="Reconstructed $\sigma^2_2(t)$")
#     plt.title(
#         "Reconstructed $\sigma^2_1(t)$ and $\sigma^2_2(t)$ using Fourier-Fejer formula"
#     )
#     plt.xlabel("Time")
#     plt.ylabel("$\sigma^2(t)$")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# def main():
#     # Parameters for the multivariate GARCH process
#     theta = np.array([0.035, 0.035])
#     omega = np.array([0.636, 0.636])
#     lambda_ = np.array([0.296, 0.296])
#     T = 1.0  # time horizon (1 day)
#     N = 10000  # number of time steps
#     p0 = np.array([0.0, 0.0])  # initial value of log-prices p(t)
#     sigma0 = np.array([0.2, 0.2])  # initial value of volatilities sigma(t)
#     mean_duration = 14  # mean duration between quotes in seconds
#     rho = 0.5  # correlation between the Brownian motions

#     # Simulate paths
#     t, p, sigma2 = simulate_multivariate_paths(
#         T, N, theta, omega, lambda_, p0, sigma0, rho
#     )

#     # Unevenly sampled times
#     uneven_times = unevenly_sampled_times(T, mean_duration)
#     uneven_p = np.interp(uneven_times, t, p[:, 0])

#     # Compute Fourier coefficients
#     sigma_ = np.sqrt(sigma2)
#     a0_p, a0_sigma2, a_p, b_p, a_sigma2, b_sigma2 = compute_fourier_coefficients(
#         t, p, sigma_
#     )

#     # Reconstruct sigma^2(t)
#     sigma2_reconstructed = reconstruct_sigma2(t, a0_sigma2, a_sigma2, b_sigma2)

#     # Plot results
#     plot_results(t, p, sigma2, sigma2_reconstructed)


# if __name__ == "__main__":
#     main()
