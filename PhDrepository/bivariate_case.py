import numpy as np
import matplotlib.pyplot as plt

# Model parameters
theta1, omega1, lambda1 = 0.035, 0.636, 0.296
theta2, omega2, lambda2 = 0.054, 0.476, 0.480
rho_values = [0.89]

T = 2 * np.pi
N = 10000  # number of time steps (1 second intervals)
dt = T / N
M = N


def simulate_paths(rho):
    t = np.linspace(0, T, N + 1)
    p1, p2 = np.zeros(N + 1), np.zeros(N + 1)
    sigma1_sq, sigma2_sq = np.zeros(N + 1), np.zeros(N + 1)
    sigma1_sq[0], sigma2_sq[0] = 0.2**2, 0.2**2

    def correlated_brownian_motion():
        cov_matrix = np.array(
            [[1, rho, 0, 0], [rho, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Cholesky decomposition (see)
        L = np.linalg.cholesky(cov_matrix)
        dW = np.random.normal(size=(N, 4)) @ L.T * np.sqrt(dt)
        return dW

    dW = correlated_brownian_motion()

    for i in range(1, N + 1):
        sigma1_sq[i] = (
            sigma1_sq[i - 1]
            + lambda1 * (omega1 - sigma1_sq[i - 1]) * dt
            + np.sqrt(2 * lambda1 * theta1) * sigma1_sq[i - 1] * dW[i - 1, 2]
        )
        sigma2_sq[i] = (
            sigma2_sq[i - 1]
            + lambda2 * (omega2 - sigma2_sq[i - 1]) * dt
            + np.sqrt(2 * lambda2 * theta2) * sigma2_sq[i - 1] * dW[i - 1, 3]
        )
        p1[i] = p1[i - 1] + np.sqrt(sigma1_sq[i - 1]) * dW[i - 1, 0]
        p2[i] = p2[i - 1] + np.sqrt(sigma2_sq[i - 1]) * dW[i - 1, 1]

    return t, p1, p2, sigma1_sq, sigma2_sq


def Fourier_coefficients(t, p, sigma):
    dt = np.diff(t)
    dp = np.diff(p)

    a0_p = (1 / (2 * np.pi)) * np.sum(dp)
    a0_sigma2 = (1 / (2 * np.pi)) * np.sum(sigma[:-1] ** 2 * dt)

    a_p = []
    b_p = []
    a_sigma_sq = []
    b_sigma_sq = []

    for k in range(1, N + 1):
        if k % 1000 == 0:
            print(f"Computing Fourier coefficients for k={k}")

        cos_kt = np.cos(k * t[:-1])
        sin_kt = np.sin(k * t[:-1])
        a_p.append((1 / np.pi) * np.sum(cos_kt * dp))
        b_p.append((1 / np.pi) * np.sum(sin_kt * dp))
        a_sigma_sq.append((1 / np.pi) * np.sum(cos_kt * sigma[:-1] ** 2 * dt))
        b_sigma_sq.append((1 / np.pi) * np.sum(sin_kt * sigma[:-1] ** 2 * dt))

    a_p = np.array(a_p)
    b_p = np.array(b_p)

    a_sigma_sq = np.array(a_sigma_sq)
    b_sigma_sq = np.array(b_sigma_sq)

    return a0_p, a_p, b_p, a0_sigma2, a_sigma_sq, b_sigma_sq


def reconstruct_sigma_sq(rho):
    t, p1, p2, sigma1_sq, sigma2_sq = simulate_paths(rho)
    sigma1 = np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)

    a0_p1, a_p1, b_p1, a0_sigma1_sq, a_sigma1_sq, b_sigma1_sq = Fourier_coefficients(
        t, p1, sigma1
    )
    a0_p2, a_p2, b_p2, a0_sigma2_sq, a_sigma2_sq, b_sigma2_sq = Fourier_coefficients(
        t, p2, sigma2
    )

    sigma1_sq_prime = np.full((N + 1,), a0_sigma1_sq)
    sigma2_sq_prime = np.full((N + 1,), a0_sigma2_sq)

    for k in range(1, M + 1):
        if k % 1000 == 0:
            print(f"Reconstructing sigma1_sq for k={k}")
        sigma1_sq_prime += (1 - k / M) * (
            a_sigma1_sq[k - 1] * np.cos(k * t) + b_sigma1_sq[k - 1] * np.sin(k * t)
        )

    for k in range(1, M + 1):
        if k % 1000 == 0:
            print(f"Reconstructing sigma2_sq for k={k}")
        sigma2_sq_prime += (1 - k / M) * (
            a_sigma2_sq[k - 1] * np.cos(k * t) + b_sigma2_sq[k - 1] * np.sin(k * t)
        )

    return (
        sigma1_sq_prime,
        sigma2_sq_prime,
        sigma1_sq,
        sigma2_sq,
        p1,
        p2,
        t,
        a0_p1,
        a_p1,
        b_p1,
        a0_sigma1_sq,
        a_sigma1_sq,
        b_sigma1_sq,
        a0_p2,
        a_p2,
        b_p2,
        a0_sigma2_sq,
        a_sigma2_sq,
        b_sigma2_sq,
    )


def plot_results(t, p1, p2, sigma1, sigma2, sigma1_reconstructed, sigma2_reconstructed):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, p1, label="$p_1(t)$")
    plt.plot(t, p2, label="$p_2(t)$")
    plt.title("Simulated Log-Prices $p_1(t)$ and $p_2(t)$")
    plt.xlabel("Time")
    plt.ylabel("$p(t)$")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, sigma1, label="True $\sigma^2_1(t)$", linestyle="--")
    plt.plot(t, sigma2, label="True $\sigma^2_2(t)$", linestyle="--")
    plt.plot(t, sigma1_reconstructed, label="Reconstructed $\sigma^2_1(t)$")
    plt.plot(t, sigma2_reconstructed, label="Reconstructed $\sigma^2_2(t)$")
    plt.title(
        "Reconstructed $\sigma^2_1(t)$ and $\sigma^2_2(t)$ using Fourier-Fejer formula"
    )
    plt.xlabel("Time")
    plt.ylabel("$\sigma^2(t)$")
    plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_correlation(sigma1, sigma2, p1, p2):
    # Calculate the means
    mu_p1 = np.mean(p1)
    mu_p2 = np.mean(p2)

    # Calculate the deviations from the mean
    dev_p1 = p1 - mu_p1
    dev_p2 = p2 - mu_p2

    # Calculate the covariance
    covariance = np.mean(dev_p1 * dev_p2)

    # Calculate the standard deviations
    sigma_p1 = np.sqrt(np.mean(dev_p1**2))
    sigma_p2 = np.sqrt(np.mean(dev_p2**2))

    # Calculate the correlation coefficient
    correlation = covariance / (sigma_p1 * sigma_p2)

    return correlation


def reconstruct_processes(
    t,
    a0_p1,
    a_p1,
    b_p1,
    a0_p2,
    a_p2,
    b_p2,
):
    """Reconstruct a  using its Fourier coefficients."""
    p1_prime = np.full((N + 1,), a0_p1)
    p2_prime = np.full((N + 1,), a0_p2)

    for k in range(1, M + 1):
        p1_prime += (1 - k / M) * (
            a_p1[k - 1] * np.cos(k * t) + b_p1[k - 1] * np.sin(k * t)
        )

        p2_prime += (1 - k / M) * (
            a_p2[k - 1] * np.cos(k * t) + b_p2[k - 1] * np.sin(k * t)
        )
    return p1_prime, p2_prime


def main():
    for rho in rho_values:
        (
            sigma1_sq_prime,
            sigma2_sq_prime,
            sigma1_sq,
            sigma2_sq,
            p1,
            p2,
            t,
            a0_p1,
            a_p1,
            b_p1,
            a0_sigma1_sq,
            a_sigma1_sq,
            b_sigma1_sq,
            a0_p2,
            a_p2,
            b_p2,
            a0_sigma2_sq,
            a_sigma2_sq,
            b_sigma2_sq,
        ) = reconstruct_sigma_sq(rho)

        p1_prime, p2_prime = reconstruct_processes(
            t, a0_p1, a_p1, b_p1, a0_p2, a_p2, b_p2
        )
        sum_sigma1_sq = np.sum(sigma1_sq_prime)
        sum_sigma2_sq = np.sum(sigma2_sq_prime)
        print(f"Sum of sigma1_sq_prime: {sum_sigma1_sq}")
        print(f"Sum of sigma2_sq_prime: {sum_sigma2_sq}")

        # plot_results(t, p1, p2, sigma1_sq, sigma2_sq, sigma1_sq_prime, sigma2_sq_prime)
        rho = calculate_correlation(
            sigma1_sq_prime, sigma2_sq_prime, p1_prime, p2_prime
        )
        print(f"Correlation: {rho:.2f}")


if __name__ == "__main__":
    main()
