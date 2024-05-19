import numpy as np
import matplotlib.pyplot as plt

# theta = 0.035  # parameter estimated by Andersen and Bollerslev (1998a)
# omega = 0.636  # parameter estimated by Andersen and Bollerslev (1998a)
# lambda_ = 0.296  # parameter estimated by Andersen and Bollerslev (1998a)
# T = 1.0  # time horizon (1 day)
# N = 10000  # number of time steps (1 second intervals)
# p0 = 0.0  # initial value of log-price p(t)
# sigma0 = 0.2  # initial value of volatility sigma(t)
# mean_duration = 14
# F_coeff = 1000

theta = 0.035  # parameter estimated by Andersen and Bollerslev (1998a)
omega = 0.636  # parameter estimated by Andersen and Bollerslev (1998a)
lambda_ = 0.296  # parameter estimated by Andersen and Bollerslev (1998a)
T = 1.0  # time horizon (1 day)
N = 10000  # number of time steps (1 second intervals)
p0 = 0.0  # initial value of log-price p(t)
sigma0 = 0.2  # initial value of volatility sigma(t)
mean_duration = 14
F_coeff = 10000


def simulate_paths(T, N, theta, omega, lambda_, p0, sigma0):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    p = np.zeros(N + 1)
    mu = 0.1
    sigma2 = np.zeros(N + 1)
    p[0] = p0
    sigma2[0] = sigma0**2
    dW1 = np.random.normal(0, np.sqrt(dt), N)
    dW2 = np.random.normal(0, np.sqrt(dt), N)
    for i in range(1, N + 1):
        sigma2[i] = (
            sigma2[i - 1]
            + theta * (omega - sigma2[i - 1]) * dt
            + np.sqrt(2 * lambda_ * theta) * sigma2[i - 1] * dW2[i - 1]
        )
        p[i] = p[i - 1] + np.sqrt(sigma2[i - 1]) * dW1[i - 1] + mu * dt

    return t, p, np.sqrt(np.array(sigma2))


def compute_fourier_coefficients(t, p, sigma):
    print("Computing Fourier coefficients...")

    dt = np.diff(t)
    dp = np.diff(p)

    a0_p = (1 / (2 * np.pi)) * np.sum(dp)
    a0_sigma2 = (1 / (2 * np.pi)) * np.sum(sigma[:-1] ** 2 * dt)

    a_p = []
    b_p = []
    a_sigma2 = []
    b_sigma2 = []

    for k in range(1, N + 1):
        if k % 1000 == 0:
            print(f"Computing Fourier coefficients for k={k}")
        cos_kt = np.cos(k * t[:-1])
        sin_kt = np.sin(k * t[:-1])
        a_p.append((1 / np.pi) * np.sum(cos_kt * dp))
        b_p.append((1 / np.pi) * np.sum(sin_kt * dp))
        a_sigma2.append((1 / np.pi) * np.sum(cos_kt * sigma[:-1] ** 2 * dt))
        b_sigma2.append((1 / np.pi) * np.sum(sin_kt * sigma[:-1] ** 2 * dt))

    print(
        "Fourier coefficients computed successfully for the univariate case!",
        a_sigma2,
        b_sigma2,
    )
    a_p = np.array(a_p)
    b_p = np.array(b_p)
    a_sigma2 = np.array(a_sigma2)
    b_sigma2 = np.array(b_sigma2)

    return a0_p, a0_sigma2, a_p, b_p, a_sigma2, b_sigma2


def Fourier_analysis():
    t, p, sigma = simulate_paths(T, N, theta, omega, lambda_, p0, sigma0)
    a0_p, a0_sigma2, a_p, b_p, a_sigma2, b_sigma2 = compute_fourier_coefficients(
        t, p, sigma
    )
    M = N

    sigma_sq_prime = np.full((N + 1,), a0_sigma2)

    for k in range(1, M + 1):
        if k % 100 == 0:
            print(f"Reconstructing sigma_sq for k={k} univariate case")
        sigma_sq_prime += (1 - k / M) * (
            a_sigma2[k - 1] * np.cos(k * t) + b_sigma2[k - 1] * np.sin(k * t)
        )

    return sigma_sq_prime, t, p, sigma**2


def unevenly_sampled_times(T, mean_duration):
    times = [0]
    while times[-1] < T:
        times.append(times[-1] + np.random.exponential(mean_duration))
    return np.array(times)


def compute_squared_daily_return(p):
    return (p[-1] - p[0]) ** 2


def compute_cumulative_intraday_returns(t, p, interval):
    interval_seconds = interval * 60
    intervals = np.arange(0, t[-1], interval_seconds)
    squared_returns = []
    for i in range(1, len(intervals)):
        idx = (t >= intervals[i - 1]) & (t < intervals[i])
        if np.sum(idx) > 0:
            squared_returns.append((p[idx][-1] - p[idx][0]) ** 2)
    return np.sum(squared_returns)


def fourier_estimator(t, p):
    N = len(t)
    dt = np.diff(t)
    d_p = np.diff(p)
    dft = np.fft.fft(d_p / np.sqrt(dt))
    freqs = np.fft.fftfreq(N - 1, d=dt.mean())
    a0 = (1 / (N - 1)) * np.sum(d_p**2 / dt)
    return 2 * np.sum((dft.real**2 + dft.imag**2) / (N - 1))


# Parameters


def main():

    # Simulate paths
    sigma_sq_fourier, t, p, sigma_sq = Fourier_analysis()

    # Unevenly sampled times
    uneven_times = unevenly_sampled_times(T, mean_duration)
    uneven_p = np.interp(uneven_times, t, p)

    # Compute integrated volatility using different estimators
    squared_daily_return = compute_squared_daily_return(uneven_p)
    cumulative_intraday_returns = compute_cumulative_intraday_returns(
        uneven_times, uneven_p, interval=5
    )
    fourier_integrated_volatility = fourier_estimator(uneven_times, uneven_p)

    print("Squared Daily Return:", squared_daily_return)
    print("Cumulative Intraday Returns:", cumulative_intraday_returns)
    print("Fourier Integrated Volatility:", fourier_integrated_volatility)

    # estimation via Fourier
    sigma = np.sqrt(sigma_sq)
    # sigma2prime, a0_sigma2, a_sigma2, b_sigma2, a0_p, a_p, b_p = Fourier_analysis()
    sum_sigma_sq = np.sum(sigma_sq)
    print("Sum of sigma_sq:", sum_sigma_sq)

    # Plotting the results
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(t, p, label="Log-Price $p(t)$")
    # plt.scatter(
    #     uneven_times, uneven_p, color="red", s=1, label="Unevenly Sampled $p(t)$"
    # )
    plt.title("Log-Price and Variance Process Simulation (Euler-Maruyama Method)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Log-Price")
    plt.xlim(0, T)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, sigma_sq, label=" true $\sigma^2(t)$", color="r", linestyle="-")
    plt.plot(t, sigma_sq_fourier, label="Fourier $\sigma^2(t)$", linestyle="-")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Variance $\sigma^2(t)$")
    plt.xlim(0, T)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # questo istogramma è inutile, dovremmo generare un sacco di campioni per capire se distribuzione è corretta ( ===> Monte Carlo Simulation)
    # plt.figure()
    # # sigma2 = np.diff(p) ** 2
    # plt.hist(sum_sigma2, bins=30, color="r", alpha=0.5, label="True $\sigma^2(t)$")
    # plt.show()

    print("Biased: ", np.sum(np.diff(p) ** 2))
    print("Unbiased: ", np.sum(sum_sigma_sq) / N)


if __name__ == "__main__":
    main()
