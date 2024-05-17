#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <cmath>
#include <fftw3.h>

// Function to compute the Fourier estimator
double fourier_estimator(const std::vector<double>& t, const std::vector<double>& p) {
    size_t N = t.size();
    std::vector<double> dt(N-1);
    std::vector<double> d_p(N-1);
    for (size_t i = 1; i < N; ++i) {
        dt[i-1] = t[i] - t[i-1];
        d_p[i-1] = p[i] - p[i-1];
    }

    // Compute the mean of dt
    double dt_mean = std::accumulate(dt.begin(), dt.end(), 0.0) / dt.size();

    // Normalize d_p by sqrt(dt)
    std::vector<double> normalized_d_p(N-1);
    for (size_t i = 0; i < N-1; ++i) {
        normalized_d_p[i] = d_p[i] / std::sqrt(dt[i]);
    }

    // Perform FFT
    std::vector<std::complex<double>> dft(N-1);
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N-1));
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N-1));
    for (size_t i = 0; i < N-1; ++i) {
        in[i][0] = normalized_d_p[i];
        in[i][1] = 0.0;
    }

    p = fftw_plan_dft_1d(N-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (size_t i = 0; i < N-1; ++i) {
        dft[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    // Compute a0
    double a0 = 0.0;
    for (size_t i = 0; i < N-1; ++i) {
        a0 += d_p[i] * d_p[i] / dt[i];
    }
    a0 /= (N - 1);

    // Compute the sum of squares of the Fourier coefficients
    double sum_squares = 0.0;
    for (size_t i = 0; i < N-1; ++i) {
        sum_squares += std::norm(dft[i]);
    }

    return 2.0 * sum_squares / (N - 1);
}

int main() {
    std::vector<double> input(1024);  // Your input data
    std::vector<fftw_complex> output(1024/2+1);  // Output data

    fftw_plan plan = fftw_plan_dft_r2c_1d(1024, input.data(), output.data(), FFTW_ESTIMATE);

    // Execute the plan
    fftw_execute(plan);

    // Don't forget to destroy the plan when you're done
    fftw_destroy_plan(plan);

    return 0;
}