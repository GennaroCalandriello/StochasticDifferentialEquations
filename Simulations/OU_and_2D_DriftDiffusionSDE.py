import numpy as np
import matplotlib.pyplot as plt
import pylab
from HestonModel import CorrRandVariables


def OU(max_time, sigma, steps, theta):

    """Ornstein-Uhlenbeck integration"""

    dt = max_time / steps
    x = np.zeros(steps)
    OUProcesses = 10  # number of paths to simulate

    for p in range(OUProcesses):

        dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=steps)

        for i in range(steps):
            x[i] = x[i - 1] - theta * x[i - 1] * dt + sigma * dw[i]

        plt.plot(range(steps), x)
    plt.show()


def TwoDimDiffusionProcess(max_time, steps, correlated=True):

    """We're talking about the following system:
    
    (dy1, dy2)=(A1, A2)dt+(B11/(1+e^(y1^2) B12, B21, B22/(1+e^y2^2)))*(dW1, dW2)
    
    """

    dt = max_time / steps

    # drift vectos
    A = np.array([2.0, 1.0])

    # diffusion matrix:
    B = np.array([[0.5, 0.0], [0.0, 0.5]])

    y = np.zeros([steps, 2])

    # Wiener processes:
    if correlated:

        dW = np.zeros([steps, 2])
        dW1, dW2, _ = CorrRandVariables(steps)

        for i in range(steps):
            dW[i, 0] = dW1[i]
            dW[i, 1] = dW2[i]

    else:
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=[steps, 2])

    # integration
    for i in range(steps):
        y[i, 0] = (
            y[i - 1, 0]
            - A[0] * y[i - 1, 0] * dt
            + B[0, 0] / (1 + np.exp(y[i - 1, 0] ** 2)) * dW[i, 0]
            + B[0, 1] * dW[i, 1]
        )
        y[i, 1] = (
            y[i - 1, 1]
            - A[1] * y[i - 1, 1] * dt
            + B[1, 0] * dW[i, 0]
            + B[1, 1] / (1 + np.exp(y[i - 1, 1] ** 2)) * dW[i, 1]
        )

    return y, np.linspace(0, max_time, steps)


if __name__ == "__main__":

    pylab.figure()
    ax = pylab.axes(projection="3d")

    for _ in range(2):
        y, t = TwoDimDiffusionProcess(20, 1000)
        ax.plot3D(t, y[:, 0], y[:, 1])

    pylab.show()

    # OU(100, 0.1, 10000, 0.2)

