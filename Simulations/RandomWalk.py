import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pylab
from numba import njit

n = 1000000
ThreeD = True
TwoD = False
randomJump = True  # instead of increasing or decreasing by 1 -> +- randint(inf, sup)


@njit()
def RandomWalk2D(n):
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(1, n):
        r = rnd.randint(1, 4)
        if randomJump:
            jump = rnd.randint(1, 3)

        if r == 1:
            x[i] = x[i - 1] + jump
            y[i] = y[i - 1]
        elif r == 2:
            x[i] = x[i - 1] - jump
            y[i] = y[i - 1]
        elif r == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + jump
        elif r == 4:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - jump

    return x, y


@njit(fastmath=True)
def RandomWalk3D(n):
    x = np.zeros(n)
    y = np.zeros(n)
    z = y.copy()
    start = 2

    x[0] = start
    y[0] = start
    x[0] = start

    for i in range(1, n):
        r = rnd.randint(1, 6)
        if randomJump:
            jump = rnd.randint(1, 7)

        if r == 1:
            x[i] = x[i - 1] + jump
            y[i] = y[i - 1]
            z[i] = z[i - 1]
        elif r == 2:
            x[i] = x[i - 1] - jump
            y[i] = y[i - 1]
            z[i] = z[i - 1]
        elif r == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + jump
            z[i] = z[i - 1]
        elif r == 4:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - jump
            z[i] = z[i - 1]
        elif r == 5:
            x[i] = x[i - 1]
            y[i] = y[i - 1]
            z[i] = z[i - 1] + jump
        elif r == 6:
            x[i] = x[i - 1]
            y[i] = y[i - 1]
            z[i] = z[i - 1] - jump

    return x, y, z


if TwoD:

    a, b = RandomWalk2D(n)
    pylab.plot(a, b)
    pylab.show()

if ThreeD:

    x, y, z = RandomWalk3D(n)
    x1, y1, z1 = RandomWalk3D(n)
    pylab.figure()
    ax = pylab.axes(projection="3d")
    ax.plot3D(x, y, z, "green")
    ax.plot3D(x1, y1, z1, "red")
    pylab.show()

