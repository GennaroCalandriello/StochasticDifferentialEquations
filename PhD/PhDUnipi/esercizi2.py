import numpy as np
import random

theta1 = np.arctan(4 / 6)
theta2 = np.arctan(-6 / 4)
print("theta2", theta2)
print("theta1", theta1)

delta_theta = np.pi - (theta2 - theta1)

print(delta_theta)
omega = delta_theta / 3
r = 7.21 / omega**2
print("raggio =", r)
# il risultato deve essere 2,92 m
