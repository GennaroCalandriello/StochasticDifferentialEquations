import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from esercizi import *

# Parametri del moto circolare uniforme
raggio = 5  # Raggio del cerchio
velocita_angolare = 10  # Velocità angolare in radianti al secondo


# Calcola le coordinate
def calcola_coordinate(t):
    x = raggio * np.cos(velocita_angolare * t)
    y = raggio * np.sin(velocita_angolare * t)
    return x, y


# Crea il grafico
fig, ax = plt.subplots()
ax.set_xlim(-raggio - 1, raggio + 1)
ax.set_ylim(-raggio - 1, raggio + 1)
ax.set_aspect("equal")  # Imposta lo stesso aspetto per gli assi x e y

# Disegna il cerchio
cerchio = plt.Circle((0, 0), raggio, color="blue", fill=False)
ax.add_artist(cerchio)

# Punto che si muove lungo il cerchio
(punto,) = ax.plot([], [], "ro")


# Funzione di inizializzazione per l'animazione
def init():
    punto.set_data([], [])
    return (punto,)


# Funzione di animazione
def animazione(t):
    x, y = calcola_coordinate(t)
    punto.set_data(x, y)
    return (punto,)


# Crea e avvia l'animazione
ani = FuncAnimation(
    fig, animazione, frames=np.linspace(0, 10, 200), init_func=init, blit=True
)
plt.show()
convert()
print(varia)
