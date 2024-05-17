import numpy as np

varia = 89


def GradiToRadianti():
    g = float(input("Inserisci i gradi "))

    r = ((np.pi * g) / 180) / np.pi

    print(f"I radianti sono {r}")


def RadiantiToGradi():
    r = float(input("Inserisci i radianti "))

    g = 180 * (r * np.pi) / np.pi

    print(f"I gradi sono {g}")


def convert():
    str = input("cosa vuoi convertire? ")

    if str == "g":
        GradiToRadianti()
    if str == "r":
        RadiantiToGradi()


if __name__ == "__main__":
    convert()
