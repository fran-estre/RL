import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin

# Dimensiones de Laikago
L = 0.437  # Largo del robot
W = 0.163  # Ancho del robot

def bodyIK(omega, phi, psi, xm, ym, zm):
    """
    Calcula las cuatro matrices de transformación para las patas del robot Laikago.
    """
    # Matrices de rotación
    Rx = np.array([
        [1, 0, 0, 0], 
        [0, np.cos(omega), -np.sin(omega), 0],
        [0, np.sin(omega), np.cos(omega), 0],
        [0, 0, 0, 1]])

    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi), 0], 
        [0, 1, 0, 0],
        [-np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1]])

    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0, 0], 
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    # Matriz de rotación total
    Rxyz = Rx @ Ry @ Rz

    # Matriz de traslación
    T = np.array([[0, 0, 0, xm], [0, 0, 0, ym], [0, 0, 0, zm], [0, 0, 0, 0]])
    Tm = T + Rxyz

    # Matrizes de transformación para las patas
    Trb = Tm @ np.array([
        [np.cos(pi/2), 0, np.sin(pi/2), -L/2],
        [0, 1, 0, 0],
        [-np.sin(pi/2), 0, np.cos(pi/2), -W/2],
        [0, 0, 0, 1]])

    Trf = Tm @ np.array([
        [np.cos(pi/2), 0, np.sin(pi/2), L/2],
        [0, 1, 0, 0],
        [-np.sin(pi/2), 0, np.cos(pi/2), -W/2],
        [0, 0, 0, 1]])

    Tlf = Tm @ np.array([
        [np.cos(pi/2), 0, np.sin(pi/2), L/2],
        [0, 1, 0, 0],
        [-np.sin(pi/2), 0, np.cos(pi/2), W/2],
        [0, 0, 0, 1]])

    Tlb = Tm @ np.array([
        [np.cos(pi/2), 0, np.sin(pi/2), -L/2],
        [0, 1, 0, 0],
        [-np.sin(pi/2), 0, np.cos(pi/2), W/2],
        [0, 0, 0, 1]])

    return Tlf, Trf, Tlb, Trb, Tm

# Definir los ángulos de rotación y la posición del cuerpo
omega = pi/4  # Rotación en el eje X
phi = 0  # Rotación en el eje Y
psi = 0  # Rotación en el eje Z
xm = 0  # Posición en el eje X
ym = 0  # Posición en el eje Y
zm = 0  # Posición en el eje Z

# Calcular las matrices de transformación para las patas
Tlf, Trf, Tlb, Trb, Tm = bodyIK(omega, phi, psi, xm, ym, zm)

# Punto de referencia (origen) para las patas
FP = [0, 0, 0, 1]

# Calcular la posición de las patas aplicando las matrices de transformación
CP = [x @ FP for x in [Tlf, Trf, Tlb, Trb]]

# Visualización de las patas
def setupView(limit):
    ax = plt.axes(projection="3d")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    return ax

setupView(0.3).view_init(elev=27., azim=20)
plt.plot([CP[0][0], CP[1][0], CP[3][0], CP[2][0], CP[0][0]],
         [CP[0][2], CP[1][2], CP[3][2], CP[2][2], CP[0][2]],
         [CP[0][1], CP[1][1], CP[3][1], CP[2][1], CP[0][1]], 'bo-', lw=2)

plt.show()
print("OK")
