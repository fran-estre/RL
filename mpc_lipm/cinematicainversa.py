from mpl_toolkits import mplot3d
import numpy as np
from math import *
import matplotlib.pyplot as plt

# Longitudes de los segmentos para la pierna derecha
l1 = 0.053565  # Cadera a pierna superior
l2 = 0.253082  # Pierna superior a pierna inferior
l3 = 0.250966  # Pierna inferior a pie

# Posición objetivo de la pierna derecha (x, y, z) del pie
x = -0.05  # Ejemplo de posición en el eje X
y = -0.10  # Ejemplo de posición en el eje Y
z = 0.0   # Ejemplo de posición en el eje Z

def setupView(limit):
    ax = plt.axes(projection="3d")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    return ax

def legIK(x, y, z):
    """
    x/y/z = Posición del pie en el espacio de la pierna
    F = Distancia desde el punto de la cadera al objetivo en el plano X/Y
    G = Distancia necesaria para llegar al punto en X/Y
    H = Distancia 3D necesaria para alcanzar el punto
    """
    # Calculando F, G y H
    F = sqrt(x**2 + y**2 - l1**2)
    G = F - l2
    H = sqrt(G**2 + z**2)

    max_distance = l2 + l3  # La distancia máxima alcanzable por la pierna
    if H > max_distance:
        raise ValueError("La posición solicitada está fuera del alcance del robot.")
    
    # Cálculo de los ángulos
    theta1 = -atan2(y, x) - atan2(F, -l1)

    D = (H**2 - l3**2 - l2**2) / (2 * l2 * l3)

     # Si D está fuera de rango, la posición es inalcanzable con los eslabones actuales
    if D > 1 or D < -1:
        raise ValueError("Posición inalcanzable: Los valores de las longitudes no permiten alcanzar esta posición.")
    
    theta3 = acos(D)

    theta2 = atan2(z, G) - atan2(l3 * sin(theta3), l2 + l3 * cos(theta3))

    return theta1, theta2, theta3

# Función para calcular las posiciones de los eslabones de la pierna
def calcLegPoints(angles):
    theta1, theta2, theta3 = angles
    theta23 = theta2 + theta3

    # Coordenadas de los eslabones
    T0 = np.array([0, 0, 0, 1])
    T1 = T0 + np.array([-l1 * cos(theta1), l1 * sin(theta1), 0, 0])
    T2 = T1 + np.array([-l2 * sin(theta1), -l2 * cos(theta1), 0, 0])
    T3 = T2 + np.array([-l3 * sin(theta1) * cos(theta2), -l3 * cos(theta1) * cos(theta2), l3 * sin(theta2), 0])

    return np.array([T0, T1, T2, T3])

# Función para dibujar los puntos de la pierna
def drawLegPoints(p):
    ax = setupView(0.3)
    ax.plot([p[0][0], p[1][0], p[2][0], p[3][0]],
            [p[0][2], p[1][2], p[2][2], p[3][2]],
            [p[0][1], p[1][1], p[2][1], p[3][1]], 'k-', lw=3)
    ax.scatter([p[0][0]], [p[0][2]], [p[0][1]], color='b', s=50)  # Cadera
    ax.scatter([p[3][0]], [p[3][2]], [p[3][1]], color='r', s=50)  # Pie

# Realizando las pruebas para diferentes posiciones de z


drawLegPoints(calcLegPoints(legIK(x - 0.1, y, 0.1)))  # Posición 2

plt.show()
print("OK")
