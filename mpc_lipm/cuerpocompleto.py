import numpy as np
import matplotlib.pyplot as plt
from math import *

# Dimensiones de Laikago
L = 0.437  # Largo del robot
W = 0.163  # Ancho del robot

# Longitudes de los segmentos para la pierna derecha
l1 = 0.053565  # Cadera a pierna superior
l2 = 0.253082  # Pierna superior a pierna inferior
l3 = 0.250966  # Pierna inferior a pie

# Configuración del gráfico 3D
def setupView(limit):
    ax = plt.axes(projection="3d")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax

# Función para la cinemática inversa de la pierna (sin cambios)
def legIK(x, y, z):
    F = sqrt(x**2 + y**2 - l1**2)
    G = F - l2
    H = sqrt(G**2 + z**2)

    if H > l2 + l3:
        raise ValueError("Posición fuera de alcance.")
    
    theta1 = -atan2(y, x) - atan2(F, -l1)
    D = (H**2 - l3**2 - l2**2) / (2 * l2 * l3)
    
    if abs(D) > 1:
        raise ValueError("Posición inalcanzable.")
    
    theta3 = acos(D)
    theta2 = atan2(z, G) - atan2(l3 * sin(theta3), l2 + l3 * cos(theta3))

    return theta1, theta2, theta3

# Función para calcular las posiciones de los eslabones (sin cambios)
def calcLegPoints(angles):
    theta1, theta2, theta3 = angles
    T0 = np.array([0, 0, 0, 1])
    T1 = T0 + np.array([-l1 * cos(theta1), l1 * sin(theta1), 0, 0])
    T2 = T1 + np.array([-l2 * sin(theta1), -l2 * cos(theta1), 0, 0])
    T3 = T2 + np.array([-l3 * sin(theta1) * cos(theta2), -l3 * cos(theta1) * cos(theta2), l3 * sin(theta2), 0])
    return np.array([T0, T1, T2, T3])

# Función modificada para dibujar usando un solo eje
def drawLegPoints(ax, p):
    ax.plot([p[0][0], p[1][0], p[2][0], p[3][0]], 
            [p[0][2], p[1][2], p[2][2], p[3][2]], 
            [p[0][1], p[1][1], p[2][1], p[3][1]], 'k-', lw=3)
    ax.scatter(p[0][0], p[0][2], p[0][1], color='b', s=50)
    ax.scatter(p[3][0], p[3][2], p[3][1], color='r', s=50)

# Función de la cinemática directa del cuerpo (sin cambios)
def bodyIK(omega, phi, psi, xm, ym, zm):
    Rx = np.array([[1, 0, 0, 0], 
                   [0, cos(omega), -sin(omega), 0],
                   [0, sin(omega), cos(omega), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[cos(phi), 0, sin(phi), 0], 
                   [0, 1, 0, 0],
                   [-sin(phi), 0, cos(phi), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[cos(psi), -sin(psi), 0, 0], 
                   [sin(psi), cos(psi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    Rxyz = Rx @ Ry @ Rz
    T = np.array([[0, 0, 0, xm], [0, 0, 0, ym], [0, 0, 0, zm], [0, 0, 0, 0]])
    Tm = T + Rxyz

    Trb = Tm @ np.array([[cos(pi/2), 0, sin(pi/2), -L/2],
                         [0, 1, 0, 0],
                         [-sin(pi/2), 0, cos(pi/2), -W/2],
                         [0, 0, 0, 1]])
    Trf = Tm @ np.array([[cos(pi/2), 0, sin(pi/2), L/2],
                         [0, 1, 0, 0],
                         [-sin(pi/2), 0, cos(pi/2), -W/2],
                         [0, 0, 0, 1]])
    Tlf = Tm @ np.array([[cos(pi/2), 0, sin(pi/2), L/2],
                         [0, 1, 0, 0],
                         [-sin(pi/2), 0, cos(pi/2), W/2],
                         [0, 0, 0, 1]])
    Tlb = Tm @ np.array([[cos(pi/2), 0, sin(pi/2), -L/2],
                         [0, 1, 0, 0],
                         [-sin(pi/2), 0, cos(pi/2), W/2],
                         [0, 0, 0, 1]])
    return Tlf, Trf, Tlb, Trb, Tm

# Función modificada para dibujar todo en un solo eje
def drawRobot(Lp, angles, center):
    ax = setupView(0.5)  # <--- Crear el eje aquí
    
    (omega, phi, psi) = angles
    (xm, ym, zm) = center

    FP = [0, 0, 0, 1]
    Tlf, Trf, Tlb, Trb, Tm = bodyIK(omega, phi, psi, xm, ym, zm)

    CP = [x @ FP for x in [Tlf, Trf, Tlb, Trb]]
    CPs = [CP[x] for x in [0, 1, 3, 2, 0]]
    
    # Dibujar cuerpo
    ax.plot([x[0] for x in CPs], 
            [x[2] for x in CPs], 
            [x[1] for x in CPs], 'go-', lw=2, alpha=0.5)

    # Dibujar patas usando el mismo eje
    def drawLegPair(Tl, Tr, Ll, Lr):
        Ix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Ll_x, Ll_y, Ll_z, _ = np.linalg.inv(Tl) @ Ll
        Lr_x, Lr_y, Lr_z, _ = np.linalg.inv(Tr) @ Lr
        
        # Pierna izquierda
        try:
            angles = legIK(Ll_x, Ll_y, Ll_z)
            points = [Tl @ x for x in calcLegPoints(angles)]
            drawLegPoints(ax, points)
        except ValueError as e:
            print(f"Error pierna izquierda: {e}")
        
        # Pierna derecha
        try:
            angles = legIK(Lr_x, Lr_y, Lr_z)
            points = [Tr @ Ix @ x for x in calcLegPoints(angles)]
            drawLegPoints(ax, points)
        except ValueError as e:
            print(f"Error pierna derecha: {e}")

    # Dibujar pares de patas
    drawLegPair(Tlf, Trf, Lp[0], Lp[1])
    drawLegPair(Tlb, Trb, Lp[2], Lp[3])

# Parámetros de prueba
Lp = np.array([
    [0.2, -0.1, 0.1, 1],
    [0.1, -0.1, -0.1, 1],
    [-0.1, -0.1, 0.1, 1],
    [-0.1, -0.1, -0.1, 1]
])

# Ejecutar
drawRobot(Lp, (0, 0, 0), (0, 0.5, 0))  #xzy
plt.show()