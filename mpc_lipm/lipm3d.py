import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Parámetros físicos
LegLength1 = 0.2522
LegLength2 = 0.2510
WeightLeg1 = 0.341
WeightLeg2 = 1.095 + 1.527
WeightBase = 13.715

Height = 0.48
Width = 0.2728
Length = 0.4373

bodyX = Length / 2
bodyY = Width / 2

# Posiciones base de las patas (LF, RF, RH, LH)
feet_positions = {
    'LF': np.array([-bodyX,  bodyY]),
    'RF': np.array([ bodyX,  bodyY]),
    'RH': np.array([ bodyX, -bodyY]),
    'LH': np.array([-bodyX, -bodyY])
}

# Secuencia tipo crawl
crawl_sequence = [
    ['RF', 'RH', 'LH'],  # LF en vuelo
    ['LF', 'RH', 'LH'],  # RF en vuelo
    ['LF', 'RF', 'LH'],  # RH en vuelo
    ['LF', 'RF', 'RH']   # LH en vuelo
]

# Inicialización
x_com, y_com = 0.0, 0.0
xdot, ydot = 0.1, 0.0
z_com = Height
g = 9.81
omega = np.sqrt(g / z_com)

T_phase = 0.5
dt = 0.05
N_phase = int(T_phase / dt)

com_traj = []

fig = plt.figure(figsize=(12, 6))

for phase_idx, support_legs in enumerate(crawl_sequence):
    support_coords = np.array([feet_positions[leg] for leg in support_legs])
    x_support = np.mean(support_coords[:, 0])
    y_support = np.mean(support_coords[:, 1])

    t = np.linspace(0, T_phase, N_phase)

    A_x = 0.5 * ((x_com - x_support) + xdot / omega)
    B_x = 0.5 * ((x_com - x_support) - xdot / omega)
    x_traj = A_x * np.exp(omega * t) + B_x * np.exp(-omega * t) + x_support

    A_y = 0.5 * ((y_com - y_support) + ydot / omega)
    B_y = 0.5 * ((y_com - y_support) - ydot / omega)
    y_traj = A_y * np.exp(omega * t) + B_y * np.exp(-omega * t) + y_support

    for i in range(N_phase):
        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title(f"Fase {phase_idx+1} - Paso {i+1}")
        ax1.set_xlim(-0.4, 0.4)
        ax1.set_ylim(-0.4, 0.4)
        ax1.set_aspect('equal')
        ax1.grid()

        # Dibuja patas en soporte
        for leg in support_legs:
            pos = feet_positions[leg]
            ax1.plot([pos[0]], [pos[1]], 'go', markersize=10)
            ax1.text(pos[0], pos[1], leg)

        # Dibuja pata levantada
        airborne_leg = [leg for leg in feet_positions if leg not in support_legs][0]
        airborne_pos = feet_positions[airborne_leg]
        ax1.plot([airborne_pos[0]], [airborne_pos[1]], 'ro', markersize=10)
        ax1.text(airborne_pos[0], airborne_pos[1], airborne_leg)

        # Dibuja polígono de soporte
        poly = Polygon(support_coords, closed=True, fill=False, edgecolor='blue', linestyle='--')
        ax1.add_patch(poly)

        # Dibuja centro de masa
        ax1.plot(x_traj[i], y_traj[i], 'kx', markersize=10, label='CoM')

        # Subplot del trazado completo
        ax2 = plt.subplot(1, 2, 2)
        com_traj.append((x_traj[i], y_traj[i]))
        com_arr = np.array(com_traj)
        ax2.plot(com_arr[:, 0], com_arr[:, 1], 'k-')
        ax2.set_title("Trayectoria del CoM")
        ax2.set_xlim(-0.4, 0.4)
        ax2.set_ylim(-0.4, 0.4)
        ax2.set_aspect('equal')
        ax2.grid()

        plt.pause(0.05)

plt.tight_layout()
plt.show()

