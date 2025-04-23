import numpy as np
import matplotlib.pyplot as plt

# Parámetros reales del robot (Laikago)
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
bodyZ = Height

# Posiciones base de las patas (LF, RF, RH, LH)
feet_positions = {
    'LF': np.array([-bodyX,  bodyY]),
    'RF': np.array([ bodyX,  bodyY]),
    'RH': np.array([ bodyX, -bodyY]),
    'LH': np.array([-bodyX, -bodyY])
}

# Secuencia de soporte tipo crawl: una pata se levanta en cada fase
crawl_sequence = [
    ['RF', 'RH', 'LH'],  # mueve LF
    ['LF', 'RH', 'LH'],  # mueve RF
    ['LF', 'RF', 'LH'],  # mueve RH
    ['LF', 'RF', 'RH']   # mueve LH
]

# Inicialización del CoM
x_com, y_com = 0.0, 0.0
xdot, ydot = 0.1, 0.0  # velocidad inicial suave
z_com = Height
g = 9.81
omega = np.sqrt(g / z_com)

# Simulación por fase
T_phase = 0.5
dt = 0.01
N_phase = int(T_phase / dt)

X_total = []
Y_total = []

for support_legs in crawl_sequence:
    # Calcular centro del polígono de soporte
    support_coords = np.array([feet_positions[leg] for leg in support_legs])
    x_support = np.mean(support_coords[:, 0])
    y_support = np.mean(support_coords[:, 1])

    # Resolver LIPM analíticamente para esta fase
    t = np.linspace(0, T_phase, N_phase)

    A_x = 0.5 * ((x_com - x_support) + xdot / omega)
    B_x = 0.5 * ((x_com - x_support) - xdot / omega)
    x_traj = A_x * np.exp(omega * t) + B_x * np.exp(-omega * t) + x_support

    A_y = 0.5 * ((y_com - y_support) + ydot / omega)
    B_y = 0.5 * ((y_com - y_support) - ydot / omega)
    y_traj = A_y * np.exp(omega * t) + B_y * np.exp(-omega * t) + y_support

    # Guardar resultados
    X_total.extend(x_traj)
    Y_total.extend(y_traj)

    # Actualizar condiciones iniciales para la siguiente fase
    x_com = x_traj[-1]
    y_com = y_traj[-1]
    xdot = omega * (A_x * np.exp(omega * T_phase) - B_x * np.exp(-omega * T_phase))
    ydot = omega * (A_y * np.exp(omega * T_phase) - B_y * np.exp(-omega * T_phase))

# Visualización
plt.figure(figsize=(6, 6))
plt.plot(X_total, Y_total, label="Trayectoria del CoM")
for leg, pos in feet_positions.items():
    plt.scatter(pos[0], pos[1], label=leg)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trayectoria CoM en caminata tipo crawl (LIPM)")
plt.axis("equal")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
