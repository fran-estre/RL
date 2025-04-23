import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# Iniciar PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Cargar terreno y robot
plane = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf", [0, 0, 0.5], [0, 0.5, 0.5, 0], useFixedBase=False)


# Definir articulaciones que controlan las patas
joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

joint_ids = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE]

# Ángulos iniciales en grados y radianes
initial_angles_deg = np.array([0, 0, -45, 0, 0, -45, 0, 0, -45, 0, 0, -45])
initial_angles_rad = np.radians(initial_angles_deg)

# Resetear la postura del robot
for i, joint_id in enumerate(joint_ids):
    p.resetJointState(robot_id, joint_id, initial_angles_rad[i])

# Parámetros LIPM
g = 9.81  # gravedad
z_com = 0.48  # altura del CoM
omega = np.sqrt(g / z_com)  # frecuencia natural del LIPM

# Inicialización del CoM
x_com, y_com = 0.0, 0.0
xdot, ydot = 0.1, 0.0  # velocidad inicial
x_support, y_support = 0.0, 0.0  # posición inicial del pie de apoyo (por defecto en el centro)

# Función para calcular la trayectoria LIPM
def lipm_trajectory(x_com, y_com, x_support, y_support, xdot, ydot, T, dt):
    t = np.linspace(0, T, int(T / dt))
    A_x = 0.5 * ((x_com - x_support) + xdot / omega)
    B_x = 0.5 * ((x_com - x_support) - xdot / omega)
    x_traj = A_x * np.exp(omega * t) + B_x * np.exp(-omega * t) + x_support

    A_y = 0.5 * ((y_com - y_support) + ydot / omega)
    B_y = 0.5 * ((y_com - y_support) - ydot / omega)
    y_traj = A_y * np.exp(omega * t) + B_y * np.exp(-omega * t) + y_support
    
    return x_traj, y_traj

# Función para la cinemática inversa de la pierna
def legIK(x, y, z, l1, l2, l3):
    F = np.sqrt(x**2 + y**2 - l1**2)
    G = F - l2
    H = np.sqrt(G**2 + z**2)

    if H > l2 + l3:
        raise ValueError("Posición fuera de alcance.")
    
    theta1 = -np.atan2(y, x) - np.atan2(F, -l1)
    D = (H**2 - l3**2 - l2**2) / (2 * l2 * l3)
    
    if abs(D) > 1:
        raise ValueError("Posición inalcanzable.")
    
    theta3 = np.acos(np.clip(D, -1.0, 1.0))
    theta2 = np.atan2(z, G) - np.atan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))

    return theta1, theta2, theta3

crawl_sequence = [
    ['FR', 'RL', 'RR'],  # FL en vuelo
    ['FL', 'RL', 'RR'],  # FR en vuelo
    ['FL', 'FR', 'RR'],  # RL en vuelo
    ['FL', 'FR', 'RL']   # RR en vuelo
]

# Definir las posiciones de las patas (inicializadas en posiciones relativas)

feet_positions = {
    'FL': np.array([-0.437 / 2, 0.2728 / 2]),  # Front Left
    'FR': np.array([0.437 / 2, 0.2728 / 2]),    # Front Right
    'RL': np.array([0.437 / 2, -0.2728 / 2]),   # Rear Left
    'RR': np.array([-0.437 / 2, -0.2728 / 2])   # Rear Right
}

# Longitudes de las piernas
l1 = 0.053565  # Cadera a pierna superior
l2 = 0.253082  # Pierna superior a pierna inferior
l3 = 0.250966  # Pierna inferior a pie

# Simulación: base para animar caminata tipo crawl
# Simulación: animar caminata tipo crawl
T = 0.5  # Duración de cada fase
dt = 1.0 / 240.0  # Paso de tiempo

# Simulación: animar caminata tipo crawl
T = 0.5  # Duración de cada fase
dt = 1.0 / 240.0  # Paso de tiempo

for step in range(10000):
    p.stepSimulation()
    
    # Determinar la fase actual (cíclica)
    phase = step % len(crawl_sequence)
    support_legs = crawl_sequence[phase]
    
    # Calcular posición de soporte
    support_coords = np.array([feet_positions[leg] for leg in support_legs])
    x_support = np.mean(support_coords[:, 0])
    y_support = np.mean(support_coords[:, 1])
    
    # Calcular trayectoria LIPM para el CoM
    x_traj, y_traj = lipm_trajectory(x_com, y_com, x_support, y_support, xdot, ydot, T, dt)
    
    # Actualizar posición del CoM usando el primer punto de la trayectoria
    x_com = x_traj[0]
    y_com = y_traj[0]
    
    # Mover patas en vuelo usando IK
    for leg in feet_positions:
        if leg not in support_legs:
            x, y = feet_positions[leg]
            try:
                # Ajustar z para levantar la pata (-0.3 en marco local de la cadera)
                theta1, theta2, theta3 = legIK(x, y, -0.3, l1, l2, l3)  # ¡Z modificado!
                joint_angles = [theta1, theta2, theta3]
                joint_ids_for_leg = [
                    joint_names.index(f"{leg}_hip_joint"),
                    joint_names.index(f"{leg}_thigh_joint"),
                    joint_names.index(f"{leg}_calf_joint")
                ]
                for joint_id, angle in zip(joint_ids_for_leg, joint_angles):
                    p.setJointMotorControl2(
                        robot_id,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=20
                    )
            except ValueError as e:
                print(f"Error en {leg}: {e}")

    time.sleep(dt)