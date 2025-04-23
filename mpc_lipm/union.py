import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# Iniciar PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Cargar modelos
plane = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf", 
                     [0, 0, 0.6], [0, 0.5, 0.5, 0], useFixedBase=True)

# Definir los índices de las articulaciones para cada pie
leg_joint_indices_dict = {
    3: [0, 1, 2],   # Cadera, pierna superior, pierna inferior para el pie 3 (FL)
    7: [4, 5, 6],   # Cadera, pierna superior, pierna inferior para el pie 7 (FR)
    11: [8, 9, 10], # Cadera, pierna superior, pierna inferior para el pie 11 (RL)
    15: [12, 13, 14] # Cadera, pierna superior, pierna inferior para el pie 15 (RR)
}

# Longitudes de las piernas (valores para Laikago)
l1 = 0.053565  # Cadera a pierna superior
l2 = 0.253082  # Pierna superior a pierna inferior
l3 = 0.250966  # Pierna inferior a pie

# Función para la cinemática inversa de la pierna (manual)
def legIK(x, y, z, l1, l2, l3):
    """
    Calcula los ángulos de las articulaciones de la pierna (cadera, muslo, rodilla)
    para alcanzar una posición (x, y, z) de la punta del pie en 3D.
    """
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

# Función para mover una pierna a una posición específica usando los índices de las articulaciones
def move_leg_to_point(robot_id, foot_joint_index, leg_joint_indices, target_pos):
    """
    Calcula la cinemática inversa y mueve la pierna a la posición deseada.
    """
    joint_poses = p.calculateInverseKinematics(robot_id, foot_joint_index, target_pos)
    return joint_poses

# Comparación de los ángulos de ambas funciones
def compare_angles(x, y, z, foot_index):
    # Calcular los ángulos usando legIK (manual)
    theta1_legik, theta2_legik, theta3_legik = legIK(x, y, z, l1, l2, l3)
    
    # Obtener los índices de las articulaciones de la pierna
    leg_joint_indices = leg_joint_indices_dict[foot_index]
    
    # Obtener los ángulos usando move_leg_to_point (PyBullet)
    joint_poses = move_leg_to_point(robot_id, foot_index, leg_joint_indices, [x, y, z])
    
    # Mostrar los resultados
    print(f"Ángulos usando legIK (manual):")
    print(f"  Cadera (hip): {np.degrees(theta1_legik)}°")
    print(f"  Muslo (upper leg): {np.degrees(theta2_legik)}°")
    print(f"  Rodilla (lower leg): {np.degrees(theta3_legik)}°\n")
    
    print(f"Ángulos usando move_leg_to_point (PyBullet):")
    print(f"  Cadera (hip): {np.degrees(joint_poses[0])}°")
    print(f"  Muslo (upper leg): {np.degrees(joint_poses[1])}°")
    print(f"  Rodilla (lower leg): {np.degrees(joint_poses[2])}°")
    
    # Comparar los resultados
    print("\nComparación:")
    print(f"  Diferencia en Cadera: {np.degrees(theta1_legik - joint_poses[0]):.2f}°")
    print(f"  Diferencia en Muslo: {np.degrees(theta2_legik - joint_poses[1]):.2f}°")
    print(f"  Diferencia en Rodilla: {np.degrees(theta3_legik - joint_poses[2]):.2f}°")

# Ejecutar la comparación para la pierna FL (pie 3) y la posición objetivo
compare_angles(0.3, 0.2, 0.0, 3)

# Finalizar conexión
p.disconnect()
