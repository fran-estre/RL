import pybullet as p
import pybullet_data
import numpy as np
import time
import csv

# 1. Configuración inicial
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(
    "D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf",
    [0, 0, 0.5],
    [0, 0.5, 0.5, 0],
    useFixedBase=False
)

# 2. Esperar estabilización
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240.0)

# 3. Mapeo de nombres a índices
name_to_index = {}
for j in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, j)
    link_name = joint_info[12].decode("utf-8")
    name_to_index[link_name] = j

# 4. Obtener posiciones de las caderas
hip_links = ["FR_hip_motor", "FL_hip_motor", "RR_hip_motor", "RL_hip_motor"]
hip_positions = {}

for link in hip_links:
    idx = name_to_index[link]
    hip_positions[link] = p.getLinkState(robot, idx, computeForwardKinematics=True)[4]

# 5. Calcular distancias del rectángulo
def dist_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Lados frontales/traseros
front = dist_3d(hip_positions["FR_hip_motor"], hip_positions["FL_hip_motor"])
rear = dist_3d(hip_positions["RR_hip_motor"], hip_positions["RL_hip_motor"])

# Lados izquierdo/derecho
left = dist_3d(hip_positions["FL_hip_motor"], hip_positions["RL_hip_motor"])
right = dist_3d(hip_positions["FR_hip_motor"], hip_positions["RR_hip_motor"])

# Diagonales
diag1 = dist_3d(hip_positions["FL_hip_motor"], hip_positions["RR_hip_motor"])
diag2 = dist_3d(hip_positions["FR_hip_motor"], hip_positions["RL_hip_motor"])
# Imprimir resultados
print("\n📐 Dimensiones del rectángulo de caderas:")
print(f" - Frente (FR ↔ FL): {front:.3f} m")
print(f" - Atrás  (RR ↔ RL): {rear:.3f} m")
print(f" - Izquierda (FL ↔ RL): {left:.3f} m")
print(f" - Derecha  (FR ↔ RR): {right:.3f} m")
print(f" - Diagonal 1 (FL ↖ RR): {diag1:.3f} m")
print(f" - Diagonal 2 (FR ↙ RL): {diag2:.3f} m")
# 6. Escribir en CSV
with open('distancias_rectangulo.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Tipo", "Desde", "Hasta", "Distancia (m)"])
    
    # Escribir todas las mediciones
    writer.writerow(["Frente", "FR", "FL", f"{front:.6f}"])
    writer.writerow(["Atrás", "RR", "RL", f"{rear:.6f}"])
    writer.writerow(["Izquierdo", "FL", "RL", f"{left:.6f}"])
    writer.writerow(["Derecho", "FR", "RR", f"{right:.6f}"])
    writer.writerow(["Diagonal 1", "FL", "RR", f"{diag1:.6f}"])
    writer.writerow(["Diagonal 2", "FR", "RL", f"{diag2:.6f}"])

# 7. Visualización en PyBullet
# Colores para cada tipo de línea
colors = {
    "front": [1, 0, 0],    # Rojo
    "rear": [0, 1, 0],     # Verde
    "sides": [0, 0, 1],    # Azul
    "diagonals": [1, 0, 1] # Magenta
}

# Dibujar el rectángulo
points = {
    "FR": hip_positions["FR_hip_motor"],
    "FL": hip_positions["FL_hip_motor"],
    "RR": hip_positions["RR_hip_motor"],
    "RL": hip_positions["RL_hip_motor"]
}

# Líneas frontales/traseras
p.addUserDebugLine(points["FR"], points["FL"], colors["front"], 3)
p.addUserDebugLine(points["RR"], points["RL"], colors["rear"], 3)

# Líneas laterales
p.addUserDebugLine(points["FL"], points["RL"], colors["sides"], 3)
p.addUserDebugLine(points["FR"], points["RR"], colors["sides"], 3)

# Diagonales
p.addUserDebugLine(points["FL"], points["RR"], colors["diagonals"], 2)
p.addUserDebugLine(points["FR"], points["RL"], colors["diagonals"], 2)

# Añadir textos flotantes
def add_distance_text(p1, p2, color, offset):
    mid_point = np.array(p1) + 0.5*(np.array(p2)-np.array(p1))
    distance = dist_3d(p1, p2)
    p.addUserDebugText(
        f"{distance:.3f}m",
        mid_point + offset,
        textColorRGB=color,
        textSize=1.2
    )

add_distance_text(points["FR"], points["FL"], colors["front"], [0, 0, 0.1])
add_distance_text(points["RR"], points["RL"], colors["rear"], [0, 0, 0.1])
add_distance_text(points["FL"], points["RL"], colors["sides"], [0.1, 0, 0])
add_distance_text(points["FR"], points["RR"], colors["sides"], [-0.1, 0, 0])
add_distance_text(points["FL"], points["RR"], colors["diagonals"], [0, 0.1, 0])
add_distance_text(points["FR"], points["RL"], colors["diagonals"], [0, -0.1, 0])

# 8. Mantener simulación
print("Simulación activa. Archivo 'distancias_rectangulo.csv' generado.")
try:
    while True:
        p.stepSimulation()
        time.sleep(1/240.0)
except KeyboardInterrupt:
    p.disconnect()