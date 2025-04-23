import pybullet as p
import pybullet_data
import time
import numpy as np
from ik_leg_yz import ik_leg_yz

# Inicializar PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
robot = p.loadURDF("D:/ITMO trabajos de la u/tesis/py/testing/pybullet_robots/data/laikago/laikago_toes.urdf", [0, 0, 0.5],[0, 0.5, 0.5, 0], useFixedBase=True)

# Esperar estabilización
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

# Obtener índices de articulaciones y nombres
joint_name_to_idx = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}

# Indices para FR
lat_idx   = joint_name_to_idx["FR_hip_motor_2_chassis_joint"]
hip_idx   = joint_name_to_idx["FR_upper_leg_2_hip_motor_joint"]
knee_idx  = joint_name_to_idx["FR_lower_leg_2_upper_leg_joint"]
toe_idx   = joint_name_to_idx["jtoeRL"]

# Guardar offsets articulares (ángulos en postura inicial)
initial_offsets = {
    lat_idx:  p.getJointState(robot, lat_idx)[0],
    hip_idx:  p.getJointState(robot, hip_idx)[0],
    knee_idx: p.getJointState(robot, knee_idx)[0],
}

# Obtener la posición stance relativa en YZ
hip_world = np.array(p.getLinkState(robot, hip_idx, computeForwardKinematics=True)[4])
toe_world = np.array(p.getLinkState(robot, toe_idx, computeForwardKinematics=True)[4])
stance_yz = toe_world[[1, 2]] - hip_world[[1, 2]]

print(f"\nStance base FR: y={stance_yz[0]:.4f}, z={stance_yz[1]:.4f}")

# Pruebas: mover en puntos cercanos a la posición base
deltas = [(0, 0), (0.02, 0.02), (-0.02, 0), (0, -0.02)]
for dy, dz in deltas:
    y_target = stance_yz[0] + dy
    z_target = stance_yz[1] + dz

    print(f"\n>> Testing target y={y_target:.4f}, z={z_target:.4f}")
    result = ik_leg_yz(y_target, z_target, debug=True)
    if result is None:
        continue

    hip_angle, knee_angle = result

    # Aplicar acción con offsets
    p.setJointMotorControl2(robot, lat_idx, p.POSITION_CONTROL, targetPosition=initial_offsets[lat_idx], force=100)
    p.setJointMotorControl2(robot, hip_idx, p.POSITION_CONTROL, targetPosition=initial_offsets[hip_idx] + hip_angle, force=100)
    p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL, targetPosition=initial_offsets[knee_idx] + knee_angle, force=100)

    # Simular para alcanzar posición
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1/240)

    # Medir posición real relativa
    hip_real = np.array(p.getLinkState(robot, hip_idx, computeForwardKinematics=True)[4])
    toe_real = np.array(p.getLinkState(robot, toe_idx, computeForwardKinematics=True)[4])
    rel_pos = toe_real[[1, 2]] - hip_real[[1, 2]]

    err = rel_pos - np.array([y_target, z_target])
    print(f"Measured rel y={rel_pos[0]:.4f}, z={rel_pos[1]:.4f} → error (dy={err[0]:.4f}, dz={err[1]:.4f})")

# Dejar la simulación abierta
