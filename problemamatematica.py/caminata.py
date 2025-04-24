import pybullet as p
import pybullet_data
import numpy as np
import time
import math

# Configurar conexión y entorno de simulación
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Cargar modelos
plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, lateralFriction=1, spinningFriction=0.5, rollingFriction=0.1)
robot_id = p.loadURDF(
    "D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf",
    [0, 0, 0.5],
    [0, 0.5, 0.5, 0],  # orientación inicial: robot mirando hacia +Y global
    useFixedBase=False
)


# Índices de los pies y articulaciones asociadas
foot_joint_indices = [3, 7, 11, 15]
leg_joint_indices_dict = {
    3: [0, 1, 2],    # FL, cadera,muslo,tibia,son los index de las patas.
    7: [4, 5, 6],    # FR
    11: [8, 9, 10],  # RL
    15: [9, 13, 14] # RR
}

# Parámetros de la caminata
step_height = 0.1
step_length = 0.1
cycle_duration = 1
dt = 1 / 240

# Posiciones base locales del pie (en el marco del cuerpo)
foot_base_positions = {
    3: [0.1, 0.2, -0.45],   # Fr
    7: [-0.1, 0.2, -0.45],  # Fl
    11: [0.1, 0.2, -0.45],   # Rr
    15: [-0.1, -0.2, -0.45]   # Rl
}

# Secuencia de caminata tipo crawl
crawl_sequence = [3, 11, 7, 15]

# Diccionario para almacenar las líneas dinámicas (debug en tiempo real)
trajectory_line_ids = {i: None for i in foot_joint_indices}

# Bucle principal de simulación
t = 0
while True:
    phase = (t % cycle_duration) / cycle_duration
    active_swing_leg_index = int((phase * 4)) % 4
    swing_leg = crawl_sequence[active_swing_leg_index]

    for foot_index in foot_joint_indices:
        base_pos = np.array(foot_base_positions[foot_index])
        progress = (phase * 4) % 1.0 if foot_index == swing_leg else ((phase * 4 + 0.75) % 1.0)

        # Trayectoria en el plano YZ
        if foot_index == swing_leg:
            y = base_pos[1] + step_length * (progress - 0.5)
            z = base_pos[2] + step_height * math.sin(math.pi * progress)
        else:
            y = base_pos[1] - step_length * (progress - 0.5)
            z = base_pos[2]

        # Mantener la posición X fija según la pierna
        foot_target_pos = [base_pos[0], y, z]

        # Calcular IK y aplicar
        joint_indices = leg_joint_indices_dict[foot_index]
        ik_result = p.calculateInverseKinematics(robot_id, foot_index, foot_target_pos)
        
        all_joint_indices = []
        all_target_positions = []

        for foot_index in foot_joint_indices:
            base_pos = np.array(foot_base_positions[foot_index])
            progress = (phase * 4) % 1.0 if foot_index == swing_leg else ((phase * 4 + 0.75) % 1.0)

            if foot_index == swing_leg:
                y = base_pos[1] + step_length * (progress - 0.5)
                z = base_pos[2] + step_height * math.sin(math.pi * progress)
            else:
                y = base_pos[1] - step_length * (progress - 0.5)
                z = base_pos[2]

            foot_target_pos = [base_pos[0], y, z]

            joint_indices = leg_joint_indices_dict[foot_index]
            ik_result = p.calculateInverseKinematics(robot_id, foot_index, foot_target_pos)

            all_joint_indices.extend(joint_indices)
            all_target_positions.extend(ik_result[:3])  # Asegúrate de que sean solo 3 articulaciones

        # Controlar todas las articulaciones a la vez
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=all_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=all_target_positions,
            forces=[20] * len(all_joint_indices)
        )

        # Visualización en tiempo real (trayectoria actual)
        base_pos_world, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        foot_target_world = np.array(base_pos_world) + base_mat @ np.array(foot_target_pos)

        current_foot_pos = p.getLinkState(robot_id, foot_index)[0]

        if trajectory_line_ids[foot_index] is not None:
            p.removeUserDebugItem(trajectory_line_ids[foot_index])
        trajectory_line_ids[foot_index] = p.addUserDebugLine(
            current_foot_pos, foot_target_world.tolist(), [1, 0, 0], 1.5, lifeTime=dt)

    p.stepSimulation()
    #time.sleep(dt)
    t += dt
