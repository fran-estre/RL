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
robot_id = p.loadURDF("D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf", 
                     [0, 0, 0.6], [0, 0.5, 0.5, 0], useFixedBase=True)

# Identificar joints de las patas
num_joints = p.getNumJoints(robot_id)
foot_joint_indices = [3, 7, 11, 15]  # Los índices de los joints de los "toes" (pies)
leg_joint_indices_dict = {
    3: [0, 1, 2],   # Cadera, pierna superior, pierna inferior para el pie 3
    7: [4, 5, 6],   # Cadera, pierna superior, pierna inferior para el pie 7
    11: [8, 9, 10], # Cadera, pierna superior, pierna inferior para el pie 11
    15: [12, 13, 14] # Cadera, pierna superior, pierna inferior para el pie 15
}

# Configuraciones de control
limits = [
    [-math.radians(20), math.radians(20)],  # Límites para la cadera
    [-math.radians(60), math.radians(53)],  # Límites para la pierna superior
    [-math.radians(90), math.radians(37)]   # Límites para la pierna inferior
]
damping = [5] * 12  # Coeficientes de amortiguamiento para las articulaciones

# Ajuste de parámetros adicionales para la cinemática inversa
ll = [limit[0] for limit in limits]  # Límites inferiores para las articulaciones
ul = [limit[1] for limit in limits]  # Límites superiores para las articulaciones
jr = [limit[1] - limit[0] for limit in limits]  # Rango de las articulaciones (diferencia entre límites)
rp = [0] * len(limits)  # Punto de reposo para las articulaciones (puedes ajustarlo si es necesario)
jd = damping  # Coeficientes de amortiguamiento que ya definiste

# Función para mover un pie específico (usando IK para obtener los ángulos de la cadera, pierna superior e inferior)
def move_foot_to_point(robot_id, foot_joint_index, leg_joint_indices, target_pos):
    # Calcular la cinemática inversa para mover el pie
    joint_poses = p.calculateInverseKinematics(robot_id, foot_joint_index, target_pos,
                                               lowerLimits=ll, upperLimits=ul,
                                               jointRanges=jr, restPoses=rp,
                                               jointDamping=jd)
    
    # Asignar los ángulos de las articulaciones correspondientes (cadera, pierna superior, pierna inferior)
    for i, joint_index in enumerate(leg_joint_indices):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_poses[i],
            force=100,
            positionGain=1,
            velocityGain=0.1
        )

# Objetivo de las posiciones de los pies (puedes ajustar estos valores)
target_positions = {
    3: [0.3, 0.2, 0.0],   # Pie 3 (delante derecha)
    7: [0.3, 0.2, 0.0],    # Pie 7 (delante izquierda)
    11: [-0.3, -0.2, 0.0], # Pie 11 (trasero derecho)
    15: [-0.3, 0.2, 0.0]   # Pie 15 (trasero izquierdo)
}

# Ejecutar la simulación y mover los pies a las posiciones objetivo
while True:
    # Mover cada pie de acuerdo a su índice en foot_joint_indices
    for foot_index in foot_joint_indices:
        leg_joint_indices = leg_joint_indices_dict[foot_index]  # Obtener los índices de las articulaciones correspondientes
        target_pos = target_positions[foot_index]  # Obtener la posición objetivo del pie
        move_foot_to_point(robot_id, foot_index, leg_joint_indices, target_pos)
    
    p.stepSimulation()  # Avanzar en la simulación
    time.sleep(0.01)  # Controlar la velocidad de la simulación (ajustar según sea necesario)

# Finalizar conexión
p.disconnect()
