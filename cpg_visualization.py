import pybullet as p
import pybullet_data
import numpy as np
import time

# Clase CPG básica
class CPG:
    def __init__(self, frequency=0.5, amplitude=0.5):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = 0.0

    def update(self, delta_t):
        self.phase += 2 * np.pi * self.frequency * delta_t
        self.phase %= 2 * np.pi

    def get_output(self):
        return self.amplitude * np.cos(self.phase)

# Inicializar simulación
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.setTimeStep(1/240)
p.setRealTimeSimulation(0)

# Cargar plano y robot
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.48], useFixedBase=True)

# Configurar cámara
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0.3, -0.2, 0.2]
)

# Definir articulaciones de la pierna frontal derecha
target_joint_names = ['FR_hip_joint', 'FR_upper_leg_joint', 'FR_lower_leg_joint']
target_joint_ids = []

for i in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')
    if joint_name in target_joint_names:
        joint_id = joint_info[0]
        target_joint_ids.append(joint_id)

        # Desactivar control por defecto (modo de fuerza cero)
        p.setJointMotorControl2(
            robot, joint_id,
            controlMode=p.VELOCITY_CONTROL,
            force=100
        )

# Instancia del CPG
cpg = CPG(frequency=0.5, amplitude=0.5)

# Simulación
delta_t = 1/240
sim_duration = 10  # segundos

for _ in range(int(sim_duration / delta_t)):
    p.stepSimulation()
    cpg.update(delta_t)
    angle = cpg.get_output()

    for joint_id in target_joint_ids:
        p.setJointMotorControl2(
            bodyIndex=robot,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            force=50  # fuerza más alta para que se note el movimiento
        )

    time.sleep(delta_t)

p.disconnect()
