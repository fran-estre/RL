import gym 
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from GaitPattern import CPG

class QuadrupedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Conexión a PyBullet (GUI o DIRECT)
        self.physics_client = p.connect(p.GUI)  # Cambiar a p.DIRECT para entrenamiento rápido p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
        urdfFlags = p.URDF_USE_SELF_COLLISION
        # Espacios de acción y observación
        self.action_space = spaces.Box(low=np.array([0.1, 0.0, 0.5, -0.1]), 
                               high=np.array([2.0, 0.5, 0.95, 0.1]), dtype=np.float32)  # CPG parameters

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32) 
        # Robot y objetivo
        self.robot = None
        self.initial_roll = 0
        self.initial_pitch = 0
        self.reset()

    def initialize_robot_pose(self):
            # Direcciones de las articulaciones (algunos motores van al revés)
        joint_directions = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        
        # Offsets para que las patas no estén totalmente rectas
        # Se asume orden de joints: [hip, upper_leg, lower_leg] * 4 patas
        joint_offsets = [0, -0.5, 0.5] * 4
        
        # Obtener sólo las articulaciones revolutas
        joint_ids = [j for j in range(p.getNumJoints(self.robot)) 
                    if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]

        for i, joint_id in enumerate(joint_ids):
            # Aplica el offset con dirección para que las patas inicien flexionadas
            initial_target = -0.1
            initial_pos = joint_directions[i] * initial_target + joint_offsets[i]
            p.resetJointState(self.robot, joint_id, initial_pos)
    
    def reset(self,seed=None, options=None):
        # Inicializar semilla
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.plane = p.loadURDF("D:\\ITMO trabajos de la u\\tesis\\py\\testing\\pybullet_robots\\data\\plane.urdf")
        self.robot = p.loadURDF("D:\ITMO trabajos de la u\\tesis\py\\testing\pybullet_robots\data\laikago\laikago_toes.urdf", [0, 0, 0.5],[0,0.5,0.5,0],flags = urdfFlags,useFixedBase=False)
        self.initialize_robot_pose()

        #posición objetivo aleatoria en un rango
        radius = 5  # distancia máxima
        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        self.target_pos = np.array([x, y, 0])

        # Visualización: esfera pequeña en la posición objetivo
        self.target_marker = p.loadURDF("sphere_small.urdf", self.target_pos, globalScaling=2)

        # Inicializar joints
        self.joint_ids = [j for j in range(p.getNumJoints(self.robot)) 
                          if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]
        
        # Obtener índices de los enlaces "toe"
        toe_links = ["toeFL", "toeFR", "toeRL", "toeRR"]
        link_name_to_index = {}
        for j in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, j)
            link_name = joint_info[12].decode("utf-8")  # Índice 12 es el nombre del link hijo
            link_name_to_index[link_name] = j
        
        # Ajustar fricción dinámicamente
        for link_name in toe_links:
            if link_name in link_name_to_index:
                link_idx = link_name_to_index[link_name]
                p.changeDynamics(
                    bodyUniqueId=self.robot,
                    linkIndex=link_idx,
                    lateralFriction=1.5,   # Nuevo valor
                    spinningFriction=0.3,
                    frictionAnchor=1
                )

        self.prev_x = 0
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        self.prev_pos = np.array(torso_pos[:2])  # <- agrega esto
        obs, _ = self._get_obs(), {}
        self.initial_roll = obs[27]
        self.initial_pitch = obs[28]
        
        sim_rate = 240  # Hz
        wait_time = 3   # segundos
        for _ in range(sim_rate * wait_time):
            p.stepSimulation()
        return obs, {} # Devolver info adicional (vacío)

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_ids)
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])
        
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        torso_euler = p.getEulerFromQuaternion(torso_orn)
        torso_vel_lin, torso_vel_ang = p.getBaseVelocity(self.robot)
        
        direction_to_target = self.target_pos[:2] - np.array(torso_pos[:2])
        distance_to_target = np.linalg.norm(direction_to_target)

        # Calcular ángulo entre orientación del robot y vector al objetivo
        robot_yaw = torso_euler[2]
        robot_facing = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        
        if np.linalg.norm(direction_to_target) > 0:
            unit_target_dir = direction_to_target / np.linalg.norm(direction_to_target)
        else:
            unit_target_dir = np.array([1.0, 0.0])  # Por defecto hacia +x

        angle_to_target = np.arccos(np.clip(np.dot(robot_facing, unit_target_dir), -1.0, 1.0))

        obs = np.concatenate([
            joint_pos,                  # 12
            joint_vel,                  # 12
            torso_pos,                  # 3
            torso_euler,                # 3
            torso_vel_lin,              # 3
            torso_vel_ang,              # 3
            [distance_to_target],       # 1
            unit_target_dir,            # 2
            [angle_to_target]           # 1
        ])
        return obs

    def step(self, action):

        cpg = CPG(frequency=action[0], amplitude=action[1], duty_factor=action[2])

        # Actualizar fases del CPG
        cpg.update(delta_t=1/240)  # La frecuencia de simulación es 240Hz

        # Obtener las posiciones de las patas generadas por el CPG
        motor_angles = cpg.get_motor_angles()
        
        max_force = 50  # Ajustar según el robot
        # Aplicar estas posiciones a los motores
        for i, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition=motor_angles[i],
                force= max_force
            )
                
        p.stepSimulation()
        
        # Nueva observación
        obs = self._get_obs()

        # Calcular recompensa y obtener desglose
        reward, reward_info = self._compute_reward(obs)

        # Verificar si el episodio termina
        done = self._check_done(obs)

        # Devolver también el desglose en info
        info = {"reward_breakdown": reward_info}

        return obs, reward, done, False, info

    def _compute_reward(self, obs):
        torso_pos = obs[24:27]
        roll, pitch, _ = obs[27:30]
        direction_to_target = obs[30:32]  # nuevo vector incluido en observación
        joint_velocities = obs[12:24]

        current_pos_xy = np.array(torso_pos[:2])
        target_vec = self.target_pos[:2] - current_pos_xy
        distance = np.linalg.norm(target_vec)
        
        # 1. Recompensa por progreso en la dirección correcta
        movement_vector = current_pos_xy - self.prev_pos
        if np.linalg.norm(target_vec) > 0:
            unit_direction = target_vec / np.linalg.norm(target_vec)
        else:
            unit_direction = np.array([0.0, 0.0])
        forward_progress = np.dot(movement_vector, unit_direction)
        reward_progress = forward_progress * 10.0

        self.prev_pos = current_pos_xy

        # 2. Recompensa por mantenerse cerca del suelo en z razonable
        z = torso_pos[2]
        desired_z = 0.45
        reward_height = -2.0 * abs(z - desired_z)

        # 3. Penalización por inclinación (roll y pitch)
        reward_stability = -3.0 * (abs(roll) + abs(pitch))

        # 4. Penalización por esfuerzo articular (opcional)
        reward_energy = -0.001 * np.sum(np.square(joint_velocities))

        #  5. Recompensa por llegar al objetivo
        reached_target = distance < 0.3
        reward_goal = 50.0 if reached_target else 0.0

        # 6. Recompensa de supervivencia por cada paso
        reward_alive = 1.0

        # Suma total ponderada
        components = {
            "reward_progress": reward_progress,
            "reward_height": reward_height,
            "reward_stability": reward_stability,
            "reward_energy": reward_energy,
            "reward_goal": reward_goal,
            "reward_alive": reward_alive
        }
        total_reward = sum(components.values())
        return total_reward, components


    def _check_done(self, obs):
        # Terminar si el torso está muy inclinado o cae
        z_pos = obs[26]  # Posición z del torso
        roll, pitch, _ = obs[27:30]
        
        delta_roll = abs(roll - self.initial_roll)
        delta_pitch = abs(pitch - self.initial_pitch)
        # Verificar si se cayó (está muy bajo)
        fallen = z_pos < 0.1
        # O si se inclinó demasiado (roll o pitch grande)
        too_tilted = delta_roll > 1 or delta_pitch > 1
        # Verificar si el chasis toca el suelo
        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.plane)
        # Opcionalmente, puedes filtrar por el índice de enlace 0 si sabes cuál es el torso
        chassis_contact = any(cp[3] == -1 or cp[3] == 0 for cp in contact_points)  # contacto con base o link 0

        if fallen  or chassis_contact:
            return True
        
        # Si está cerca del objetivo, termina con éxito
        if np.linalg.norm(obs[24:26] - self.target_pos[:2]) < 0.3:
            return True
        return False

    def render(self, mode='human'):
        pass  # PyBullet ya renderiza en GUI

    def close(self):
        p.disconnect(self.physics_client)