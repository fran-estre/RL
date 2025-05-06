import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import math

class QuadrupedEnv(gym.Env):
    def __init__(self, render_mode=None, goal_range=2.0, epsilon=0.1):
        super().__init__()
        # Parámetros de goal-conditioned MDP
        self.goal_range = goal_range    # L: rango de muestreo del waypoint
        self.epsilon    = epsilon       # umbral de éxito
        self.cam_distance = 1.5
        self.cam_yaw      = 50
        self.cam_pitch    = -30
        # Conexión con PyBullet
        mode = p.GUI if render_mode=="human" else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space (igual)
        self.action_space = spaces.Box(-1.0,1.0,(12,),dtype=np.float32)
        self.robot = None
        self.joint_ids = None
        self.initial_joint_angles_deg = np.array([0, 0, -45, 0.0, 0,-45, 0.0, 0, -45, 0.0, 0, -45])
        self.initial_joint_angles_rad = np.radians(self.initial_joint_angles_deg)
        # Observation: 33 estados + 2 metas de velocidad (si lo mantienes) + 2 deltas de posición
        # Aquí sólo añadimos los 2 deltas posicionales:
        low  = np.concatenate([ -np.inf*np.ones(33),  [-self.goal_range, -self.goal_range] ])
        high = np.concatenate([  np.inf*np.ones(33),  [ self.goal_range,  self.goal_range] ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Inicializa
        self.reset()

    def create_rough_terrain(physics_client_id):
        """
        Crea un terreno irregular usando malla de altura.
        """
        rows = cols = 512
        data = [0]*(rows*cols)
        for j in range(cols//2):
            for i in range(rows//2):
                h = np.random.uniform(0, 0.06)
                idx = 2*i + 2*j*rows
                data[idx] = data[idx+1] = data[idx+rows] = data[idx+rows+1] = h

        # usa p.createCollisionShape y pasa physicsClientId
        shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.09, 0.05, 1],
            heightfieldTextureScaling=(rows-1)/2,
            heightfieldData=data,
            numHeightfieldRows=rows,
            numHeightfieldColumns=cols,
            physicsClientId=physics_client_id
        )
        # idem al crear el multibody
        plane = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            physicsClientId=physics_client_id
        )
        return plane    
    
    def render(self, mode="human"):
        if mode != "human":
            return
        pos, _ = p.getBasePositionAndOrientation(
            self.robot,
            physicsClientId=self.physics_client
        )
        p.resetDebugVisualizerCamera(
            cameraDistance     = self.cam_distance,
            cameraYaw          = self.cam_yaw,
            cameraPitch        = self.cam_pitch,
            cameraTargetPosition = pos,
            physicsClientId    = self.physics_client
        )
   
    def reset(self, seed=None, options=None):
        if options and 'goal_pos' in options:
            self.goal_pos = np.array(options['goal_pos'], dtype=np.float32)
        else:
            θ     = np.random.uniform(0, 2*math.pi)
            v_mod = np.random.uniform(0, 1.0)
            self.goal_pos = np.array([math.cos(θ), math.sin(θ)]) * v_mod
            
        # Reinicio de PyBullet
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago/laikago_toes_zup.urdf",
                                 [0,0,0.55],[0,0,0,1],
                                 useFixedBase=False)
        # (resto de lod joints/fricción idéntico...)
        p.changeDynamics(self.plane, -1, lateralFriction=1, spinningFriction=0.5, rollingFriction=0.1)

        lower_legs = ["FR_lower_leg", "FL_lower_leg", "RR_lower_leg", "RL_lower_leg"]

        '''for j in range(p.getNumJoints(self.robot)):
                joint_info = p.getJointInfo(self.robot, j)
                link_name = joint_info[12].decode("utf-8")
                if link_name in lower_legs:
                    p.setCollisionFilterGroupMask(self.robot, j, 0, 0)'''
        
        # Ajustar fricción y dinámica de contacto para los toes
        '''toe_links = ["toeFR", "toeFL", "toeRR", "toeRL"]
        for j in range(p.getNumJoints(self.robot)):
            link_name = p.getJointInfo(self.robot, j)[12].decode("utf-8")
            if link_name in toe_links:
                p.changeDynamics(
                    self.robot, j,
                    lateralFriction=3,
                    spinningFriction=1,
                    rollingFriction=0.4,
                    contactStiffness=30000,
                    contactDamping=2000
                )'''

        self.joint_ids = [j for j in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]

        self.link_name_map = {j: p.getJointInfo(self.robot, j)[12].decode('utf-8') for j in range(p.getNumJoints(self.robot))}
        self.link_name_map[-1] = "base"

        #for i, joint_id in enumerate(self.joint_ids):
          #  p.resetJointState(self.robot, joint_id, self.initial_joint_angles_rad[i])

        # Contadores
        self.step_counter = 0
        self.last_dist = None

        # Obtener obs inicial
        obs33 = self._get_obs33()
        delta = self._get_delta(obs33)
        self.last_dist = np.linalg.norm(delta)
        p.stepSimulation()
        return np.concatenate([obs33, delta]), {}

    def _get_obs33(self):
        # Igual que antes: joint_pos (12), joint_vel (12), torso lin(3), ang(3), euler(3)
        joint_states = p.getJointStates(self.robot, list(range(12)))
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])
        lin_vel, ang_vel = p.getBaseVelocity(self.robot,
                                             physicsClientId=self.physics_client)
        orn = p.getBasePositionAndOrientation(self.robot)[1]
        euler = p.getEulerFromQuaternion(orn)
        return np.concatenate([joint_pos, joint_vel, lin_vel, ang_vel, euler])

    def _get_delta(self, obs33):
        # Posición actual del torso
        pos = p.getBasePositionAndOrientation(self.robot,
                                              physicsClientId=self.physics_client)[0]
        dx = self.goal_pos[0] - pos[0]
        dy = self.goal_pos[1] - pos[1]
        return np.array([dx, dy], dtype=np.float32)

    def step(self, action):
        # Aplica acción y avanza la simulación
        max_force = 70
        
         # offsets de la pose neutra (deg→rad)
        neutral = np.radians([0, 0, -45]*4)
        span_pos = np.array([math.radians(20),
                      math.radians(57),
                      math.radians(65)])     # rango simétrico aprox
        for i, j in enumerate(self.joint_ids):
            jt = i % 3
            scaled_action = neutral[i] + action[i]*span_pos[jt]
             
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition=scaled_action,
                force=max_force,
            )
        p.stepSimulation()
        self.step_counter += 1

        # Observaciones
        obs33 = self._get_obs33()
        delta = self._get_delta(obs33)
        obs = np.concatenate([obs33, delta])

        # Recompensa y done
        reward, done = self._compute_reward(obs33, delta)
        return obs, reward, done, False, {}

    def _compute_reward(self, obs33, delta):
        # Distancia actual y previa
        dist = np.linalg.norm(delta)
        # Término principal: reducción de distancia
        r_dist = (self.last_dist - dist) if (self.last_dist is not None) else 0.0
        self.last_dist = dist

        # Bonus de éxito
        r_goal = 0.0
        if dist < self.epsilon:
            r_goal = +10.0
        # Pequeña penalización si sobrepasa demasiado (+umbral)
        if dist > self.goal_range + 0.5:
            r_goal -= 5.0

        # Otros términos de estabilidad/energía opcionales
        roll, pitch, yaw = obs33[30:33]
        joint_velocities = obs33[12:24]
        torso_vel_lin = obs33[24:27]
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        z_pos = torso_pos[2]
        z0 = 0.42   # altura nominal
        vel_x = torso_vel_lin[0]

        # avance +X
        reward_speed = 5.0 * vel_x    
            # penalizacion por altura menor a 0.42
        reward_height = -20.0 * abs(z_pos - z0)
            # refuerzo por mantenerse vivo
        reward_time=2
            #penalizacion por rotar demasiado 
        reward_stability = -4.0 * (abs(roll)+abs(pitch)+abs(yaw))
            #penalizacion por velocidades demasiado altas en las piernas
        reward_energy = -1e-3 * np.sum(np.square(joint_velocities))

        reward = r_dist + r_goal+reward_speed +reward_time+ reward_stability + reward_energy+reward_height
        done = (dist < self.epsilon) or (self.step_counter > 5000) or z_pos < 0.20 or abs(roll)>0.7 or abs(pitch)>0.7
        return reward, done

    def close(self):
        p.disconnect(self.physics_client)
