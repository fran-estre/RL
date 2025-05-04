import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import math
import os

class QuadrupedEnv(gym.Env):      
    def __init__(self, render_mode=None):
        super().__init__()
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.physics_client = p.connect(mode)
        self.cam_distance = 1.5
        self.cam_yaw      = 50
        self.cam_pitch    = -30
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (33,), np.float32)
        self.robot = None
        self.joint_ids = None

        self.initial_joint_angles_deg = np.array([0, 0, -45, 0.0, 0,-45, 0.0, 0, -45, 0.0, 0, -45])
        self.initial_joint_angles_rad = np.radians(self.initial_joint_angles_deg)
        self.last_action = None
        self.initial_roll = 0
        self.initial_pitch = 0
        self.step_counter = 0
        self.reset()

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

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.plane = self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago/laikago_toes_zup.urdf", 
                     [0, 0, 0.55],[0, 0, 0, 1],useFixedBase=False)

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

        obs = self._get_obs()
        self.step_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        p.stepSimulation()
            
        return obs, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_ids)
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])

        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        torso_euler = p.getEulerFromQuaternion(torso_orn)
        torso_vel_lin, torso_vel_ang = p.getBaseVelocity(self.robot)
        
        return np.concatenate([joint_pos,#12
                         joint_vel,#12
                         torso_vel_lin, #3       # vx,vy,vz
                         torso_vel_ang, #3       # wx,wy,wz
                         torso_euler])    #3     # roll,pitch,yaw

    def step(self, action):
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
        obs = self._get_obs()
        reward= self._compute_reward(obs,action)
        done = self._check_done(obs)

        return obs, reward, done, False, {}
  
    def _compute_reward(self, obs, action):
            roll, pitch, yaw = obs[30:33]
            joint_velocities = obs[12:24]
            torso_vel_lin = obs[24:27]
            torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
            z_pos = torso_pos[2]
            z0 = 0.42   # altura nominal
            vel_x = torso_vel_lin[0]

            delta = action - self.last_action
            reward_action = -0.1 * np.sum(delta**2)
            # avance +X
            reward_speed = 5.0 * vel_x    
            # penalizacion por altura menor a 0.42
            reward_height = -20.0 * abs(z_pos - z0)
            # refuerzo por mantenerse vivo
            reward_time=1
            #penalizacion por rotar demasiado 
            reward_stability = -2.0 * (abs(roll)+abs(pitch)+abs(yaw))
            #penalizacion por velocidades demasiado altas en las piernas
            reward_energy = -1e-3 * np.sum(np.square(joint_velocities))

            total_reward = reward_speed +reward_time+ reward_stability + reward_energy+reward_height+reward_action
            return total_reward

    def _check_done(self, obs):
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        z_pos = torso_pos[2]  
        roll, pitch, _ = obs[30:33]
        
        
        fallen = z_pos < 0.20 or abs(roll)>0.7 or abs(pitch)>0.7 
        return fallen
  
    def close(self):
        p.disconnect(self.physics_client)

      