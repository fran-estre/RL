import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import csv
import os

class QuadrupedEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.controlled_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.controlled_joints),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self.locked_joints = [0, 3, 6, 9]  # Hip joints
        self.joint_damping = {
            0: 50, 1: 8, 2: 12,   # Pierna frontal derecha
            3: 50, 4: 8, 5: 12,   # Pierna frontal izquierda
            6: 50, 7: 8, 8: 12,   # Pierna trasera derecha
            9: 50, 10: 8, 11: 12  # Pierna trasera izquierda
        }

        self.robot = None
        self.target_pos = None
        self.target_marker = None
        self.joint_ids = None
        self.prev_pos = None

        self.initial_joint_angles_deg = np.array([0, 0, -45, 0.0, 0,-45, 0.0, 0, -45, 0.0, 0, -45])
        self.initial_joint_angles_rad = np.radians(self.initial_joint_angles_deg)

        self.initial_roll = 0
        self.initial_pitch = 0
        self.reward_log = []
        self.step_counter = 0
        self.log_file = "joint_velocities_log.csv"
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"joint_{i}" for i in range(12)])
        
        self.contact_log_file = "contact_info_log.csv"
        with open(self.contact_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "link_index", "link_name", "normal_force", "lateral_friction"])

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago\\laikago_toes.urdf", [0, 0, 0.5], [0, 0.5, 0.5, 0], useFixedBase=False)

        p.changeDynamics(self.plane, -1, lateralFriction=1, spinningFriction=0.5, rollingFriction=0.1)

        lower_legs = ["FR_lower_leg", "FL_lower_leg", "RR_lower_leg", "RL_lower_leg"]

        for j in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, j)
            link_name = joint_info[12].decode("utf-8")
            if link_name in lower_legs:
                p.setCollisionFilterGroupMask(self.robot, j, 0, 0)
        
        # Ajustar fricción y dinámica de contacto para los toes
        toe_links = ["toeFR", "toeFL", "toeRR", "toeRL"]
        for j in range(p.getNumJoints(self.robot)):
            link_name = p.getJointInfo(self.robot, j)[12].decode("utf-8")
            if link_name in toe_links:
                p.changeDynamics(
                    self.robot, j,
                    lateralFriction=3,
                    spinningFriction=0.5,
                    rollingFriction=0.1,
                    contactStiffness=30000,
                    contactDamping=2000
                )


        self.joint_ids = [j for j in range(p.getNumJoints(self.robot)) if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]

        self.link_name_map = {j: p.getJointInfo(self.robot, j)[12].decode('utf-8') for j in range(p.getNumJoints(self.robot))}
        self.link_name_map[-1] = "base"

        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot, joint_id, self.initial_joint_angles_rad[i])

        self.target_pos = [0, 5, 0]
        self.target_marker = p.loadURDF("sphere_small.urdf", self.target_pos, globalScaling=2)

        self.prev_pos = np.array(p.getBasePositionAndOrientation(self.robot)[0][:2])

        obs = self._get_obs()
        self.initial_roll = obs[27]
        self.initial_pitch = obs[28]

        self.step_counter = 0

        # Bloquear articulaciones hip
        for joint_index in self.locked_joints:
            joint_id = self.joint_ids[joint_index]
            p.setJointMotorControl2(
                self.robot, joint_id,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )
        p.stepSimulation()

        return obs, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_ids)
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])

        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        torso_euler = p.getEulerFromQuaternion(torso_orn)
        torso_vel_lin, torso_vel_ang = p.getBaseVelocity(self.robot)

        direction_to_target = self.target_pos[:2] - np.array(torso_pos[:2])
        distance_to_target = np.linalg.norm(direction_to_target)

        robot_yaw = torso_euler[2]
        robot_facing = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        unit_target_dir = direction_to_target / distance_to_target if distance_to_target > 0 else np.array([1.0, 0.0])
        angle_to_target = np.arccos(np.clip(np.dot(robot_facing, unit_target_dir), -1.0, 1.0))

        return np.concatenate([joint_pos, joint_vel, torso_pos, torso_euler, torso_vel_lin, torso_vel_ang,
                               [distance_to_target], unit_target_dir, [angle_to_target]])

    def step(self, action):
        max_force = 80
        for i, joint_index in enumerate(self.controlled_joints):
            joint_id = self.joint_ids[joint_index]
            damping_gain = self.joint_damping.get(joint_index, 0.0)
            p.setJointMotorControl2(
                self.robot, joint_id,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=max_force,
                positionGain=0.3,
                velocityGain=damping_gain
            )
        # Aplicar damping pasivo a las articulaciones hip (bloqueadas)
        for joint_index in self.locked_joints:
            joint_id = self.joint_ids[joint_index]
            damping_gain = self.joint_damping.get(joint_index, 0.0)
            p.setJointMotorControl2(
                self.robot, joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=100,
                velocityGain=damping_gain
            )

        p.stepSimulation()

        self.step_counter += 1
        if self.step_counter % 10 == 0:
            joint_states = p.getJointStates(self.robot, self.joint_ids)
            joint_velocities = [round(s[1], 4) for s in joint_states]
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(joint_velocities)

        if self.step_counter % 10 == 0:
            self.log_contact_info()

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(obs)
        done = self._check_done(obs)

    
        return obs, reward, done, False, {"reward_breakdown": reward_info}
    
    def log_contact_info(self):
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.plane)
        with open(self.contact_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for contact in contacts:
                link_idx = contact[3]
                link_name = self.link_name_map.get(link_idx, f"link_{link_idx}")
                normal_force = contact[9]
                dynamics_info = p.getDynamicsInfo(self.robot, link_idx)
                lateral_friction = dynamics_info[1] if dynamics_info else None
                writer.writerow([self.step_counter, link_idx, link_name, normal_force, lateral_friction])

    def _compute_reward(self, obs):
            torso_pos = obs[24:27]
            roll, pitch, _ = obs[27:30]
            direction_to_target = obs[30:32]
            joint_velocities = obs[12:24]
            torso_vel_lin = obs[30:32]
            vel_y = torso_vel_lin[1]    # Componente en el eje Y

            reward_forward_velocity = 10.0 * vel_y
            if vel_y < 0:
                reward_forward_velocity = 10.0 * vel_y  # o podrías usar -10.0 * abs(vel_y)

            roll_error = abs(roll - self.initial_roll)
            pitch_error = abs(pitch - self.initial_pitch)
            roll_limit = np.radians(10)
            pitch_limit = np.radians(10)
            roll_penalty = max(0, roll_error - roll_limit)
            pitch_penalty = max(0, pitch_error - pitch_limit)
            reward_stability = -10.0 * (roll_penalty**2 + pitch_penalty**2)

            current_pos_xy = np.array(torso_pos[:2])
            target_vec = self.target_pos[:2] - current_pos_xy
            distance = np.linalg.norm(target_vec)
            movement_vector = current_pos_xy - self.prev_pos
            unit_direction = target_vec / np.linalg.norm(target_vec) if np.linalg.norm(target_vec) > 0 else np.array([0.0, 0.0])

            # Recompensa por avanzar hacia el objetivo
            forward_progress = np.dot(movement_vector, unit_direction)
            reward_progress = 5.0 * forward_progress

            # Recompensa por orientación hacia el objetivo
            robot_yaw = obs[29]
            robot_facing = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            unit_target_dir = unit_direction
            reward_heading = 1.0 * np.dot(robot_facing, unit_target_dir)

            # Recompensa por velocidad hacia el objetivo
            reward_velocity = 4.0 * np.dot(torso_vel_lin, unit_target_dir)

            # Penalización por alejarse
            if forward_progress < 0:
                penalty_away = -10.0 * abs(forward_progress)
            else:
                penalty_away = 0.0

            prev_distance = np.linalg.norm(self.prev_pos - self.target_pos[:2])
            distance_diff = distance - prev_distance
            reward_distance_change = -5.0 * distance_diff  # penaliza alejarse
            self.prev_pos = current_pos_xy

            reward_height = -2.0 * abs(torso_pos[2] - 0.4)
            reward_energy = -0.005 * np.sum(np.square(joint_velocities))
            reward_goal = 50.0 if distance < 0.3 else 0.0
            reward_alive = 1.0

            components = {
                "reward_distance_change":reward_distance_change,
                "reward_progress": reward_progress,
                "reward_heading": reward_heading,
                "reward_velocity": reward_velocity,
                "penalty_away": penalty_away,
                "reward_height": reward_height,
                "reward_stability": reward_stability,
                "reward_energy": reward_energy,
                "reward_goal": reward_goal,
                "reward_alive": reward_alive
            }
            total_reward = sum(components.values())
            return total_reward, components

    def _check_done(self, obs):
        z_pos = obs[26]
        roll, pitch, _ = obs[27:30]
        delta_roll = abs(roll - self.initial_roll)
        delta_pitch = abs(pitch - self.initial_pitch)
        fallen = z_pos < 0.1
        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.plane)
        chassis_contact = any(cp[3] == -1 or cp[3] == 0 for cp in contact_points)
        current_y = obs[25]
        retroceso_excesivo = current_y < -1.0
        reached_goal = np.linalg.norm(obs[24:26] - self.target_pos[:2]) < 0.3

        return fallen or chassis_contact or retroceso_excesivo or reached_goal

    def render(self, mode='human'):
        pass
    
    def close(self):
        p.disconnect(self.physics_client)