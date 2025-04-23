def _compute_reward(self, obs):
        torso_pos = obs[24:27]
        roll, pitch, _ = obs[27:30]
        direction_to_target = obs[30:32]
        joint_velocities = obs[12:24]
        torso_vel_lin = obs[30:32]

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

        reward_movement = reward_progress + reward_heading + reward_velocity + penalty_away

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
            "reward_movement": reward_movement,
            "reward_height": reward_height,
            "reward_stability": reward_stability,
            "reward_energy": reward_energy,
            "reward_goal": reward_goal,
            "reward_alive": reward_alive
        }
        total_reward = sum(components.values())
        return total_reward, components