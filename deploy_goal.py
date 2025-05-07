import os
import time
import numpy as np
import pybullet as p
import csv
from stable_baselines3 import PPO
from env_goal_rough import QuadrupedEnv  # tu entorno goal-conditioned

def rollout_to_goal(model, env, goal, max_steps=2000, csv_file="deviation_x.csv"):
    """Ejecuta un episodio hacia un único goal. Registra desviación en eje X."""
    obs, _ = env.reset(options={'goal_pos': goal})
    total_r = 0.0

    with open(csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Step", "Deviation_X"])

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            total_r += r
            env.render()
            time.sleep(getattr(env, 'time_step', 1/240.))

            # Desviación en eje Y: goal_y – robot_y
            deviation_y = env.goal_pos[1] - \
                          p.getBasePositionAndOrientation(env.robot)[0][1]
            csv_writer.writerow([step, deviation_y])

            if done:
                break

    return total_r


def main():
    # 1) Carga del modelo entrenado
    model_file = "laikago_goal.zip"
    assert os.path.isfile(model_file), f"No encuentro {model_file}"
    model = PPO.load(model_file)

    # 2) Lista de waypoints [(x1,y1), (x2,y2), ...]
    waypoints = [
        (5, 0.0)
    ]

    # 3) Crear entorno con GUI
    env = QuadrupedEnv(render_mode="human",
                       goal_range=2.0,
                       epsilon=0.1)

    # 4) Recorrer waypoints y registrar desviación en CSV
    for idx, wp in enumerate(waypoints, start=1):
        print(f"\n=== Waypoint {idx}: {wp} ===")
        reward = rollout_to_goal(model, env, wp)
        print(f"Recompensa total para waypoint {idx}: {reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
