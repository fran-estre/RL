#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_multi_waypoints.py

Despliega al agente PPO para que recorra una lista de waypoints secuenciales.
"""

import os, time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from env_goal import QuadrupedEnv   # tu entorno goal-conditioned

def rollout_to_goal(model, env, goal, max_steps=2000):
    """Ejecuta un episodio hacia un único goal. Retorna recompensa acumulada."""
    obs, _ = env.reset(options={'goal_pos': goal})
    total_r = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        total_r += r
        env.render()
        time.sleep(getattr(env, 'time_step', 1/240.))
        if done:
            break
    return total_r

def main():
    # 1) Carga del modelo entrenado
    model_file = "laikago_goal.zip"
    assert os.path.isfile(model_file), f"No encuentro {model_file}"
    model = PPO.load(model_file, device="cpu")

    # 2) Lista de waypoints [(x1,y1), (x2,y2), ...]
    waypoints = [
        ( 1.0,  0.0),
        ( 0.5,  0.5),
        ( 0.0,  1.0),
        (-0.5,  0.5),
        (-1.0,  0.0),
    ]

    # 3) Crear entorno con GUI
    env = QuadrupedEnv(render_mode="human",
                       goal_range=2.0,
                       epsilon=0.1)

    # 4) Recorrer waypoints
    for idx, wp in enumerate(waypoints, start=1):
        print(f"\n=== Waypoint {idx}: {wp} ===")
        reward = rollout_to_goal(model, env, wp)
        print(f"Recompensa total para waypoint {idx}: {reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
