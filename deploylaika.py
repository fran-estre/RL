#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluación cuantitativa y visual de un agente PPO entrenado
para Laikago en el entorno QuadrupedEnv, con cámara que sigue al robot.
"""

import os
import time
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env import QuadrupedEnv

def get_pybullet_client(env):
    """Recupera el ID del cliente de PyBullet desde el env."""
    # Tu env almacena el cliente en env.physics_client :contentReference[oaicite:1]{index=1}
    return getattr(env, "physics_client", None)

def follow_camera(client_id, target_pos,
                  distance=1.5, yaw=50, pitch=-30, up_axis_index=2):
    """
    Ajusta la cámara de PyBullet para que siga `target_pos`.
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target_pos,
        physicsClientId=client_id
    )

def make_eval_env(render_mode="human"):
    return QuadrupedEnv(render_mode=render_mode)

def main():
    model_path = "laikago_ppo_angles.zip"
    assert os.path.isfile(model_path), f"No encuentro {model_path}"
    model = PPO.load(model_path, device="cpu")
    print(f"Modelo cargado desde '{model_path}'")

    eval_env = make_eval_env(render_mode="human")
    client_id = get_pybullet_client(eval_env)

    # --- evaluación automática (sin cámara dinámica) ---
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        render=True,
        deterministic=True,
        warn=False
    )
    print(f"\n=== Resultado evaluate_policy ===")
    print(f"Recompensa media: {mean_reward:.2f} ± {std_reward:.2f}\n")

    # --- evaluación manual con cámara siguiendo al robot ---
    n_manual = 3
    print(f"=== Evaluación manual de {n_manual} episodios ===")
    for ep in range(1, n_manual+1):
        obs, _ = eval_env.reset()

        # Centrar cámara justo al reset
        base_pos, _ = p.getBasePositionAndOrientation(
            eval_env.robot, physicsClientId=client_id)
        follow_camera(client_id, base_pos)

        done = False
        total_reward = 0.0
        step = 0
        dt = getattr(eval_env, "time_step", 1/240.)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward
            step += 1

            # Actualizar cámara cada paso
            base_pos, _ = p.getBasePositionAndOrientation(
                eval_env.robot, physicsClientId=client_id)
            follow_camera(client_id, base_pos)

            time.sleep(dt)

        print(f"Episodio {ep}: pasos={step}, recompensa={total_reward:.2f}")

    eval_env.close()
    print("\nEvaluación finalizada.")

if __name__ == "__main__":
    main()
