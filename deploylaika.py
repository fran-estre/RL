#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluación cuantitativa y visual de un agente PPO entrenado
para Laikago en el entorno QuadrupedEnv.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Si quieres grabar vídeo, descomenta estas líneas:
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

from env import QuadrupedEnv

def make_eval_env(render_mode="human"):
    """
    Crea una instancia del entorno para evaluación.
    Si render_mode="rgb_array" o None, no se abre GUI.
    """
    return QuadrupedEnv(render_mode=render_mode)

def main():
    # 1) Ruta al modelo entrenado
    model_path = "laikago_ppo_angles.zip"
    assert os.path.isfile(model_path), f"No encuentro {model_path}"
    
    # 2) Carga del modelo
    model = PPO.load(model_path, device="cpu")
    print(f"Modelo cargado desde '{model_path}'")

    # 3) Entorno para evaluate_policy (sin vectorización)
    eval_env = make_eval_env(render_mode="human")
    
    # 4) Evaluación automática (10 episodios)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        render=True,        # True para ver GUI en cada paso
        deterministic=True, # usa la acción más probable
        warn=False
    )
    print(f"\n=== Resultado evaluate_policy ===")
    print(f"Recompensa media: {mean_reward:.2f} ± {std_reward:.2f}\n")

    # 5) Evaluación manual con detalles por episodio
    n_manual = 3
    print(f"=== Evaluación manual de {n_manual} episodios ===")
    for ep in range(1, n_manual+1):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            total_reward += reward
            step += 1
        print(f"Episodio {ep}: pasos={step}, recompensa={total_reward:.2f}")
    
    # 6) Cierre
    eval_env.close()
    print("\nEvaluación finalizada.")

if __name__ == "__main__":
    main()
