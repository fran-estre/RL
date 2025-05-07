# deploylaika.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploylaika.py

Evaluación visual (con cámara siguiendo al robot) y cuantitativa
de un agente PPO entrenado para Laikago.
"""
import os, time
import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from env_flat import QuadrupedEnv
#from env import QuadrupedEnv
def make_env():
    # Forzamos head-up: render_mode="human"
    return QuadrupedEnv(render_mode="human")

def rollout(model, env, n_episodes=10):
    rewards = []
    for ep in range(1, n_episodes+1):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        step   = 0

        # Centra la cámara al inicio
        env.render()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            total_r += r
            step   += 1

            # render reposiciona la cámara en tu env.render():
            env.render()
            # pausa para ver ~real time
            time.sleep(getattr(env, "time_step", 3/240.))

        print(f"Episodio {ep:2d}: pasos={step:4d}, recompensa={total_r:7.2f}")
        rewards.append(total_r)
    return rewards

def main():
    model_path = "laikago_ppo_flat.zip"
    assert os.path.isfile(model_path), f"No encuentro '{model_path}'"
    model = PPO.load(model_path, device="cpu")
    print(f"  Modelo cargado desde '{model_path}'\n")

    env = make_env()
    # Ejecuta 10 episodios manualmente, con cámara que sigue
    rewards = rollout(model, env, n_episodes=10)
    env.close()

    r = np.array(rewards)
    print(f"\nRecompensa media  : {r.mean():.2f}")
    print(f"Desvío estándar : {r.std():.2f}")

if __name__ == "__main__":
    main()
