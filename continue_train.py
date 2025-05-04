#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continue_training.py

Continúa el entrenamiento de un agente PPO previamente entrenado
para Laikago en QuadrupedEnv.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_flat import QuadrupedEnv
#from env import QuadrupedEnv
def make_env(rank, seed=0):
    def _init():
        env = QuadrupedEnv(render_mode=None)
        
        return env
    return _init

if __name__ == "__main__":
    # 1) Carga tu modelo
    model = PPO.load("laikago_ppo_angles.zip")
    print("Modelo cargado. timesteps =", model.num_timesteps)

    # 2) Crea un VecEnv con 8 procesos
    num_envs = 4
    envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # 3) Asocia el nuevo vec_env
    model.set_env(envs)

    # 4) Continúa entrenando (por ejemplo 2e6 pasos más)
    model.learn(
        total_timesteps=1000000,
        reset_num_timesteps=False,
        tb_log_name="PPO_flat_angles_continued"
    )

    # 5) Guarda
    model.save("laikago_ppo_angles_flat.zip")
    envs.close()

