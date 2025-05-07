#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploylaika_metrics.py

Evalúa cuantitativamente un agente PPO para Laikago,
extrayendo recompensas, longitud de episodios, tasa de éxito
y real‐time factor.
"""
import os, time
import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from env_flat import QuadrupedEnv

def make_env():
    return QuadrupedEnv(render_mode="human")

def rollout(model, env, n_episodes=5, success_threshold=None):
    rewards = []
    steps   = []
    rtfs    = []  # real-time factor por episodio

    max_steps = getattr(env, "max_episode_steps", None)
    successes = 0

    for ep in range(1, n_episodes+1):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        step   = 0

        start_wall = time.time()
        # simulación no tiene tiempo acumulado por defecto, lo calculamos:
        dt = getattr(env, "time_step", 3/240.)

        env.render()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            total_r += r
            step   += 1

            env.render()
            time.sleep(dt)

        elapsed_wall = time.time() - start_wall
        elapsed_sim  = step * dt
        rtf = elapsed_sim / elapsed_wall

        # criterio de éxito: si cumple todo el episodio sin caer:
        if max_steps is not None:
            is_success = (step >= max_steps)
        elif success_threshold is not None:
            is_success = (total_r >= success_threshold)
        else:
            is_success = True  # si no definimos nada, asumimos éxito
        successes += int(is_success)

        print(f"Episodio {ep:2d}: pasos={step:4d}, recompensa={total_r:7.2f}, RTF={rtf:.2f}, éxito={is_success}")
        rewards.append(total_r)
        steps.append(step)
        rtfs.append(rtf)

    # convertir a arrays para estadística
    rewards = np.array(rewards)
    steps   = np.array(steps)
    rtfs    = np.array(rtfs)

    # métricas de recompensa
    print("\n== Recompensa ==")
    print(f"  Media       : {rewards.mean():.2f}")
    print(f"  Mediana     : {np.median(rewards):.2f}")
    print(f"  Desv. std.  : {rewards.std():.2f}")
    print(f"  Mínimo      : {rewards.min():.2f}")
    print(f"  Máximo      : {rewards.max():.2f}")
    print(f"  Percentiles : 25%={np.percentile(rewards,25):.2f}, 75%={np.percentile(rewards,75):.2f}")

    # métricas de longitud de episodio
    print("\n== Longitud de episodio (pasos) ==")
    print(f"  Media       : {steps.mean():.1f}")
    print(f"  Mediana     : {np.median(steps):.1f}")
    print(f"  Desv. std.  : {steps.std():.1f}")
    print(f"  Mínimo      : {steps.min()}")
    print(f"  Máximo      : {steps.max()}")

    # tasa de éxito
    éxito_pct = successes / len(rewards) * 100
    print(f"\n== Éxitos ==\n  {successes}/{n_episodes} ({éxito_pct:.1f} %)")

    # real-time factor
    print("\n== Real‐Time Factor (RTF) ==")
    print(f"  Media       : {rtfs.mean():.2f}")
    print(f"  Desv. std.  : {rtfs.std():.2f}")

    return {
        "rewards": rewards,
        "steps": steps,
        "rtf": rtfs,
        "success_rate": éxito_pct,
    }

def main():
    model_path = "laikago_ppo_flat_parallel.zip"
    assert os.path.isfile(model_path), f"No encuentro '{model_path}'"
    model = PPO.load(model_path, device="cpu")
    print(f"Modelo cargado desde '{model_path}'\n")

    env = make_env()
    stats = rollout(model, env, n_episodes=1, success_threshold=None)
    env.close()

if __name__ == "__main__":
    main()
