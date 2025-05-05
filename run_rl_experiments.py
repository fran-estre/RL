#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rl_experiments.py

Ejecuta simulaciones con el agente PPO entrenado sobre distintos terrenos
y registra las mismas métricas para compararlas con MPC.
"""

import os
import csv
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from env_flat  import QuadrupedEnv as EnvFlat
from env_rough import QuadrupedEnv as EnvRough

def collect_metrics(model: PPO, env: p, max_steps: int):
    """Ejecuta un episodio y devuelve listas de time, torque, energy, speed."""
    obs, _ = env.reset()
    t_data, torque_data, energy_data, speed_data = [], [], [], []
    # identificadores de juntas por pierna
    joints_per_leg = [list(range(0,3)), list(range(3,6)),
                      list(range(6,9)), list(range(9,12))]
    for _ in range(max_steps):
        # acción determinista
        action, _ = model.predict(obs, deterministic=True)
        # aplica acción
        obs, reward, done, _, _ = env.step(action)
        # paso de simulación interno
        # tiempo
        t = env.step_counter * getattr(env, 'time_step', 1/240)
        t_data.append(t)
        # torques: p.getJointStates devuelve appliedJointMotorTorque en índice 3
        js = p.getJointStates(env.robot, env.joint_ids, physicsClientId=env.physics_client)
        torques = np.array([state[3] for state in js])
        torque_data.append(torques.tolist())
        # energía por pierna
        energies = [np.sum(torques[idx]**2) for idx in joints_per_leg]
        energy_data.append(energies)
        # velocidad
        lin_vel, _ = p.getBaseVelocity(env.robot, physicsClientId=env.physics_client)
        speed_data.append(lin_vel[0])
        if done:
            break
    return {
        'time': t_data,
        'torque': torque_data,
        'energy_legs': energy_data,
        'speed': speed_data
    }

def save_csv(data: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['time'] \
               + [f'torque_{i}' for i in range(12)] \
               + [f'energy_leg{j}' for j in range(4)] \
               + ['speed']
        writer.writerow(header)
        for t, torques, energies, v in zip(
                data['time'],
                data['torque'],
                data['energy_legs'],
                data['speed']):
            writer.writerow([t, *torques, *energies, v])
    print(f"[RL] Resultados guardados en {filename}")

def main():
    experiments = [
        ('flat',   EnvFlat,  'laikago_ppo_angles_flat.zip', 1000),
        ('rough',  EnvRough, 'laikago_ppo_angles_parallel.zip',      1000),
    ]
    for name, EnvClass, model_file, max_steps in experiments:
        # carga modelo PPO
        assert os.path.isfile(model_file), f"No existe {model_file}"
        model = PPO.load(model_file, device="cpu")
        # crea entorno headless
        env = EnvClass(render_mode=None)
        # recaba métricas
        metrics = collect_metrics(model, env, max_steps)
        # vuelca a CSV
        save_csv(metrics, filename=f"results/rl_{name}.csv")
        env.close()

if __name__ == "__main__":
    main()
