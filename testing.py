#!/usr/bin/env python3
"""
visualize_laika_realtime.py

Abre la GUI de PyBullet y durante `duration` segundos
muestra en tiempo real (consola + overlay) los ángulos
roll, pitch, yaw de Laika en cada paso de simulación.
"""

import time
import numpy as np
import pybullet as p
from env import QuadrupedEnv

def get_pybullet_client(env):
    # Intenta deducir el ID del cliente de PyBullet
    for attr in ("_physics_client_id", "_pybullet_client", "client"):
        if hasattr(env, attr):
            return getattr(env, attr)
    return None

def visualize_realtime(duration: float = 5.0):
    # 1) Inicializa entorno en modo GUI
    env = QuadrupedEnv(render_mode="human")
    obs, _ = env.reset()

    # 2) Obtén time step y número de iteraciones
    dt      = getattr(env, "time_step", 1.0 / 240.0)
    n_steps = int(round(duration / dt))

    # 3) Prepara acción neutra y cliente
    zero_action = np.zeros(env.action_space.shape)
    client_id   = get_pybullet_client(env)

    print(" t (s)  |   roll (°)   pitch (°)    yaw (°)")
    print("--------+--------------------------------")
    t = 0.0
    for i in range(n_steps):
        obs, *_ = env.step(zero_action)
        # Extrae y convierte a grados
        roll, pitch, yaw = np.degrees(obs[30:33])

        # 4a) Consola
        print(f"{t:6.2f}  |  {roll:8.4f}  {pitch:10.4f}  {yaw:9.4f}", flush=True)

        # 4b) Overlay en la GUI (vida corta = dt)
        if client_id is not None:
            text = f"r {roll:6.2f}°, p {pitch:6.2f}°, y {yaw:6.2f}°"
            # Pinta cerca del origen (ajusta la posición [x,y,z] si quieres)
            p.addUserDebugText(
                text,
                textPosition=[0, 0, 1.2],
                textSize=1.2,
                lifeTime=dt,
                physicsClientId=client_id
            )

        # 5) Sincroniza a tiempo real
        time.sleep(dt)
        t += dt

    print("\nSimulación completada. Pulsa Ctrl+C o cierra la ventana para salir.")
    # No cerramos env automáticamente para que puedas inspeccionar libremente:
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    visualize_realtime(duration=5.0)
