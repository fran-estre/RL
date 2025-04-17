import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import QuadrupedEnv  # Asegúrate de que tu entorno esté correctamente exportado

NUM_EPISODES = 10
MAX_STEPS = 1000

# Cargar modelo
model = PPO.load("quadruped_ppo")

# Crear entorno (NO vectorizado aquí)
env = QuadrupedEnv()

episode_rewards = []
reward_components_all = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Para guardar componentes individuales
    reward_components = {
        "reward_progress": [],
        "reward_height": [],
        "reward_stability": [],
        "reward_energy": [],
        "reward_goal": [],
        "reward_alive": []
    }

    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step += 1

        for key in reward_components:
            reward_components[key].append(info["reward_breakdown"].get(key, 0.0))

    episode_rewards.append(total_reward)
    reward_components_all.append(reward_components)

    print(f"✅ Episodio {episode + 1}: Recompensa total = {total_reward:.2f}, pasos = {step}")

env.close()

# ---- GRAFICAR ----
# Promedio por componente a lo largo del tiempo (por episodio)
avg_components_per_ep = {key: [] for key in reward_components_all[0].keys()}

for ep in reward_components_all:
    for key in ep:
        avg = np.mean(ep[key])
        avg_components_per_ep[key].append(avg)

# Gráfica 1: Recompensa total por episodio
plt.figure()
plt.plot(episode_rewards, marker='o')
plt.title("Recompensa total por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.grid()

# Gráfica 2: Promedio de componentes de recompensa por episodio
plt.figure()
for key, values in avg_components_per_ep.items():
    plt.plot(values, label=key)
plt.title("Promedio de componentes de recompensa por episodio")
plt.xlabel("Episodio")
plt.ylabel("Valor promedio")
plt.legend()
plt.grid()

plt.show()
