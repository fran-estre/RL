from stable_baselines3 import PPO
from env import QuadrupedEnv

# Crear el entorno
env = QuadrupedEnv()

# Cargar el modelo
model = PPO.load("quadruped_ppo_interrupt.zip", env=env)

# Continuar entrenamiento
model.learn(total_timesteps=500_000)  # suma a lo anterior
model.save("quadruped_ppo_continued")
