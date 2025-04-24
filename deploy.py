from stable_baselines3 import DDPG
from env import QuadrupedEnv

# Cargar modelo
model = DDPG.load("D:\\ITMO trabajos de la u\\tesis\\py\\testing\\RL\\quadruped_ddpg1.zip")

# Evaluar
env = QuadrupedEnv()
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
