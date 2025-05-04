import gymnasium as gym
from stable_baselines3 import PPO      # SAC o TD3 también funcionan
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_rough import QuadrupedEnv
import os

class StepPrinter(BaseCallback):
    def _on_step(self) -> bool:
        print("Timestep =", self.num_timesteps)       # global SB3
        local_steps = self.training_env.get_attr("step_counter")[0]
        
        print("Step env[0] =", local_steps)
        return True

def make_env():
    # Devuelve una función que instanciará el entorno
    def _init():
        return QuadrupedEnv()
    return _init
if __name__ == "__main__":
    env = QuadrupedEnv(render_mode="human")      # ← usa p.GUI

    model = PPO("MlpPolicy", env,
                n_steps=2048,
                batch_size=512,
                verbose=1)

    model.learn(total_timesteps=1_000_000,callback=StepPrinter()) 
        
    model.save("laikago_ppo_angles")
    


