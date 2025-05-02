import gymnasium as gym
from stable_baselines3 import PPO      # SAC o TD3 también funcionan
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import QuadrupedEnv
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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # opcional, si usas Intel MKL
    num_envs = 4
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)],
                            start_method="spawn")   # explícito, aunque spawn es el default en Win
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=2_000_000,callback=StepPrinter())
        model.save("laikago_ppo_angles")
    
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido manualmente. Guardando modelo actual...")
        model.save("laikago_ppo_angles_interrupt")
        print("✅ Modelo guardado como 'quadruped_ddpg_interrupt.zip'")
    model.save("quadruped_ddpg1")


