import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
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

def make_env(render_mode, log_dir, rank):
    """
    Devuelve un constructor de entorno que:
     - usa GUI si render_mode="human", DIRECT si None
     - envuelve el entorno con Monitor para loguear en log_dir/env_{rank}/monitor.csv
    """
    def _init():
        env = QuadrupedEnv(render_mode=render_mode)
        env = Monitor(
            env,
            filename=os.path.join(log_dir, f"env_{rank}", "monitor.csv"),
            allow_early_resets=True
        )
        return env
    return _init

if __name__ == "__main__":
    # 1) Crear directorio de logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 2) Número de entornos paralelos
    num_envs = 8

    # 3) Construir las fábricas de entornos (todos headless)
    env_fns = []
    for i in range(num_envs):
        env_fns.append(make_env(render_mode=None, log_dir=log_dir, rank=i))

    # 4) Vectorizar
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # 5) Crear el modelo PPO (sin tensorboard)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
    )

    try:
        model.learn(
            total_timesteps=2_000_000
        )   
        print("simulacion terminada")
        model.save("laikago_ppo_angles")
        vec_env.close()
        print("Entrenamiento completado. Logs en 'logs/env_*'/monitor.csv")
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido manualmente. Guardando modelo actual...")
        model.save("laikago_ppo_angles_interrupt")
        print("✅ Modelo guardado como 'quadruped_ddpg_interrupt.zip'")
        vec_env.close()
        print("Entrenamiento completado. Logs en 'logs/env_*'/monitor.csv")

    model.save("quadruped_ddpg1")


