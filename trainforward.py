from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from env_2 import QuadrupedEnv
from stable_baselines3.common.callbacks import EvalCallback
from progress_callback import ProgressBarCallback

# Crear entorno vectorizado (opcional para paralelismo)
env = make_vec_env(QuadrupedEnv, n_envs=1, wrapper_class=None)

# Configurar el modelo DDPG
model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    buffer_size=1000000,
    batch_size=128,
    gamma=0.99,
    tau=0.005,
    device="auto"
)

total_steps = 10_000_000
progress_callback = ProgressBarCallback(total_timesteps=total_steps, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100000,
                             deterministic=True, render=False)

# Entrenar
try:
    model.learn(total_timesteps=total_steps, callback=[progress_callback, eval_callback])
    env.env_method("export_rewards", filename="reward_log.csv")
except KeyboardInterrupt:
    env.env_method("export_rewards", filename="reward_log.csv")
    print("\n🛑 Entrenamiento detenido manualmente. Guardando modelo actual...")
    model.save("quadruped_ddpg_interrupt")
    print("✅ Modelo guardado como 'quadruped_ddpg_interrupt.zip'")
model.save("quadruped_ddpg1")

# Cerrar entorno
env.close()
