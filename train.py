from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import QuadrupedEnv
from stable_baselines3.common.callbacks import EvalCallback
from progress_callback import ProgressBarCallback

# Crear entorno vectorizado (opcional para paralelismo)
env = make_vec_env(QuadrupedEnv, n_envs=1, wrapper_class=None)  # Correcto 

# Configurar el modelo PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=4096,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.97,
    ent_coef=0.2,
    n_epochs=10, 
    device="auto"  # Usar GPU si está disponible
)

total_steps = 10_000_000
progress_callback = ProgressBarCallback(total_timesteps=total_steps, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100000,
                             deterministic=True, render=False)
# Entrenar
try:
    model.learn(total_timesteps=total_steps,callback=[progress_callback, eval_callback])
    env.env_method("export_rewards", filename="reward_log.csv")
except KeyboardInterrupt:
    env.env_method("export_rewards", filename="reward_log.csv")
    print("\n🛑 Entrenamiento detenido manualmente. Guardando modelo actual...")
    model.save("quadruped_ppo_interrupt")
    print("✅ Modelo guardado como 'quadruped_ppo_interrupt.zip'")
model.save("quadruped_ppo1")

# Cerrar entorno
env.close()