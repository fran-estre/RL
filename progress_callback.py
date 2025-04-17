from stable_baselines3.common.callbacks import BaseCallback

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.verbose:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            print(f"[Progreso] Paso: {self.num_timesteps} / {self.total_timesteps} ({progress:.2f}%)", end='\r')
        return True