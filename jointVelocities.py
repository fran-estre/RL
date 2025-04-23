import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo de velocidades articulares
log_file = "joint_velocities_log.csv"
df = pd.read_csv(log_file)

# Graficar velocidades de cada articulación
plt.figure(figsize=(12, 6))
for joint in df.columns:
    plt.plot(df[joint], label=joint)

plt.title("Velocidades articulares durante la simulación")
plt.xlabel("Pasos de registro (cada 10 pasos de simulación)")
plt.ylabel("Velocidad angular (rad/s)")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.tight_layout()
plt.show()