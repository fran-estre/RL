import matplotlib.pyplot as plt
import csv

steps = []
reward_movement = []
forward_progress = []

with open("reward_log.csv", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        reward_movement.append(float(row["reward_movement"]))
        forward_progress.append(float(row["forward_progress"]))

plt.figure(figsize=(12, 6))

plt.plot(steps, reward_movement, label="Reward Movement", linewidth=2)
plt.plot(steps, forward_progress, label="Forward Progress", linewidth=2, linestyle='--')

plt.xlabel("Simulation Step")
plt.ylabel("Value")
plt.title("Evolución del progreso y reward de movimiento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
