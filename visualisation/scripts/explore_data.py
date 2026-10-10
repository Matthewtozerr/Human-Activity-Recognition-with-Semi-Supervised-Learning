from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_path = Path(__file__).parent.parent.parent / "data" / "RawData"
output_path = Path(__file__).parent.parent / "outputs" 
output_path.mkdir(parents=True, exist_ok=True)

fs = 50 #Hz

experiment_id = 4
user_id = 2

acc_file = data_path / f'acc_exp{experiment_id:02d}_user{user_id:02d}.txt' #accelerometer (linear)
gyro_file = data_path / f'gyro_exp{experiment_id:02d}_user{user_id:02d}.txt' #gyroscope (angular)

acc = np.loadtxt(acc_file)
print(f"Accelerometer data shape: {acc.shape}")
gyro = np.loadtxt(gyro_file)
print(f"Gyroscope data shape: {gyro.shape}")

n_samples = len(acc) # = len(gyro)

time = np.arange(n_samples) / fs
print(f"Recording duration: {time[-1]:.2f} seconds")

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(14, 8),
    sharex=True
)

axis_names = ["X", "Y", "Z"]

for i in range(3):
    axes[0].plot(time, acc[:, i], label=axis_names[i])
    axes[1].plot(time, gyro[:, i], label=axis_names[i])

axes[0].set_title("Accelerometer")
axes[0].set_ylabel("Acceleration (g)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(time, gyro[:, i], label=axis_names[i])
axes[1].set_title("Gyroscope")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].legend()

axes[1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(output_path / f"exp{experiment_id:02d}_user{user_id:02d}.png", dpi=300)

acc_magnitude = np.linalg.norm(acc, axis=1)

plt.figure(figsize=(12, 4))
plt.plot(time, acc_magnitude)
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration magnitude (g)")
plt.title("Acceleration Magnitude")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / f"acceleration_magnitude_exp{experiment_id:02d}_user{user_id:02d}.png", dpi=300)