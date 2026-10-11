from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = Path(__file__).parent.parent / "data" / "RawData"
output_path = Path(__file__).parent.parent / "outputs" 
output_path.mkdir(parents=True, exist_ok=True)

fs = 50 #Hz

experiment_id = 1
user_id = 1

#load sensor data
acc_file = data_path / f'acc_exp{experiment_id:02d}_user{user_id:02d}.txt' #accelerometer (linear)
gyro_file = data_path / f'gyro_exp{experiment_id:02d}_user{user_id:02d}.txt' #gyroscope (angular)

acc = np.loadtxt(acc_file)
print(f"Accelerometer data shape: {acc.shape}")
gyro = np.loadtxt(gyro_file)
print(f"Gyroscope data shape: {gyro.shape}")

n_samples = len(acc) # = len(gyro)

time = np.arange(n_samples) / fs
print(f"Recording duration: {time[-1]:.2f} seconds")

#load activity names
activity_file = data_path.parent / "activity_labels.txt"

activity_labels = pd.read_csv(
    activity_file,
    sep=r"\s+",
    header=None,
    names=["activity_id", "activity"]
)

#load annotations
labels = pd.read_csv(
    data_path / "labels.txt",
    sep=r"\s+",
    header=None,
    names=[
        "experiment_id",
        "user_id",
        "activity_id",
        "start",
        "end"
    ]
)

#join activity names
labels = labels.merge(
    activity_labels,
    on="activity_id",
    how="left"
)

#select relevant annotations
recording_labels = labels[
    (labels["experiment_id"] == experiment_id)
    & (labels["user_id"] == user_id)
]

print(recording_labels)

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(14, 8),
    sharex=True
)

axis_names = ["X", "Y", "Z"]

# Use a consistent pastel colour for each activity, behind the sensor traces.
activity_colours = plt.get_cmap("Set3", len(activity_labels))
seen_activities = set()
for annotation in recording_labels.itertuples(index=False):
    activity_id = annotation.activity_id
    for ax in axes:
        ax.axvspan(
        annotation.start / fs,
        annotation.end / fs,
        facecolor=activity_colours(activity_id - 1),
        alpha=0.5,
        zorder=0,
        label=annotation.activity.replace("_", " ").title()
        if activity_id not in seen_activities else "_nolegend_",
    )
    seen_activities.add(activity_id)

for i in range(3):
    axes[0].plot(time, acc[:, i], label=axis_names[i])
    axes[1].plot(time, gyro[:, i], label=axis_names[i])

axes[0].set_title("Accelerometer")
axes[0].set_ylabel("Acceleration (g)")
axes[0].legend(ncol=4, fontsize=8, loc='upper left')
axes[0].grid(alpha=0.3)

axes[1].set_title("Gyroscope")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].legend(ncol=4, fontsize=8, loc='upper left')
axes[1].grid(alpha=0.3)

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


def load_activity_segment(row):

    exp = row.experiment_id
    user = row.user_id

    start = row.start - 1
    end = row.end

    return(acc[start:end, :], gyro[start:end, :])

activities = recording_labels['activity'].unique().tolist()

fig, axes = plt.subplots(
    nrows=4, 
    ncols=3,
    figsize=(12, 12)
)
axes = axes.ravel()

for ax, activity in zip(axes, activities):
    matches = recording_labels[
        recording_labels["activity"] == activity
]
    # Select the first available example
    row = matches.iloc[0]

    acc_segment, _ = load_activity_segment(row)

    t = np.arange(len(acc_segment)) / fs

    ax.plot(t, acc_segment[:, 0], label="X")
    ax.plot(t, acc_segment[:, 1], label="Y")
    ax.plot(t, acc_segment[:, 2], label="Z")

    ax.set_title(activity, fontsize=8)
    ax.set_ylabel("Acceleration (g)")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Time (s)")
axes[0].legend()

plt.tight_layout()
plt.savefig(output_path / "activity_comparison.png", dpi=200)
