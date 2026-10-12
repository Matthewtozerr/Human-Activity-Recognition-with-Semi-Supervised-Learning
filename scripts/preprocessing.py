from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

raw_data_path = Path(__file__).parent.parent / "data" / "RawData"
preprocessed_data_path = Path(__file__).parent.parent / "data" / "ProcessedData"
preprocessed_data_path.mkdir(parents=True, exist_ok=True)

recordings = {}

acc_files = list(sorted(raw_data_path.glob("acc*.txt")))

#grabs 3 linear and 3 angular parameters for each experiement and user
for acc_file in acc_files:
    recording_name = acc_file.stem.split("_", 1)[1]  #e.g. 'exp01_user01'
    gyro_file = raw_data_path / f"gyro_{recording_name}.txt"

    experiment, user = recording_name.split("_")
    experiment_id = int(experiment.removeprefix("exp"))
    user_id = int(user.removeprefix("user"))

    acc = np.loadtxt(acc_file)
    gyro = np.loadtxt(gyro_file)

    if acc.shape[1] != 3 or gyro.shape[1] != 3:
        raise ValueError(f"{recording_name}: each sensor must have three columns")
    if acc.shape[0] != gyro.shape[0]:
        raise ValueError(f"{recording_name}: sensor samples counts do not match")
    
    sensor_data = np.concatenate((acc, gyro), axis=1)  

    recordings[(experiment_id, user_id)] = sensor_data

#grabs activity id and start/end times for each experiment and user
labels = pd.read_csv(
    raw_data_path / "labels.txt",
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

#extracts all data points for each activity
def extract_labelled_segments(recordings, labels):

    segements = []
    
    for row in labels.itertuples(index=False):
        key = (int(row.experiment_id), int(row.user_id))
        
        if key not in recordings:
            raise ValueError(f"Recording for experiment {row.experiment_id}, user {row.user_id} not found")
        
        data = recordings[key]
        start = int(row.start) - 1
        end = int(row.end)

        if not (0 <= start < end <= len(data)):
            raise ValueError(f"Invalid interval for {key}: ",
                            f"start={start}, end={end}")
    
        segment_data = data[start:end, :]

        segements.append({
            "experiment_id": key[0],
            "user_id": key[1],
            "activity_id": int(row.activity_id),
            "data": segment_data
        })
    
    return segements

segments = extract_labelled_segments(recordings, labels)

segments_df = pd.DataFrame(segments)
# segments_df.to_csv(preprocessed_data_path / "segments.csv", index=False)

#splits the segments into equal overlapping 'windows'
def create_windows(data, window_size=128, step_size=64):

    windows = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end, :]
        windows.append(window)

    if not windows:
        return np.empty((0, window_size, data.shape[1]), dtype=data.dtype)
    
    return np.stack(windows)

#apply windowing to all segements

X_list = []
y_list = []
user_list = []

for segment in segments:
    segment_data = segment["data"]
    windows = create_windows(segment_data)

    n_windows = len(windows)
    if n_windows == 0:
        continue

    X_list.append(windows)
    y_list.extend([segment["activity_id"]] * n_windows) #add activity for each window
    user_list.extend([segment["user_id"]] * n_windows) #add user for each window

if not X_list:
    raise ValueError("No valid windows were created.")

X = np.concatenate(X_list, axis=0)
y = np.array(y_list, dtype=np.int64)
user_ids = np.array(user_list, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("User IDs shape:", user_ids.shape)
print(f"Number of activities: {len(np.unique(y))}")
print(f"Number of participants: {len(np.unique(user_ids))}")

splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.66, random_state=42)

train_idx, temp_idx = next(splitter1.split(X, y, groups=user_ids))

X_train = X[train_idx]
y_train = y[train_idx]
user_train = user_ids[train_idx]

X_temp = X[temp_idx]
y_temp = y[temp_idx]
user_temp = user_ids[temp_idx]

val_idx, test_idx = next(splitter2.split(X_temp, y_temp, groups=user_temp))

X_val = X_temp[val_idx]
y_val = y_temp[val_idx]
user_val = user_temp[val_idx]

X_test = X_temp[test_idx]
y_test = y_temp[test_idx]
user_test = user_temp[test_idx]

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

def print_activity_distribution(name, labels):
    activity_ids, counts = np.unique(
        labels,
        return_counts=True
    )

    print(f"\n{name} activity distribution:")

    for activity_id, count in zip(activity_ids, counts):
        percentage = 100 * count / len(labels)

        print(
            f"Activity {activity_id}: {count} windows ({percentage:.1f}%)"
        )


print_activity_distribution("Training", y_train)
print_activity_distribution("Validation", y_val)
print_activity_distribution("Test", y_test)

#check all activities are represented in training
all_classes = set(np.unique(y))
train_classes = set(np.unique(y_train))

missing_classes = all_classes - train_classes

if missing_classes:
    raise ValueError(
        f"Training set is missing activity IDs: {missing_classes}"
    )

for name, labels_split in [
    ("Validation", y_val),
    ("Test", y_test)
]:
    missing = all_classes - set(np.unique(labels_split))
    print(f"{name} missing classes: {sorted(missing)}")

np.save(preprocessed_data_path / "subject_train.npy", user_train)
np.save(preprocessed_data_path / "subject_val.npy", user_val)
np.save(preprocessed_data_path / "subject_test.npy", user_test)

#z-score normalization (mean = 0, std = 1), all channels on a comparable scale

#channel means and stds
#only use training data to compute mean and std to avoid data leakage
mean = np.mean(X_train, axis=(0, 1))
std = np.std(X_train, axis=(0, 1))
print("Channel means:", mean)
print("Channel stds:", std)

#z = (X - mean) / std
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

print("After normalization:")
print("X_train mean:", np.mean(X_train, axis=(0, 1)))
print("X_train std:", np.std(X_train, axis=(0, 1)))

if not np.isfinite(X_train).all() or not np.isfinite(X_val).all() or not np.isfinite(X_test).all():
    raise ValueError("NaN or Inf values found in the normalized data.")

#save the normalization parameters for later use
np.save(preprocessed_data_path / "mean.npy", mean)
np.save(preprocessed_data_path / "std.npy", std)

np.save(preprocessed_data_path / "X_train.npy", X_train)
np.save(preprocessed_data_path / "X_val.npy", X_val)
np.save(preprocessed_data_path / "X_test.npy", X_test)

np.save(preprocessed_data_path / "y_train.npy", y_train)
np.save(preprocessed_data_path / "y_val.npy", y_val)
np.save(preprocessed_data_path / "y_test.npy", y_test)