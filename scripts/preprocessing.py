from pathlib import Path
import numpy as np
import pandas as pd

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
segments_df.to_csv(preprocessed_data_path / "segments.csv", index=False)

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
subject_ids = np.array(user_list, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Subject IDs shape:", subject_ids.shape)