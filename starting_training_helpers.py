import os, glob
import ujson as json
import pandas as pd
import numpy as np

from subprocess import Popen, PIPE

from trajnetbaselines.lstm.trainer import main

training_dirs = ["train", "test", "val", "test_private"]

def training_folder_is_valid(path):
    has_all_training_dirs = [os.path.isdir(os.path.join(path, training_dir)) for training_dir in
                             training_dirs]
    has_all_training_dirs = all(has_all_training_dirs)
    if not path:
        return (False, 'No training data selected.')
    elif not has_all_training_dirs:
        return (False, "The selected directory needs all of these folders: {}".format(training_dirs))

    return (True, "")

def get_training_data(training_data_path):
    is_valid, msg = training_folder_is_valid(training_data_path)
    if not is_valid:
        return msg

    training_file_paths = []
    for training_dir in training_dirs:
        path = os.path.join(training_data_path, training_dir)
        for file in os.listdir(path):
            if file.endswith(".ndjson"):
                training_file_paths.append(os.path.join(path, file))

    training_data_df = pd.DataFrame()
    for training_file in training_file_paths:
        records = map(json.loads, open(training_file))
        df = pd.DataFrame.from_records(records)
        training_data_df = training_data_df.append(df)

    training_data_df = training_data_df.drop("scene", axis=1).dropna().reset_index(drop=True)

    return training_data_df

def get_training_df_positions(training_df: pd.DataFrame):
    training_df["frame"] = training_df['track'].apply(lambda x: x["f"])
    training_df = training_df.sort_values("frame")

    person_paths = []
    person_to_index_map = {}
    counter = 0
    for _, row in training_df.iterrows():
        person_id = row["track"]["p"]
        if not person_id in person_to_index_map:
            person_to_index_map[person_id] = counter
            person_paths.append([[], []])
            counter += 1
        index = person_to_index_map[person_id]
        person_paths[index][0].append(row["track"]["x"])
        person_paths[index][1].append(row["track"]["y"])

    return person_paths

def start_training_thread(training_folder_name, epochs, pred_length, obs_length):
    args = ["python3.7", "-m", "trajnetbaselines.lstm.trainer", "--type", "social", "--path", training_folder_name, "--epochs", epochs, "--pred_length", pred_length, "--obs_length", obs_length]
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    return process

def pharus_recording_is_valid(path):
    if not path:
        return (False, "No pharus file selected.")
    elif not os.path.splitext(os.path.basename(path))[1] == ".trk":
        return (False, "Not a .trk file selected.")
    return (True, "")