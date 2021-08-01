import os
from collections import OrderedDict
import argparse
import socket
import queue
import sys

import pickle
from joblib import Parallel, delayed
import scipy

from pythontuio import TuioClient
from pythontuio import Cursor
from pythontuio import TuioListener
from threading import Thread
import time


import trajnetplusplustools
import evaluator.write as write
from evaluator.design_pd import Table

import shutil
import os
import pickle

import torch
import numpy as np

import trajnetplusplustools
import trajnetbaselines

## Parallel Compute
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

def make_timestamp():
    return str(round(time.time() * 1000))

def cursor_to_row(timestamp, cursor):
    return trajnetplusplustools.data.TrackRow(frame=int(timestamp), pedestrian=cursor.session_id, x=50*cursor.position[0], y=50*cursor.position[1])

def process_scene(predictor, model_name, paths, scene_goal, args):
    ## For each scene, get predictions
    if 'sf_opt' in model_name:
        predictions = predictor(paths, sf_params=[0.5, 5.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal sf_params (no collision constraint) [0.5, 1.0, 0.1],
    elif 'orca_opt' in model_name:
        predictions = predictor(paths, orca_params=[0.4, 1.0, 0.3], n_predict=args.pred_length, obs_length=args.obs_length) ## optimal orca_params (no collision constraint) [0.25, 1.0, 0.3]
    elif  ('sf' in model_name) or ('orca' in model_name) or ('kf' in model_name):
        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)
    elif 'cv' in model_name:
        predictions = predictor(paths, n_predict=args.pred_length, obs_length=args.obs_length)
    else:
        predictions = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)
    return predictions

def serve_forever(args=None):
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])
    all_goals = {}
    seq_length = args.obs_length + args.pred_length

    ## Handcrafted Baselines (if included)
    if args.kf:
        args.output.append('/kf.pkl')
    if args.sf:
        args.output.append('/sf.pkl')
        args.output.append('/sf_opt.pkl')
    if args.orca:
        args.output.append('/orca.pkl')
        args.output.append('/orca_opt.pkl')
    if args.cv:
        args.output.append('/cv.pkl')

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Start writing predictions in dataset/test_pred
        for dataset in datasets:
            # Model's name
            name = dataset.replace(args.path.replace('_pred', '') + 'test/', '') + '.ndjson'
            print('NAME: ', name)

            # Loading the APPROPRIATE model
            ## Keep Adding Different Model Architectures to this List
            print("Model Name: ", model_name)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            device_name = "cpu"
            if(args.gpu):
                device_name = "cuda"
            goal_flag = False
            if 'kf' in model_name:
                print("Kalman")
                predictor = trajnetbaselines.classical.kalman.predict
            elif 'sf' in model_name:
                print("Social Force")
                predictor = trajnetbaselines.classical.socialforce.predict
            elif 'orca' in model_name:
                print("ORCA")
                predictor = trajnetbaselines.classical.orca.predict
            elif 'cv' in model_name:
                print("CV")
                predictor = trajnetbaselines.classical.constant_velocity.predict
            elif 'sgan' in model_name:
                print("SGAN")
                predictor = trajnetbaselines.sgan.SGANPredictor.load(model)
                device = torch.device(device_name)
                predictor.model.to(device)
                goal_flag = predictor.model.generator.goal_flag
            elif 'vae' in model_name:
                print("VAE")
                predictor = trajnetbaselines.vae.VAEPredictor.load(model)
                device = torch.device(device_name)
                predictor.model.to(device)
                goal_flag = predictor.model.goal_flag
            elif 'lstm' in model_name:
                print("LSTM")
                predictor = trajnetbaselines.lstm.LSTMPredictor.load(model)
                device = torch.device(device_name)
                predictor.model.to(device)
                goal_flag = predictor.model.goal_flag
            else:
                print("Model Architecture not recognized")
                raise ValueError

            # Read Scenes from 'test' folder
            reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
            ## Necessary modification of train scene to add filename (for goals)
            scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]

            ## Consider goals
            ## Goal file must be present in 'goal_files/test_private' folder
            ## Goal file must have the same name as corresponding test file
            if goal_flag:
                print("Loading Test Goals file")
                goal_dict = pickle.load(open('goal_files/test_private/' + dataset +'.pkl', "rb"))
                all_goals[dataset] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scenes}

            ## Get Goals
            if goal_flag:
                scene_goals = [np.array(all_goals[filename][scene_id]) for filename, scene_id, _ in scenes]
            else:
                scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

            # Get the model prediction and write them in corresponding test_pred file
            # VERY IMPORTANT: Prediction Format
            # The predictor function should output a dictionary. The keys of the dictionary should correspond to the prediction modes.
            # ie. predictions[0] corresponds to the first mode. predictions[m] corresponds to the m^th mode.... Multimodal predictions!
            # Each modal prediction comprises of primary prediction and neighbour (surrrounding) predictions i.e. predictions[m] = [primary_prediction, neigh_predictions]
            # Note: Return [primary_prediction, []] if model does not provide neighbour predictions
            # Shape of primary_prediction: Tensor of Shape (Prediction length, 2)
            # Shape of Neighbour_prediction: Tensor of Shape (Prediction length, n_tracks - 1, 2).
            # (See LSTMPredictor.py for more details)
            scenes = tqdm(scenes)

            import timeit

            UDP_IP = "localhost"
            UDP_PORT = 6666
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            def print_cursor(cursor):
                print(str(round(time.time() * 1000)) + '|' + str(cursor.session_id) + '|' + str(cursor.position[0]) + '|' + str(cursor.position[1]))

            class MyListener(TuioListener):
                def __init__(self):
                    super().__init__()
                    self.q = queue.Queue()
                    self.current_timestamp = ""
                    self.frame_count = 0
                def put_cursor(self, cursor):
                    tmstmp = make_timestamp()
                    if self.current_timestamp != tmstmp:
                        if self.frame_count > args.obs_length * 5:
                            paths = self.get_paths()
                            self.make_prediction(paths)
                            self.frame_count = 0
                        else:
                            self.frame_count += 1
                    self.q.put((tmstmp, cursor))
                def add_tuio_cursor(self, cursor: Cursor):
                    print("detect a new Cursor")
                    self.put_cursor(cursor)
                def update_tuio_cursor(self, cursor: Cursor):
                    self.put_cursor(cursor)
                def remove_tuio_cursor(self, cursor: Cursor):
                    print("a cursor was removed")
                    self.put_cursor(cursor)

                def get_paths(self):
                    cursors = []
                    paths = []
                    while not self.q.empty():
                        cursors.append(self.q.get())
                        self.q.task_done()
                    self.q.join()
                    person_to_index = {}
                    counter = 0
                    for (timestamp, cursor) in cursors:
                        row = cursor_to_row(timestamp, cursor)
                        if not cursor.session_id in person_to_index:
                            paths.append([])
                            person_to_index[cursor.session_id] = counter
                            counter += 1
                        index = person_to_index[cursor.session_id]
                        paths[index].append(row)
                    paths_filtered = [path[0::5] for path in paths if len(path[0::5]) >= args.obs_length]

                    if len(paths_filtered) < 1:
                        for cursor in cursors:
                            self.q.put(cursor)
                    return paths_filtered

                def make_prediction(self, paths):
                    if len(paths) < 1:
                        print("No paths that are long enough")
                        return False
                    start = timeit.default_timer()
                    scene_goal = []
                    for i in range(len(paths)):
                        scene_goal.append([.0, .0])
                    prediction_list = predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args, device=device_name)
                    stop = timeit.default_timer()
                    #print(prediction_list)
                    print('Prediction time: ', stop - start)
                    print("Length paths")
                    print(len(paths))
                    print("Length of path in paths")
                    for path in paths:
                        print(len(path))
                    print(paths)
                    print(prediction_list)

                    msg = ""

                    ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                    scene_id = 1
                    predictions = prediction_list
                    observed_path = paths[0]
                    frame_diff = observed_path[1].frame - observed_path[0].frame
                    first_frame = observed_path[args.obs_length-1].frame + frame_diff
                    ped_id = observed_path[0].pedestrian
                    ped_id_ = []
                    for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                       ped_id_.append(paths[j+1][0].pedestrian)

                    ## Write SceneRow
                    scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame,
                                                            observed_path[0].frame + (seq_length - 1) * frame_diff, 2.5, 0)
                    # scenerow = trajnetplusplustools.SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
                    """
                    msg = trajnetplusplustools.writers.trajnet(scenerow) + '\n'
                    sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))
                    print(msg)
                    """

                    for m in range(len(predictions)):
                       prediction, neigh_predictions = predictions[m]
                       ## Write Primary
                       for i in range(len(prediction)):
                           track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                                 prediction[i, 0].item(), prediction[i, 1].item(), m, scene_id)
                           msg += trajnetplusplustools.writers.trajnet(track) + ', '
                           #sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))
                           #print(msg)

                       ## Write Neighbours (if non-empty)
                       if len(neigh_predictions):
                           for n in range(neigh_predictions.shape[1]):
                               neigh = neigh_predictions[:, n]
                               for j in range(len(neigh)):
                                   track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                                         neigh[j, 0].item(), neigh[j, 1].item(), m, scene_id)
                                   msg += trajnetplusplustools.writers.trajnet(track) + ', '
                                   #sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))
                                   #print(msg)
                    #sys.exit()
                    msg = "[{}]".format(msg[:-2])
                    sock.sendto(bytes(msg, "utf-8"), (UDP_IP, UDP_PORT))
                    print(len(bytes(msg, "utf-8")))
                    print(msg)

            client = TuioClient(("localhost",3333))
            t = Thread(target=client.start)
            listener = MyListener()
            client.add_listener(listener)
            t.start()

"""

            with open(args.path + '{}/{}'.format(model_name, name), "a") as myfile:
                ## Get all predictions in parallel. Faster!
                pred_list = Parallel(n_jobs=12)(delayed(process_scene)(predictor, model_name, paths, scene_goal, args)
                                                for (_, _, paths), scene_goal in zip(scenes, scene_goals))

                ## Write All Predictions
                for (predictions, (_, scene_id, paths)) in zip(pred_list, scenes):
                    ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                    observed_path = paths[0]
                    frame_diff = observed_path[1].frame - observed_path[0].frame
                    first_frame = observed_path[args.obs_length-1].frame + frame_diff
                    ped_id = observed_path[0].pedestrian
                    ped_id_ = []
                    for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                        ped_id_.append(paths[j+1][0].pedestrian)

                    ## Write SceneRow
                    scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame,
                                                             observed_path[0].frame + (seq_length - 1) * frame_diff, 2.5, 0)
                    # scenerow = trajnetplusplustools.SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
                    myfile.write(trajnetplusplustools.writers.trajnet(scenerow))
                    myfile.write('\n')

                    for m in range(len(predictions)):
                        prediction, neigh_predictions = predictions[m]
                        ## Write Primary
                        for i in range(len(prediction)):
                            track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                                  prediction[i, 0].item(), prediction[i, 1].item(), m, scene_id)
                            myfile.write(trajnetplusplustools.writers.trajnet(track))
                            myfile.write('\n')

                        ## Write Neighbours (if non-empty)
                        if len(neigh_predictions):
                            for n in range(neigh_predictions.shape[1]):
                                neigh = neigh_predictions[:, n]
                                for j in range(len(neigh)):
                                    track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                                          neigh[j, 0].item(), neigh[j, 1].item(), m, scene_id)
                                    myfile.write(trajnetplusplustools.writers.trajnet(track))
                                    myfile.write('\n')

"""
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='directory of data to test')
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--sf', action='store_true',
                        help='consider socialforce in evaluation')
    parser.add_argument('--orca', action='store_true',
                        help='consider orca in evaluation')
    parser.add_argument('--kf', action='store_true',
                        help='consider kalman in evaluation')
    parser.add_argument('--cv', action='store_true',
                        help='consider constant velocity in evaluation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    parser.add_argument('--gpu', default=False, type=bool,
                        help='if should use GPU')
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.output = args.output if args.output is not None else []
    ## assert length of output models is not None
    if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
        assert len(args.output), 'No output file is provided'

    ## Path to the data folder name to predict
    args.path = 'DATA_BLOCK/' + args.path + '/'

    ## Test_pred : Folders for saving model predictions
    args.path = args.path + 'test_pred/'

    ## Writes to Test_pred
    ## Does NOT overwrite existing predictions if they already exist ###
    serve_forever(args)

if __name__ == '__main__':
    main()
