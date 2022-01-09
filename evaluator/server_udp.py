import argparse
import socket
import sys
import threading
import time
from queue import Queue, Empty
from collections import deque
from threading import Thread

import torch
import numpy as np
from pythontuio import TuioClient
from pythontuio import Cursor
from pythontuio import TuioListener

import trajnetplusplustools
import trajnetbaselines

QUEUE_MAX_LENGTH = 1
UDP_PORT = 6666
TUIO_HOST = "0.0.0.0"
TUIO_PORT = 3000
INPUT_SLIDING_WINDOW_SIZE = 1
PREDICTION_START_OFFSET = 0


def average_prediction_path(ped_id, n, x, y, path_deque):
    prev_vals = []
    for path_dict in list(path_deque):
        if ped_id in path_dict:
            prev_vals.append(path_dict[ped_id])
    #if len(prev_vals) == 0:
    #    return (x, y)

    xy_vals = [pred_list[n] for pred_list in prev_vals]
    x_vals = [coord[0] for coord in xy_vals]
    y_vals = [coord[1] for coord in xy_vals]
    x_avg = (sum(x_vals) + x) / (len(x_vals) + 1)
    y_avg = (sum(y_vals) + y) / (len(y_vals) + 1)
    return (x_avg, y_avg)


def cursor_to_row(timestamp, cursor):
    return trajnetplusplustools.data.TrackRow(frame=int(timestamp),
                                              pedestrian=cursor.session_id,
                                              x=cursor.position[0],
                                              y=cursor.position[1])


def resize_deques_dict(deques_dict, size):
    deques_dict_copy = {}
    for session_id, dq in deques_dict.items():
        deques_dict_copy[session_id] = deque(maxlen=size)
        for item in list(dq):
            deques_dict_copy[session_id].append(item)
    return deques_dict_copy


def serve_forever(args=None, pharus_receiver_ip="127.0.0.1", touch_designer_ip="", ml_fps_callback=None, pharus_fps_callback=None):

    global pharus_sender_fps
    pharus_sender_fps = int(args.fps / 2.5)

    global prediction_deque
    prediction_deque = deque(maxlen=25)

    # Handcrafted Baselines (if included)
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

    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        # Loading the APPROPRIATE model
        # Keep Adding Different Model Architectures to this List
        print("Model Name: ", model_name)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        device_name = "cpu"
        if args.gpu:
            device_name = "cuda"

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

        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def send_to_touchdesigner(msg):
            formatted_msg = "[{}]".format(msg[:-2])
            udp_socket.sendto(
                bytes(formatted_msg, "utf-8"), (touch_designer_ip, UDP_PORT))

        def make_prediction(paths):
            scene_goal = []
            for _, _ in enumerate(paths):
                scene_goal.append([.0, .0])
            prediction_list = predictor(paths,
                                        scene_goal,
                                        n_predict=args.pred_length,
                                        obs_length=args.obs_length,
                                        modes=args.modes,
                                        args=args,
                                        device=device_name)

            # Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
            prediction_paths = []
            prediction_paths_dict = {}
            scene_id = 1
            predictions = prediction_list
            observed_path = paths[0]
            frame_diff = observed_path[1].frame - observed_path[0].frame
            first_frame = observed_path[args.obs_length -
                                        1].frame + frame_diff
            ped_id = observed_path[0].pedestrian
            ped_id_ = []
            for j, _ in enumerate(paths[1:]):  # Only need neighbour ids
                ped_id_.append(paths[j+1][0].pedestrian)

            for _, m in enumerate(predictions):
                prediction, neigh_predictions = predictions[m]
                # Write Primary
                msg = ""
                for i, _ in enumerate(prediction):
                    x = prediction[i, 0].item()
                    y = prediction[i, 1].item()
                    avg_x, avg_y = average_prediction_path(
                        ped_id, i, x, y, prediction_deque)
                    if not ped_id in prediction_paths_dict:
                        prediction_paths_dict[ped_id] = []
                    prediction_paths_dict[ped_id].append((x, y))
                    last_pharus_x = paths[0][-1].x
                    last_pharus_y = paths[0][-1].y
                    track_relative = trajnetplusplustools.TrackRow(first_frame + i * frame_diff,
                                                          ped_id,
                                                          avg_x-last_pharus_x,
                                                          avg_y-last_pharus_y,
                                                          m,
                                                          scene_id)
                    track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff,
                                                                   ped_id,
                                                                   avg_x,
                                                                   avg_y,
                                                                   m,
                                                                   scene_id)
                    if i < PREDICTION_START_OFFSET:
                        continue
                    prediction_paths.append(track)
                    msg += trajnetplusplustools.writers.trajnet(
                        track_relative) + ', '
            send_to_touchdesigner(msg)

            # Write Neighbours (if non-empty)
            if len(neigh_predictions):
                for n in range(neigh_predictions.shape[1]):
                    msg = ""
                    neigh = neigh_predictions[:, n]
                    for j, _ in enumerate(neigh):
                        ped_id = ped_id_[n]
                        x = neigh[j, 0].item()
                        y = neigh[j, 1].item()
                        x_avg, y_avg = average_prediction_path(
                            ped_id, j, x, y, prediction_deque)
                        if not ped_id in prediction_paths_dict:
                            prediction_paths_dict[ped_id] = []
                        prediction_paths_dict[ped_id].append((x, y))
                        last_pharus_x = paths[n+1][-1].x
                        last_pharus_y = paths[n+1][-1].y
                        track_relative = trajnetplusplustools.TrackRow(first_frame + j * frame_diff,
                                                              ped_id,
                                                              x_avg-last_pharus_x,
                                                              y_avg-last_pharus_y,
                                                              m,
                                                              scene_id)
                        track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff,
                                                              ped_id,
                                                              x_avg,
                                                              y_avg,
                                                              m,
                                                              scene_id)
                        if j < PREDICTION_START_OFFSET:
                            continue
                        prediction_paths.append(track)
                        msg += trajnetplusplustools.writers.trajnet(
                            track_relative) + ', '
                    send_to_touchdesigner(msg)
            prediction_deque.append(prediction_paths_dict)
            return prediction_paths

        q = Queue()
        fps_deque = deque(maxlen=25)

        class PredictionThread(threading.Thread):
            def __init__(self, name="ML Prediction Thread"):
                self._stop_event = threading.Event()
                super().__init__(name=name)

            def run(self):
                while not self._stop_event.is_set():
                    try:
                        paths = q.get_nowait()
                    except Empty:
                        continue
                    new_frame_time = time.time()
                    pred_paths = make_prediction(paths)
                    fps = 1/(time.time() - new_frame_time)
                    fps = int(fps)
                    if ml_fps_callback:
                        ml_fps_callback(fps, paths, pred_paths)
                    while q.qsize() > QUEUE_MAX_LENGTH:
                        try:
                            paths = q.get_nowait()
                        except Empty:
                            continue
                    sys.stdout.write("ML FPS: %d  --- Queue Length: %d \r"
                                     % (fps, q.qsize()))

                udp_socket.close()

            def join(self, timeout=None):
                self._stop_event.set()
                super().join(timeout)

        class MyListener(TuioListener):
            def __init__(self):
                super().__init__()
                self.people = {}
                self.people_deques = {}
                self.bundle = []
                self.fseq = 0
                self.frame_count = 0
                self.prev_frame_time = 0
                self.new_frame_time = 0
                self.ml_fps = 0
                self.fps_callback = pharus_fps_callback
                self.sliding_window_size = INPUT_SLIDING_WINDOW_SIZE
                self.update_obs_length_size = False
                self.callbacks = []

            def put_cursor(self, cursor):
                self.bundle.append(cursor)

            def add_tuio_cursor(self, cursor: Cursor):
                global pharus_sender_fps
                length = int(args.obs_length*pharus_sender_fps)
                self.people[cursor.session_id] = deque(maxlen=length)
                self.people_deques[cursor.session_id] = deque(
                    maxlen=self.sliding_window_size)
                self.put_cursor(cursor)

            def update_tuio_cursor(self, cursor: Cursor):
                if not cursor.session_id in self.people:
                    self.add_tuio_cursor(cursor)
                self.put_cursor(cursor)

            def remove_tuio_cursor(self, cursor: Cursor):
                if cursor.session_id in self.people:
                    del self.people[cursor.session_id]
                    del self.people_deques[cursor.session_id]

            def refresh(self, fseq):
                self.new_frame_time = time.time()
                fps = 1/(self.new_frame_time - self.prev_frame_time)
                fps = int(fps)
                self.prev_frame_time = self.new_frame_time

                for cursor in self.bundle:
                    cursor_copy = Cursor(cursor.session_id)
                    cursor_copy.position = cursor.position
                    item = (fseq, cursor_copy)
                    self.people[cursor.session_id].append(item)
                for session_id, dq in self.people.items():
                    if len(dq) == dq.maxlen:
                        self.people_deques[session_id].append(list(dq))
                paths = self.get_paths()
                if paths:
                    q.put(paths)

                self.bundle = []

                if self.fps_callback:
                    self.fps_callback(fps, paths)
                if self.update_obs_length_size:
                    self.finalize_update_obs_length()
                    self.update_obs_length_size = False

                while(len(self.callbacks) > 0):
                    self.callbacks.pop()()

            def average_path(self, session_id):
                if not session_id in self.people_deques:
                    print("Error session id {} not in people_deques: {}".format(
                        session_id, str(self.people_deques)))
                    return
                average_x = np.zeros(self.people[session_id].maxlen)
                average_y = np.zeros(self.people[session_id].maxlen)
                for dq_list in self.people_deques[session_id]:
                    x_pos_list = [tup[1].position[0] for tup in dq_list]
                    y_pos_list = [tup[1].position[1] for tup in dq_list]
                    x_pos_list = np.array(x_pos_list)
                    y_pos_list = np.array(y_pos_list)
                    average_x += x_pos_list
                    average_y += y_pos_list
                buffer_len = len(self.people_deques[session_id])
                average_x /= buffer_len
                average_y /= buffer_len

                cursors = []
                zipped_pos = zip(average_x, average_y)
                for (timestamp, cursor), pos in zip(list(self.people[session_id]), zipped_pos):
                    cursor_copy = Cursor(session_id)
                    cursor_copy.position = pos
                    cursors.append((timestamp, cursor))

                return cursors

            def update_sliding_window_size(self, size):
                self.sliding_window_size = size
                self.people_deques = resize_deques_dict(
                    self.people_deques, size)
                print("Updated input sliding window size to {}".format(size))

            def update_sliding_window_output_size(self, size):
                global prediction_deque
                prediction_deque = deque(maxlen=size)
                print("Updated output sliding window size to {}".format(size))

            def update_pred_length(self, pred_length):
                def update_callback():
                    args.pred_length = pred_length
                self.callbacks.append(update_callback)
                print("Updated prediction length to {}".format(pred_length))

            def update_obs_length(self, obs_length):
                self.update_obs_length_size = obs_length
                print("Updated obs_length to {}".format(obs_length))

            def update_pharus_fps(self, fps):
                def update_callback():
                    global pharus_sender_fps
                    pharus_sender_fps = int(fps / 2.5)
                    self.people = {}
                    self.people_deques = {}
                self.callbacks.append(update_callback)
                print("Updated incoming pharus fps to {}".format(fps))

            def finalize_update_obs_length(self):
                self.people = {}
                self.people_deques = {}
                args.obs_length = self.update_obs_length_size

            def get_paths(self):
                cursors = []
                paths = []
                for session_id, dq in self.people.items():
                    avg_dq_list = self.average_path(session_id)
                    if len(dq) == dq.maxlen:
                        for cursor in avg_dq_list[::pharus_sender_fps]:
                            cursors.append(cursor)

                if len(cursors) == 0:
                    return None

                person_to_index = {}
                counter = 0
                for timestamp, cursor in cursors:
                    row = cursor_to_row(timestamp, cursor)
                    if not cursor.session_id in person_to_index:
                        paths.append([])
                        person_to_index[cursor.session_id] = counter
                        counter += 1
                    index = person_to_index[cursor.session_id]
                    paths[index].append(row)

                return paths

        client = TuioClient((pharus_receiver_ip, TUIO_PORT))
        t1 = Thread(target=client.start)
        listener = MyListener()
        client.add_listener(listener)
        t1.start()

        t2 = PredictionThread()
        t2.start()

        return [client, t1, t2]


def main(args, pharus_receiver_ip="127.0.0.1", touch_designer_ip="192.168.0.2", fps_callback=None, pharus_fps_callback=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--fps', default=30, type=int,
                        help='FPS of incoming frames')
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
    args = parser.parse_args(args)

    args.output = args.output if args.output is not None else []
    # assert length of output models is not None
    if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
        assert len(args.output), 'No output file is provided'

    return serve_forever(args, pharus_receiver_ip, touch_designer_ip, fps_callback, pharus_fps_callback)


if __name__ == '__main__':
    main(sys.argv[1:])
    while True:
        pass
