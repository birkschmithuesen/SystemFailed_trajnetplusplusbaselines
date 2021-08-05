import argparse
import socket
import sys
import time
from queue import Queue
from collections import deque
from threading import Thread

import torch
from pythontuio import TuioClient
from pythontuio import Cursor
from pythontuio import TuioListener

import trajnetplusplustools
import trajnetbaselines

QUEUE_MAX_LENGTH = 10
UDP_PORT = 6666
TUIO_PORT = 3334
PHARUS_FIELD_SIZE_X = 16.4
PHARUS_FIELD_SIZE_Y = 9.06


def cursor_to_row(timestamp, cursor):
    return trajnetplusplustools.data.TrackRow(frame=int(timestamp),
                                              pedestrian=cursor.session_id,
                                              x=PHARUS_FIELD_SIZE_X *
                                              cursor.position[0],
                                              y=PHARUS_FIELD_SIZE_Y * cursor.position[1])


def serve_forever(args=None, touch_designer_ip="", ml_fps_callback=None, pharus_fps_callback=None, pharus_sender_fps=60):

    pharus_sender_fps = int(pharus_sender_fps / 2.5)

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
                    track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff,
                                                          ped_id,
                                                          prediction[i,
                                                                     0].item(),
                                                          prediction[i,
                                                                     1].item(),
                                                          m,
                                                          scene_id)
                    prediction_paths.append(track)
                    msg += trajnetplusplustools.writers.trajnet(
                        track) + ', '
            send_to_touchdesigner(msg)

            # Write Neighbours (if non-empty)
            if len(neigh_predictions):
                for n in range(neigh_predictions.shape[1]):
                    msg = ""
                    neigh = neigh_predictions[:, n]
                    for j, _ in enumerate(neigh):
                        track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff,
                                                              ped_id_[n],
                                                              neigh[j, 0].item(
                                                              ),
                                                              neigh[j, 1].item(
                                                              ),
                                                              m,
                                                              scene_id)
                        msg += trajnetplusplustools.writers.trajnet(
                            track) + ', '
                        prediction_paths.append(track)
                    send_to_touchdesigner(msg)

            return prediction_paths

        q = Queue()

        def prediction_loop():
            while True:
                paths = q.get()
                new_frame_time = time.time()
                pred_paths = make_prediction(paths)
                fps = 1/(time.time() - new_frame_time)
                fps = int(fps)
                if ml_fps_callback:
                    ml_fps_callback(fps, paths, pred_paths)
                while q.qsize() > QUEUE_MAX_LENGTH:
                    q.get()
                sys.stdout.write("ML FPS: %d  --- Queue Length: %d \r"
                                 % (fps, q.qsize()))

        class MyListener(TuioListener):
            def __init__(self):
                super().__init__()
                self.people = {}
                self.bundle = []
                self.fseq = 0
                self.frame_count = 0
                self.prev_frame_time = 0
                self.new_frame_time = 0
                self.ml_fps = 0
                self.fps_callback = pharus_fps_callback

            def put_cursor(self, cursor):
                self.bundle.append(cursor)

            def add_tuio_cursor(self, cursor: Cursor):
                length = int(args.obs_length*pharus_sender_fps + 1)
                self.people[cursor.session_id] = deque(maxlen=length)
                self.put_cursor(cursor)

            def update_tuio_cursor(self, cursor: Cursor):
                if not cursor.session_id in self.people:
                    self.add_tuio_cursor(cursor)
                self.put_cursor(cursor)

            def remove_tuio_cursor(self, cursor: Cursor):
                if cursor.session_id in self.people:
                    del self.people[cursor.session_id]

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
                paths = self.get_paths()
                if paths:
                    q.put(paths)

                self.bundle = []

                if self.fps_callback:
                  self.fps_callback(fps, paths)

            def get_paths(self):
                cursors = []
                paths = []
                for _, dq in self.people.items():
                    if len(dq) == dq.maxlen:
                        for cursor in list(dq)[::pharus_sender_fps]:
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

        client = TuioClient(("localhost", TUIO_PORT))
        t1 = Thread(target=client.start)
        listener = MyListener()
        client.add_listener(listener)
        t1.start()

        t2 = Thread(target=prediction_loop)
        t2.start()

        return [t1, t2]


def main(args, touch_designer_ip="192.168.0.2", fps_callback=None, pharus_fps_callback=None):

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args(args)

    args.output = args.output if args.output is not None else []
    # assert length of output models is not None
    if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
        assert len(args.output), 'No output file is provided'

    return serve_forever(args, touch_designer_ip, fps_callback, pharus_fps_callback)


if __name__ == '__main__':
    main(sys.argv[1:])
    while True:
        pass
