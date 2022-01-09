import subprocess
from threading import Timer

from evaluator import server_udp

def start_udp_splitter(receiver_ip="127.0.0.1", target_ip="192.168.0.2"):
    return subprocess.Popen(["./../udp-splitter/udp-splitter", "{}:3333".format(receiver_ip), "localhost:3334", "{}:3333".format(target_ip)])

def start_inference_server(model_path="OUTPUT_BLOCK/pharus_kreis_mit_stehenbleiben/lstm_social_None.pkl.epoch10",
                           pharus_receiver_ip="localhost",
                           touch_designer_ip="192.168.0.2",
                           fps_callback=None,
                           pharus_fps_callback=None,
                           pred_length=12,
                           obs_length=9,
                           fps=30):

    args = ["--output", model_path, "--gpu",
            "True", "--obs_length", str(obs_length), "--pred_length", str(pred_length),
            "--fps", str(fps)]

    client_and_threads = server_udp.main(args, pharus_receiver_ip, touch_designer_ip, fps_callback, pharus_fps_callback)
    return client_and_threads

if __name__ == "__main__":
    start_inference_server()
    while True:
        pass