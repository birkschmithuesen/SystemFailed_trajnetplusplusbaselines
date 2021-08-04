import subprocess
from threading import Timer

from evaluator import server_udp

def start_udp_splitter(receiver_ip="127.0.0.1", target_ip="192.168.0.2"):
    return subprocess.Popen(["./../udp-splitter/udp-splitter", "{}:3333".format(receiver_ip), "localhost:3334", "{}:3333".format(target_ip)])

def start_inference_server(model_path="OUTPUT_BLOCK/pharus/lstm_social_None.pkl.epoch45",
                           pharus_receiver_ip="localhost",
                           touch_designer_ip="192.168.0.2",
                           fps_callback=None,
                           pharus_fps_callback=None):
    udp_splitter = start_udp_splitter(pharus_receiver_ip, touch_designer_ip)

    args = ["--output", model_path, "--gpu",
            "True"]
    return server_udp.main(args, touch_designer_ip, fps_callback, pharus_fps_callback).append(udp_splitter)

if __name__ == "__main__":
    start_inference_server()
    while True:
        pass