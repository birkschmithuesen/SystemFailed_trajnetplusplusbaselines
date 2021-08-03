import subprocess
from threading import Timer

from evaluator import server_udp

def start_udp_splitter(receiver_ip="127.0.0.1", target_ip="192.168.0.2"):
    return subprocess.Popen(["./../udp-splitter/udp-splitter", "{}:3333".format(receiver_ip), "localhost:3334", "{}:3333".format(target_ip)])

def start_inference_server(model_path="OUTPUT_BLOCK/pharus_saturday/pharus_saturday_only/lstm_social_None.pkl.epoch45",
                           pharus_receiver_ip="localhost",
                           pharus_splitter_target_ip="192.168.0.2",
                           fps_callback=None,
                           pharus_fps_callback=None):
    udp_splitter = start_udp_splitter(pharus_receiver_ip)

    args = ["--output", model_path, "--gpu",
            "True"]
    server_udp.main(args, fps_callback, pharus_fps_callback)

if __name__ == "__main__":
    start_inference_server()