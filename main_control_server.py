import subprocess
from threading import Timer

from evaluator import server_udp


def start_udp_splitter():
    return subprocess.Popen(["./../udp-splitter/udp-splitter", "localhost:3333", "localhost:3334", "192.168.0.2:3333"])


udp_splitter = start_udp_splitter()

args = ["--output", "OUTPUT_BLOCK/pharus_saturday/lstm_social_None.pkl.epoch45", "--gpu", "True"]
server_udp.main(args)
