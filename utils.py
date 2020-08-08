import datetime
from typing import List


def get_now_str():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M")


def parse_args(argv):
    HOST = "localhost"
    PORT = 1111
    if len(argv) == 3:
        HOST = str(argv[1])
        PORT = int(argv[2])
    return HOST, PORT
