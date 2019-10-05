from . import utils
import pickle
from pickle import *
import os.path
import sys

sys.modules['utils'] = utils


n_bytes = 2**31
max_bytes = 2**31 - 1
data = bytearray(n_bytes)


def dump(data, f_out):
    bytes_out = pickle.dumps(data)
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

def load(f_in):
    bytes_in = bytearray(0)
    input_size = os.path.getsize(os.path.abspath(f_in.name))
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
    data2 = pickle.loads(bytes_in)
    return data2
