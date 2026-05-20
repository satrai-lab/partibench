import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psutil
import pickle
import mxnet as mx
import core

assert len(sys.argv) == 3  # script name, block id, input shape
block_id = sys.argv[1]
input_shape = eval(sys.argv[2])  # note: use eval to transform to tuple


# Returns current process' memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # physical RAM consumption in MB


# Loads and returns a layer block object from a binary file, using 'pickle' module
def load_layer_block(file_path):
    with open(file_path, 'rb') as infile:
        block_object = pickle.load(infile)
    return block_object


# ---- MAIN ----

# Load layer block
fpath = "block_" + block_id + ".lb"
block = load_layer_block(fpath)

# Create dummy input
inp = mx.nd.random.uniform(low=0, high=1, shape=input_shape)

# Run inference
outp = core.manual_inference(block, inp)
mx.nd.waitall()

# Measure memory usage
print('{"mem":' + str(get_memory_usage()) + '}')  # TODO IS THIS THE CORRECT WAY ? maybe increase measurement granul. -> measure inside manual inference funct.
