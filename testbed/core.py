import mxnet as mx
from mxnet.gluon import nn
import gluoncv
import numpy as np
import math
from mxnet.gluon.model_zoo import vision
import socket
import select
import struct
import sys
import pickle
import os
import time
import csv

# Connections are retried because pods may not all be ready at the same instant.
CONNECT_RETRY_DELAY_SEC = 3.0
CONNECT_MAX_RETRIES = 60

# Read delays from file (to simulate net delays programmatically) and store them in a dict.
# In the testbed the file exists but has no data rows, so delays_dict stays empty
# and no delays are ever simulated.
delays_dict = {}
with open('net_rules.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)  # skip header
    for row in reader:
        remote_ip = row[1]
        one_way_delay_in_sec = int(row[3][:-2]) / 1000
        delays_dict[remote_ip] = one_way_delay_in_sec


# Helper function and global var that allows reading the next positional argument everytime
arg_num = 0
def read_next_arg():
    global arg_num
    arg_num += 1
    return sys.argv[arg_num]


# Helper function that receives mxnet data from a single socket
# If a non-zero delay is given, add it after receive to simulate network delay
def receive_data_from_socket(sock, delay):

    # Read metadata (first 24 bytes) and extract it
    metadata_chunks = []
    md_size = 24
    TOTAL_RECEIVED = 0
    while TOTAL_RECEIVED < md_size:
        chunk = sock.recv(md_size - TOTAL_RECEIVED)
        TOTAL_RECEIVED += len(chunk)
        metadata_chunks.append(chunk)
    metadata = b"".join(metadata_chunks)
    *array_shape, merge_order, payload_size = struct.unpack('!6I', metadata)
    array_shape = tuple(array_shape)
    if array_shape[-1] == 0:
        array_shape = (array_shape[0], array_shape[1])

    # Receive actual data in chunks
    payload_chunks = []
    TOTAL_RECEIVED = 0
    while TOTAL_RECEIVED < payload_size:
        chunk = sock.recv(payload_size - TOTAL_RECEIVED)
        TOTAL_RECEIVED += len(chunk)
        payload_chunks.append(chunk)
    payload = b"".join(payload_chunks)

    # Simulate possible network delay
    if delay:
        time.sleep(delay)

    # Convert to mxnet nd array format
    convert_data = mx.nd.array(np.frombuffer(payload, dtype=np.float32)).reshape(array_shape)

    return merge_order, convert_data


# Receive mxnet inputs from multiple destinations
def receive_input(units, port, server_socket, client_socket_array, sim_delay=False):

    global delays_dict

    # Init received inputs holder
    inp = [None for _ in range(units)]

    if client_socket_array is not None:  # connections and sockets previously initialized -> just use them

        # Extract sockets from (socket, ip) tuples of client_socket_array
        only_sockets = []
        for tup in client_socket_array:
            only_sockets.append(tup[0])

        # Read data from all sources
        while any(v is None for v in inp):

            # At this moment, which sockets have available data?
            readable, _, _ = select.select(only_sockets, [], [], None)  # no timeout

            # Sockets that have some available data
            for sock in readable:
                if sim_delay:
                    assert units == 1
                    d = delays_dict.get(client_socket_array[0][1])
                else:
                    d = None
                merge_order, data = receive_data_from_socket(sock, d)
                inp[merge_order] = data

    else:  # initialize connections and sockets -- DO NOT simulate delay on init

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 25 * 1024 * 1024)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(units - 1)
        client_socket_array = []

        for i in range(units):
            sock, addr = server_socket.accept()
            client_socket_array.append((sock, addr[0]))
            merge_order, data = receive_data_from_socket(sock, None)
            inp[merge_order] = data

    return server_socket, client_socket_array, inp


# Send an mxnet output matrix, already converted to bytes.
# Retries the initial connection to tolerate pods that are not yet ready.
def send_output(data_in_bytes, shape, merge_order, dest_ip, port, client_socket, sim_delay=False):

    global delays_dict

    # Init socket (the first time only), with retry logic for K8s pod readiness
    if client_socket is None:
        last_error = None
        for attempt in range(CONNECT_MAX_RETRIES):
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 13 * 1024 * 1024)
            try:
                time.sleep(CONNECT_RETRY_DELAY_SEC)
                client_socket.connect((dest_ip, port))
                break
            except ConnectionRefusedError as exc:
                last_error = exc
                client_socket.close()
                client_socket = None
                if attempt == CONNECT_MAX_RETRIES - 1:
                    raise
            except Exception:
                client_socket.close()
                client_socket = None
                raise
        if client_socket is None and last_error is not None:
            raise last_error

    # Convert metadata (array shape, merge_order and payload size) to bytes
    if len(shape) == 2:
        shape = (*shape, 0, 0)
    assert len(shape) == 4
    metadata_to_bytes = struct.pack('!6I', *shape, merge_order, len(data_in_bytes))

    mx.nd.waitall()
    if sim_delay:
        d = delays_dict.get(dest_ip)
        if d:
            time.sleep(d)

    dts = metadata_to_bytes + data_in_bytes
    client_socket.sendall(dts)

    return client_socket


# Receives a layer and returns whether it has a kernel, a stride and a padding attribute or not
def layer_has_ksp(l):
    return (hasattr(l, '_kwargs') and
            all(key in l._kwargs for key in ('kernel', 'pad', 'stride')))


# Compute a layer's input dimension based on its output (considering asymmetrical padding)
def layer_input_dim(output_dim, kernel, padding, stride):
    return stride * output_dim - stride + kernel - 1 * padding


# Receives a reversed flattened model (or set of consecutive layers), its output size and a list of 2 ratios and
# returns the sizes of the two input splits (based on ratios) if the model was to be executed using data partitioning
def split_input_sizes(reversed_model, outp, ratios):

    assert len(ratios) == 2
    sum_ratios = sum(ratios)
    right_padding_exists = False
    if sum_ratios == 1:
        right_padding_exists = True  # current right split is the rightmost split

    # Calculate the size of two output splits according to given ratio
    left_output = round(outp * ratios[0])
    right_output = round(outp * sum_ratios) - left_output

    # Compute the model's split input sizes based on their output sizes for every layer, starting from the last one
    for layer_tuple in reversed_model:
        layer = layer_tuple[0]
        if layer_has_ksp(layer):
            assert len(layer_tuple) == 4  # (layer, padding, input, output)
            kernel = layer._kwargs['kernel'][0]
            stride = layer._kwargs['stride'][0]
            padding = layer_tuple[1]

            # Left split
            left_output = layer_input_dim(left_output, kernel, padding, stride)

            # Right split
            if not right_padding_exists:
                padding = 0
                fract = 0
            else:
                expected_output = layer_tuple[3]
                fract = round(expected_output % 1, 1)
            right_output = layer_input_dim(right_output + fract, kernel, padding, stride)

    assert left_output % 1 == 0 and right_output % 1 == 0
    return int(left_output), int(right_output)


# Receives a flattened model (or a set of consecutive layers) and a list of N ratios and
# returns a list of N split offsets for splitting the model's input (for data partitioning) according to ratios
def split_input_offsets(flattened_model, ratios):

    assert sum(ratios) == 1
    num_splits = len(ratios)
    assert num_splits > 1
    offsets = [None] * num_splits
    reversed_model = flattened_model[::-1]
    last_layer_output = int(flattened_model[-1][3])

    for i in range(1, num_splits):
        r = [sum(ratios[:-i])] + ratios[-i:]
        r_first_two = r[:2]

        left, right = split_input_sizes(reversed_model, last_layer_output, r_first_two)

        if i == 1:
            offsets[num_splits-i] = (-right, None)
        else:
            offsets[num_splits-i] = (prev_left - right, prev_left)
        prev_left = left
    offsets[0] = (0, prev_left)

    return offsets


# Receives a layer and constructs a layer tuple to be inserted in a flattened model
def construct_layer_tuple(l, expected_input_size):
    padding = None
    if isinstance(l, nn.Flatten) or isinstance(l, nn.Dense):
        output_size = 1
    elif layer_has_ksp(l):
        kernel = l._kwargs['kernel'][0]
        stride = l._kwargs['stride'][0]
        padding = l._kwargs['pad'][0]

        # Remove padding; it will be added manually at inference
        l._kwargs['pad'] = (0, 0)

        output_size = 1 + (expected_input_size + 2 * padding - kernel) / stride
    else:
        output_size = expected_input_size

    return l, padding, expected_input_size, output_size


# Loads and transforms MobilenetV2 into a (flat) list of
# consecutive (layer, padding, input, output) tuples and shortcut (add) operation points
def flatten_mobilenet(root='./mobilenet_data'):
    model = gluoncv.model_zoo.get_model('MobileNetV2_0.5', pretrained=True, root=root)
    expected_input_size = 224
    fl = []
    for f in model.features:
        if isinstance(f, gluoncv.model_zoo.mobilenet.LinearBottleneck):
            if f.use_shortcut:
                fl.append(('add_start', None, expected_input_size, expected_input_size))
            for l in f.out:
                layer_tuple = construct_layer_tuple(l, expected_input_size)
                fl.append(layer_tuple)
                expected_input_size = int(layer_tuple[3])
            if f.use_shortcut:
                fl.append(('add_end', None, expected_input_size, expected_input_size))
        else:
            layer_tuple = construct_layer_tuple(f, expected_input_size)
            fl.append(layer_tuple)
            expected_input_size = int(layer_tuple[3])
    for o in model.output:
        layer_tuple = construct_layer_tuple(o, expected_input_size)
        fl.append(layer_tuple)
        expected_input_size = int(layer_tuple[3])
    return fl


# Loads and transforms VGG16 into a (flat) list of
# consecutive (layer, padding, input, output) tuples
def flatten_vgg16(root='./vgg16_data'):
    model = gluoncv.model_zoo.get_model('VGG16', pretrained=True, root=root)
    expected_input_size = 224
    fl = []
    for f in model.features:
        if isinstance(f, nn.Dropout):
            continue
        layer_tuple = construct_layer_tuple(f, expected_input_size)
        fl.append(layer_tuple)
        expected_input_size = int(layer_tuple[3])
    fl.append(construct_layer_tuple(model.output, expected_input_size))
    return fl


# Loads and returns flattened model
def load_model(model_name):
    flat_path = model_name + '.flat'
    if os.path.exists(flat_path):
        with open(flat_path, 'rb') as infile:
            model = pickle.load(infile)
    else:
        if model_name == "vgg16":
            model = flatten_vgg16()
        with open(flat_path, 'wb') as outfile:
            pickle.dump(model, outfile)
    return model


# Perform manual inference (layer-by-layer)
def manual_inference(flattened_model, inp, assym_pad=None):
    block_inp = None

    for layer_tuple in flattened_model:
        layer = layer_tuple[0]
        if layer == 'add_start':
            block_inp = inp
        elif layer == 'add_end':
            inp_w = inp.shape[3]
            block_inp_w = block_inp.shape[3]
            w_dif = block_inp_w - inp_w
            assert w_dif >= 0
            if w_dif > 0:
                if assym_pad == 'left':
                    block_inp = block_inp[:, :, :, :-w_dif]
                elif assym_pad == 'right':
                    block_inp = block_inp[:, :, :, w_dif:]
            inp = mx.nd.elemwise_add(block_inp, inp)
        else:
            padding = layer_tuple[1]
            if padding not in (0, None):
                if assym_pad is None:
                    inp = mx.nd.pad(inp, mode='constant', constant_value=0,
                                    pad_width=(0, 0, 0, 0, padding, padding, padding, padding))
                elif assym_pad == 'left':
                    inp = mx.nd.pad(inp, mode='constant', constant_value=0,
                                    pad_width=(0, 0, 0, 0, padding, padding, padding, 0))
                elif assym_pad == 'right':
                    inp = mx.nd.pad(inp, mode='constant', constant_value=0,
                                    pad_width=(0, 0, 0, 0, padding, padding, 0, padding))
                elif assym_pad == 'center':
                    inp = mx.nd.pad(inp, mode='constant', constant_value=0,
                                    pad_width=(0, 0, 0, 0, padding, padding, 0, 0))
                else:
                    assert False
            inp = layer(inp)

    return inp


# Receives a convolutional layer and partitioning limits (in filters dimension) and generates
# a layer with the same configurations but only a partition of the filter parameters
def conv2d_partition(layer, from_filter, to_filter):
    layer_params = list(layer.params.values())
    kernel_data = layer_params[0].data()
    use_bias = not layer._kwargs['no_bias']

    act = None
    if "Activation(relu)" in str(layer):
        act = "relu"

    gen_layer = nn.Conv2D(channels=to_filter-from_filter,
                          kernel_size=layer._kwargs['kernel'],
                          strides=layer._kwargs['stride'],
                          padding=0,
                          dilation=layer._kwargs['dilate'],
                          groups=layer._kwargs['num_group'],
                          layout=layer._kwargs['layout'],
                          activation=act,
                          use_bias=use_bias,
                          in_channels=kernel_data.shape[1])

    gen_layer_params = list(gen_layer.params.values())
    gen_layer_params[0].initialize()
    gen_layer_params[0].set_data(
        kernel_data[from_filter:to_filter, :, :, :]
    )
    if use_bias:
        gen_layer_params[1].initialize()
        gen_layer_params[1].set_data(
            layer_params[1].data()[from_filter:to_filter]
        )

    return gen_layer


# Receives a batchnorm layer and partitioning limits (in filters dimension) and generates
# a layer with the same configurations but only a partition of the parameters in the filter dimension
def batchnorm_partition(layer, from_filter, to_filter):
    layer_params = list(layer.params.values())

    gen_layer = nn.BatchNorm(axis=layer._kwargs['axis'],
                             momentum=layer._kwargs['momentum'],
                             epsilon=layer._kwargs['eps'],
                             scale=not layer._kwargs['fix_gamma'],
                             use_global_stats=layer._kwargs['use_global_stats'],
                             in_channels=to_filter-from_filter)

    gen_layer_params = list(gen_layer.params.values())
    assert len(gen_layer_params) == len(layer_params)
    for i in range(len(gen_layer_params)):
        gen_layer_params[i].initialize()
        gen_layer_params[i].set_data(layer_params[i].data()[from_filter:to_filter])

    return gen_layer


# Receives a dense (fully-connected) layer and partitioning limits (in neurons/units dimension or in weights
# dimension) and generates a layer with the same configurations but only a partition of the neurons/weights
def dense_partition(layer, from_neuron=None, to_neuron=None, from_weight=None, to_weight=None):
    layer_params = list(layer.params.values())
    weights = layer_params[0].data()
    biases = layer_params[1].data()

    act = None
    if "Activation(relu)" in str(layer):
        act = "relu"

    if from_weight is None and to_weight is None:
        gen_layer = nn.Dense(units=to_neuron-from_neuron,
                             activation=act,
                             in_units=weights.shape[1])

        gen_layer_params = list(gen_layer.params.values())

        gen_layer_params[0].initialize()
        gen_layer_params[0].set_data(
            weights[from_neuron:to_neuron, :]
        )
        gen_layer_params[1].initialize()
        gen_layer_params[1].set_data(
            biases[from_neuron:to_neuron]
        )

    else:
        gen_layer = nn.Dense(units=weights.shape[0],
                             in_units=to_weight-from_weight)

        gen_layer_params = list(gen_layer.params.values())

        gen_layer_params[0].initialize()
        gen_layer_params[0].set_data(
            weights[:, from_weight:to_weight]
        )
        gen_layer_params[1].initialize()
        if from_weight == 0:
            gen_layer_params[1].set_data(biases)

    return gen_layer


# Receives a flattened model (list of layer/padding/in/out tuples),
# and returns only the tuples from 'from_layer' to 'to_layer'
# If 'from_filter' and 'to_filter' are given, it also performs further splitting in the filter dimension,
# according to the latter arguments
# If 'from_weight' and 'to_weight' are given (only for FC layers), it trims each neuron weights array instead
def generate_model_partition(flattened_model, from_layer, to_layer,
                             from_filter=None, to_filter=None,
                             from_weight=None, to_weight=None):

    m = flattened_model[from_layer:to_layer]

    balance = 0
    for layer_tuple in m:
        if layer_tuple[0] == 'add_start':
            balance += 1
        elif layer_tuple[0] == 'add_end':
            balance -= 1
        assert balance >= 0
    assert balance == 0

    ret = []
    for layer_tuple in m:
        layer = layer_tuple[0]

        if isinstance(layer, nn.Conv2D) or isinstance(layer, nn.BatchNorm) or (isinstance(layer, nn.Dense) and from_weight is None):
            if from_filter is None:
                assert to_filter is None
                ff = 0
                tf = list(layer.params.values())[0].shape[0]
            else:
                ff = from_filter
                tf = to_filter

        if isinstance(layer, nn.Conv2D):
            ret.append((conv2d_partition(layer, ff, tf),) + layer_tuple[1:])

        elif isinstance(layer, nn.BatchNorm):
            ret.append((batchnorm_partition(layer, ff, tf),) + layer_tuple[1:])

        elif isinstance(layer, nn.Dense):
            if from_weight is None and to_weight is None:
                ret.append(
                    (dense_partition(layer=layer, from_neuron=ff, to_neuron=tf),) + layer_tuple[1:]
                )
            else:
                ret.append(
                    (dense_partition(layer=layer, from_weight=from_weight, to_weight=to_weight),) + layer_tuple[1:]
                )

        else:
            ret.append(layer_tuple)

    return ret
