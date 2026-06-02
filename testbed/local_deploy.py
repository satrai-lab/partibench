import socket
import json
import sys
import os
import core
import psutil
import multiprocessing
import mxnet as mx
from mxnet.gluon import nn
import gluoncv
import time
import numpy as np
import pickle

# Paths are resolved relative to this file so the container can be started
# from any working directory (e.g. after `cd /output` for benchmarking).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


# Helper to free a socket
def close_socket(s):
    if s is None:
        return
    s.close()


# Receives text data on recv_port, reading until the sender closes the connection.
# Used for deployment info (place.py closes the socket after sending).
# For the first call client_socket is None and a new server socket is created.
def receive_text(recv_port, server_socket, client_socket):

    if client_socket is None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', recv_port))
        server_socket.listen()
        client_socket, _ = server_socket.accept()

    chunks = []
    while True:
        chunk = client_socket.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks).decode()

    return server_socket, client_socket, data


# Receives a single short text message on a persistent (non-closing) socket.
# Used by the PRODUCER to receive inference results from the final BRIDGE.
def receive_result(recv_port, server_socket, client_socket):

    if client_socket is None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', recv_port))
        server_socket.listen()
        client_socket, _ = server_socket.accept()

    data = client_socket.recv(1024).decode()

    return server_socket, client_socket, data


# Implements BLOCK component that accepts input, performs inference and sends output
def block_component(block_bytes, assym_pad, block_order, recv_port, dest_ip, dest_port):

    block_obj = pickle.loads(block_bytes)
    del block_bytes
    str_pid = str(os.getpid())
    print("Started BLOCK process (" + str_pid + ")", flush=True)
    server_socket_at_recv = None
    client_sockets_at_recv = None
    client_socket_at_send = None
    inference_execs = 0

    try:

        while True:

            print("BLOCK " + str_pid + " waiting for input on port " + str(recv_port), flush=True)
            server_socket_at_recv, client_sockets_at_recv, inp = core.receive_input(
                units=1, port=recv_port,
                server_socket=server_socket_at_recv, client_socket_array=client_sockets_at_recv,
                sim_delay=True)
            print("BLOCK " + str_pid + " received input", flush=True)
            mx.nd.waitall()
            st = time.time()

            outp = core.manual_inference(flattened_model=block_obj, inp=inp[0], assym_pad=assym_pad)
            outp_bytes = outp.asnumpy().tobytes()
            mx.nd.waitall()
            end = time.time()

            print("~>~<~>~<~> BLOCK TIME", (end - st) * 1000)

            client_socket_at_send = core.send_output(
                data_in_bytes=outp_bytes, shape=outp.shape, merge_order=block_order,
                dest_ip=dest_ip, port=dest_port, client_socket=client_socket_at_send,
                sim_delay=True)

            inference_execs += 1
            print("BLOCK " + str_pid + " executed inference! Current executions: " + str(inference_execs), flush=True)

    except KeyboardInterrupt:
        print("Exiting BLOCK (" + str_pid + ") and closing sockets...")

    finally:
        close_socket(server_socket_at_recv)
        if client_sockets_at_recv:
            for tup in client_sockets_at_recv:
                close_socket(tup[0])
        close_socket(client_socket_at_send)


# Implements BRIDGE component that accepts input(s), merges if needed, splits if needed and sends new input(s)
def bridge_component(inp_units, merge_type, split_ratios, split_offsets, flatten, recv_port, dest_ips, dest_port, send_results):

    str_pid = str(os.getpid())
    print("Started BRIDGE process (" + str_pid + ")", flush=True)
    server_socket_at_recv = None
    client_sockets_at_recv = None
    client_sockets_at_send = [None for _ in range(len(dest_ips))]
    client_socket_at_res_send = None
    inference_execs = 0

    if merge_type == 'filter':
        merge_dim = 1
    elif merge_type == 'width':
        merge_dim = 3
    elif merge_type is None:
        assert inp_units == 1
    else:
        assert merge_type == 'aggregation'

    assert len(split_ratios) == len(dest_ips)

    categories = None

    try:

        while True:

            print("BRIDGE " + str_pid + " waiting for " + str(inp_units) + " input(s) on port " + str(recv_port), flush=True)
            server_socket_at_recv, client_sockets_at_recv, inputs = core.receive_input(
                units=inp_units, port=recv_port,
                server_socket=server_socket_at_recv, client_socket_array=client_sockets_at_recv)
            print("BRIDGE " + str_pid + " received input(s)", flush=True)

            if inp_units > 1:
                if merge_type != 'aggregation':
                    new_data = mx.nd.concat(*inputs, dim=merge_dim)
                else:
                    new_data = inputs[0]
                    for i in inputs[1:]:
                        new_data += i
                    new_data = nn.Activation('relu')(new_data)
            elif inp_units == 1:
                new_data = inputs[0]
            else:
                assert False

            if send_results == "yes":
                results = ""

                if categories is None:
                    labels_path = os.path.join(ASSETS_DIR, 'image_net_labels.json')
                    categories = np.array(json.load(open(labels_path, 'r')))
                    assert client_socket_at_res_send is None
                    client_socket_at_res_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket_at_res_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    client_socket_at_res_send.connect((dest_ips[0], dest_port))

                pred = new_data.softmax()
                top_pred = pred.topk(k=3)[0].asnumpy()
                for index in top_pred:
                    probability = pred[0][int(index)]
                    category = categories[int(index)]
                    results += "{}: {:.2f}%\n".format(category, probability.asscalar() * 100)

                client_socket_at_res_send.sendall(results.encode('utf-8'))

            else:

                if flatten == 'yes' and len(new_data.shape) != 2:
                    new_data = nn.Flatten()(new_data)

                len_split_ratios = len(split_ratios)
                if len_split_ratios == 1:
                    assert split_ratios[0] == 1
                    client_sockets_at_send[0] = core.send_output(
                        data_in_bytes=new_data.asnumpy().tobytes(), shape=new_data.shape,
                        merge_order=0, dest_ip=dest_ips[0], port=dest_port,
                        client_socket=client_sockets_at_send[0])
                else:
                    if all(r == 1 for r in split_ratios):
                        for i in range(len_split_ratios):
                            client_sockets_at_send[i] = core.send_output(
                                data_in_bytes=new_data.asnumpy().tobytes(), shape=new_data.shape,
                                merge_order=0, dest_ip=dest_ips[i], port=dest_port,
                                client_socket=client_sockets_at_send[i])
                    else:
                        if split_offsets is not None:
                            assert len(split_offsets) == len_split_ratios
                            assert len(new_data.shape) == 4
                            for i in range(len(split_offsets)):
                                split_data = new_data[:, :, :, split_offsets[i][0]:split_offsets[i][1]]
                                client_sockets_at_send[i] = core.send_output(
                                    data_in_bytes=split_data.asnumpy().tobytes(), shape=split_data.shape,
                                    merge_order=0, dest_ip=dest_ips[i], port=dest_port,
                                    client_socket=client_sockets_at_send[i])
                        else:
                            assert len(new_data.shape) == 2
                            split_start = 0
                            dim_size = new_data.shape[1]
                            for i in range(len_split_ratios):
                                if i < len_split_ratios - 1:
                                    split_end = split_start + round(split_ratios[i] * dim_size)
                                else:
                                    split_end = dim_size
                                split_data = new_data[:, split_start:split_end]
                                client_sockets_at_send[i] = core.send_output(
                                    data_in_bytes=split_data.asnumpy().tobytes(), shape=split_data.shape,
                                    merge_order=0, dest_ip=dest_ips[i], port=dest_port,
                                    client_socket=client_sockets_at_send[i])
                                split_start = split_end

            inference_execs += 1
            print("BRIDGE " + str_pid + " executed inference! Current executions: " + str(inference_execs), flush=True)

    except KeyboardInterrupt:
        print("Exiting BRIDGE (" + str_pid + ") and closing sockets...")

    finally:
        close_socket(server_socket_at_recv)
        if client_sockets_at_recv:
            for tup in client_sockets_at_recv:
                close_socket(tup[0])
        if client_sockets_at_send:
            for s in client_sockets_at_send:
                close_socket(s)
        close_socket(client_socket_at_res_send)


# Implements PRODUCER component that initiates inference and receives results
def producer_component(dest_ip, dest_port, recv_port, infq):

    str_pid = str(os.getpid())
    print("Started PRODUCER process (" + str_pid + ")", flush=True)
    client_socket_at_send = None
    server_socket_at_res_recv = None
    client_socket_at_res_recv = None
    inference_execs = 0

    # Load and transform input image
    img = mx.image.imread(os.path.join(ASSETS_DIR, 'ship.png'))
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    transformed_img_to_bytes = transformed_img.asnumpy().tobytes()

    try:

        while True:

            while True:
                if not infq.empty():
                    if infq.get() == "s":
                        break

            print("PRODUCER: Inference started!", flush=True)

            start_time = time.time()
            print("PRODUCER " + str_pid + " sending to " + str(dest_ip) + ":" + str(dest_port), flush=True)
            client_socket_at_send = core.send_output(
                data_in_bytes=transformed_img_to_bytes, shape=transformed_img.shape,
                merge_order=0, dest_ip=dest_ip, port=dest_port, client_socket=client_socket_at_send)
            server_socket_at_res_recv, client_socket_at_res_recv, results = receive_result(
                recv_port=recv_port, server_socket=server_socket_at_res_recv,
                client_socket=client_socket_at_res_recv)
            mx.nd.waitall()
            elapsed_time = (time.time() - start_time) * 1000

            inference_execs += 1
            print("\nPRODUCER " + str_pid + " executed inference! Current executions: " + str(inference_execs))
            print("PRODUCER: Inference took: " + str(elapsed_time) + " ms")
            print("PRODUCER: Results:\n" + results)

    except KeyboardInterrupt:
        print("Exiting PRODUCER (" + str_pid + ") and closing sockets...")

    finally:
        close_socket(client_socket_at_send)
        close_socket(server_socket_at_res_recv)
        close_socket(client_socket_at_res_recv)


# Only in parent process
if __name__ == "__main__":

    producer_deployed = False


    def deploy_components(C, infq):

        global producer_deployed
        producer_deployed = False  # reset on each deployment

        model_name = C.get("model")
        model = None
        components = C.get("components")

        print("Number of components: " + str(len(components)))
        print(components)

        for c in components:

            component_type = c.get("type")
            process = None

            if component_type == "block":

                if model is None:
                    model = core.load_model(model_name)

                block_obj = core.generate_model_partition(
                    flattened_model=model,
                    from_layer=c.get("from_layer"), to_layer=c.get("to_layer"),
                    from_filter=c.get("from_filter"), to_filter=c.get("to_filter"),
                    from_weight=c.get("from_weight"), to_weight=c.get("to_weight"))

                block_obj = pickle.dumps(block_obj, protocol=pickle.HIGHEST_PROTOCOL)

                process = multiprocessing.Process(target=block_component, args=(
                    block_obj,
                    c.get("assym_pad"),
                    c.get("block_order"),
                    c.get("recv_port"),
                    c.get("dest_ip"),
                    c.get("dest_port")))

            elif component_type == "bridge":

                process = multiprocessing.Process(target=bridge_component, args=(
                    c.get("inp_units"),
                    c.get("merge_type"),
                    c.get("split_ratio"),
                    c.get("split_offsets"),
                    c.get("flatten"),
                    c.get("recv_port"),
                    c.get("dest_ips"),
                    c.get("dest_port"),
                    c.get("send_results")))

            elif component_type == "producer":

                producer_deployed = True

                process = multiprocessing.Process(target=producer_component, args=(
                    c.get("dest_ip"),
                    c.get("dest_port"),
                    c.get("recv_port"),
                    infq))

            else:
                assert False

            process.start()
            print("Started child process " + str(process.pid) + " for component type " + str(component_type), flush=True)


    def terminate_old_components(parent_process, pid_to_ignore):

        children = parent_process.children(recursive=True)

        if len(children) > 1:
            print("\nTerminating deployed component processes. Re-deploying...\n")

            children_to_terminate = []
            for child in children:
                if child.pid != pid_to_ignore:
                    children_to_terminate.append(child)
            assert children_to_terminate

            for child in children_to_terminate:
                child.terminate()

            gone, alive = psutil.wait_procs(children_to_terminate, timeout=5)

            for child in alive:
                child.kill()


    # ==== MAIN ====

    multiprocessing.set_start_method('spawn', force=True)
    inference_queue = multiprocessing.Queue()

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    assert len(children) == 1  # only the queue process
    queue_pid = children[0].pid

    try:

        while True:

            print("Waiting for component deployment info...", flush=True)

            ss = cs = None
            ss, cs, c_info = receive_text(recv_port=6999, server_socket=None, client_socket=None)
            close_socket(ss)
            close_socket(cs)

            terminate_old_components(parent_process=parent, pid_to_ignore=queue_pid)
            deploy_components(json.loads(c_info), inference_queue)

            # Auto-queue 16 inference runs (5 s apart) when this node hosts the producer.
            # No interactive stdin — the container has no TTY in Kubernetes.
            total_runs = 0
            while producer_deployed and total_runs < 16:
                print("LOCAL_DEPLOY: queueing inference run #" + str(total_runs + 1), flush=True)
                inference_queue.put("s")
                total_runs += 1
                time.sleep(5)

    except KeyboardInterrupt:
        print("Exiting LOCAL_DEPLOY (parent process) and closing sockets...")

    finally:
        close_socket(ss)
        close_socket(cs)
