import socket
import json
import sys
import core
import os
import psutil
import multiprocessing
import mxnet as mx
from mxnet.gluon import nn
import gluoncv
import time
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


# Helper to free a socket
def close_socket(s):
    if s is None:
        return
    s.close()


# Using socket programming, this function receives text data in the specified port
# IMPORTANT: No net delay is assumed in current implementation
def receive_text(num_bytes, recv_port, server_socket, client_socket):

    # Init receiving socket (only first time) and accept a connection from client
    if client_socket is None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', recv_port))
        server_socket.listen()
        client_socket, _ = server_socket.accept()

    chunks = []
    while True:
        chunk = client_socket.recv(num_bytes)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks).decode()

    return server_socket, client_socket, data


# Implements BLOCK component that accepts input, performs inference and sends output
def block_component(block_bytes, assym_pad, block_order, recv_port, dest_ip, dest_port):

    block_obj = pickle.loads(block_bytes)
    del block_bytes
    str_pid = str(os.getpid())
    print("Started BLOCK process (" + str_pid + ")")
    server_socket_at_recv = None  # socket that accepts inference data for block
    client_sockets_at_recv = None  # socket (array of size 1) of client that sends inference data for block
    client_socket_at_send = None  # socket that sends block's output
    inference_execs = 0

    try:

        while True:

            print("BLOCK " + str_pid + " waiting for input on port " + str(recv_port), flush=True)
            server_socket_at_recv, client_sockets_at_recv, inp = core.receive_input(units=1, port=recv_port, 
                                                server_socket=server_socket_at_recv, client_socket_array=client_sockets_at_recv)
            print("BLOCK " + str_pid + " received input on port " + str(recv_port), flush=True)
            mx.nd.waitall()
            st = time.time()

            outp = core.manual_inference(flattened_model=block_obj, inp=inp[0], assym_pad=assym_pad)
            outp_bytes = outp.asnumpy().tobytes()
            mx.nd.waitall()
            end = time.time()

            print("~>~<~>~<~> BLOCK TIME", (end-st)*1000)

            client_socket_at_send = core.send_output(data_in_bytes=outp_bytes, shape=outp.shape, merge_order=block_order, dest_ip=dest_ip, port=dest_port, sim_delay=True, client_socket=client_socket_at_send)

            inference_execs += 1
            print("BLOCK " + str_pid + " executed inference! Current executions: " + str(inference_execs))

    except KeyboardInterrupt:
        print("Exiting BLOCK (" + str_pid + ") and closing sockets...")
        
    finally:

        # Free sockets
        close_socket(server_socket_at_recv)
        if client_sockets_at_recv:
            for tup in client_sockets_at_recv:
                close_socket(tup[0])
        close_socket(client_socket_at_send)


# Implements BRIDGE component that accepts input(s), merges if needed, splits if needed and sends new input(s)
# IMPORTANT: BRIDGE does not add network delays. These are simulated by communicating BLOCK components
def bridge_component(inp_units, merge_type, split_ratios, split_offsets, flatten, recv_port, dest_ips, dest_port, send_results):

    str_pid = str(os.getpid())
    print("Started BRIDGE process (" + str_pid + ")")
    server_socket_at_recv = None  # socket that accepts data to merge
    client_sockets_at_recv = None  # sockets of clients that send data to merge
    client_sockets_at_send = [None for _ in range(len(dest_ips))]  # sockets that send split data
    client_socket_at_res_send = None  # socket that sends results back to start node
    inference_execs = 0

    # How to merge (if needed) received data?
    if merge_type == 'filter':
        merge_dim = 1
    elif merge_type == 'width':
        merge_dim = 3
    elif merge_type == None:
        assert inp_units == 1
    else:
        assert merge_type == 'aggregation'

    assert len(split_ratios) == len(dest_ips)  # just to be sure

    categories = None  # makes sense only for last bridge

    try:

        while True:

            # Receive inputs and merge if needed
            print("BRIDGE " + str_pid + " waiting for " + str(inp_units) + " input(s) on port " + str(recv_port), flush=True)
            server_socket_at_recv, client_sockets_at_recv, inputs = core.receive_input(units=inp_units, port=recv_port,
                                server_socket=server_socket_at_recv, client_socket_array=client_sockets_at_recv)
            print("BRIDGE " + str_pid + " received input(s) on port " + str(recv_port), flush=True)

            if inp_units > 1:  # merging needed
                if merge_type != 'aggregation':  # normal merging (concat)
                    new_data = mx.nd.concat(*inputs, dim=merge_dim)
                else:  # aggregation (previous block was an input split FC block)
                    new_data = inputs[0]
                    for i in inputs[1:]:
                        new_data += i
                    # TODO: HOW TO KNOW WHICH ACTIVATION??????
                    new_data = nn.Activation('relu')(new_data)  # TODO: REMOVE THIS, IT IS NOT A GENERIC SOLUTION!
            elif inp_units == 1:  # merging not needed (single input received)
                new_data = inputs[0]
            else:
                assert False

            # TODO REMOVE
            #if inp_units > 1:
            #print(">>>>> TIME AFTER MERGE: ", time.time())

            # If this is the last bridge component, compute and send results back to start node
            if send_results == "yes":
                results = ""

                # Load dataset labels and init connection to send results to start node
                if categories is None:
                    labels_path = os.path.join(ASSETS_DIR, 'image_net_labels.json')
                    categories = np.array(json.load(open(labels_path, 'r')))
                    assert client_socket_at_res_send is None
                    client_socket_at_res_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket_at_res_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    client_socket_at_res_send.connect((dest_ips[0], dest_port))

                # Transform scores to probabilities and keep top 3
                pred = new_data.softmax()
                top_pred = pred.topk(k=3)[0].asnumpy()
                for index in top_pred:
                    probability = pred[0][int(index)]
                    category = categories[int(index)]
                    results += "{}: {:.2f}%\n".format(category, probability.asscalar() * 100)

                # Finally, send results to start node
                client_socket_at_res_send.sendall(results.encode('utf-8'))

            else:

                # If received inputs are still feature maps and next block is an input split FC --> flatten data fist
                if flatten == 'yes' and len(new_data.shape) != 2:
                    new_data = nn.Flatten()(new_data)

                # Split data if needed and send to destinations

                # print("--- TIME BEFORE SEND: ", time.time())  # TODO REMOVE

                len_split_ratios = len(split_ratios)
                if len_split_ratios == 1:  # no split needed, send to a single destination
                    assert split_ratios[0] == 1
                    client_sockets_at_send[0] = core.send_output(data_in_bytes=new_data.asnumpy().tobytes(),
                             shape=new_data.shape, merge_order=0, dest_ip=dest_ips[0], port=dest_port, client_socket=client_sockets_at_send[0])
                else:
                    if all(r == 1 for r in split_ratios):  # no data split needed, send to multiple destinations (filter split)
                        for i in range(len_split_ratios):
                            client_sockets_at_send[i] = core.send_output(data_in_bytes=new_data.asnumpy().tobytes(),
                             shape=new_data.shape, merge_order=0, dest_ip=dest_ips[i], port=dest_port, client_socket=client_sockets_at_send[i])

                    else:  # split needed
                        if split_offsets is not None:  # split data for non-Dense blocks (input split)
                            assert len(split_offsets) == len_split_ratios
                            assert len(new_data.shape) == 4
                            for i in range(len(split_offsets)):
                                split_data = new_data[:, :, :, split_offsets[i][0]:split_offsets[i][1]]
                                client_sockets_at_send[i] = core.send_output(data_in_bytes=split_data.asnumpy().tobytes(), shape=split_data.shape,
                                 merge_order=0, dest_ip=dest_ips[i], port=dest_port, client_socket=client_sockets_at_send[i])
                        else:  # split data for Dense blocks (input split)
                            assert len(new_data.shape) == 2
                            split_start = 0
                            dim_size = new_data.shape[1]
                            for i in range(len_split_ratios):

                                # Find split end-limit
                                if i < len_split_ratios - 1:  # not the last split
                                    split_end = split_start + round(split_ratios[i] * dim_size)
                                else:  # last split
                                    split_end = dim_size

                                split_data = new_data[:, split_start:split_end]
                                client_sockets_at_send[i] = core.send_output(data_in_bytes=split_data.asnumpy().tobytes(), 
                                 shape=split_data.shape, merge_order=0, dest_ip=dest_ips[i], port=dest_port, client_socket=client_sockets_at_send[i])
                                split_start = split_end

            inference_execs += 1
            print("BRIDGE " + str_pid + " executed inference! Current executions: " + str(inference_execs))

    except KeyboardInterrupt:
        print("Exiting BRIDGE (" + str_pid + ") and closing sockets...")

    finally:

        # Free sockets
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
    print("Started PRODUCER process (" + str_pid + ")")
    client_socket_at_send = None  # socket that sends initial inference data
    server_socket_at_res_recv = None  # socket that receives results
    client_socket_at_res_recv = None  # socket of client that sends results
    inference_execs = 0

    # Load and transform input image
    img = mx.image.imread(os.path.join(ASSETS_DIR, 'ship.png'))
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    transformed_img_to_bytes = transformed_img.asnumpy().tobytes()

    try:

        # Init inference manually (when reading a start signal from queue)
        while True:

            while True:
                if not infq.empty():  # Check if there's a start signal from the queue
                    if infq.get() == "s":
                        break

            print("PRODUCER: Inference started!")

            start_time = time.time()
            print("~~~~ STAAAART TIME: ", start_time) # TODO REMOVE
            print("PRODUCER " + str_pid + " sending first input to " + str(dest_ip) + ":" + str(dest_port), flush=True)
            client_socket_at_send = core.send_output(data_in_bytes=transformed_img_to_bytes, shape=transformed_img.shape, merge_order=0, dest_ip=dest_ip, port=dest_port, sim_delay=False, client_socket=client_socket_at_send)
            print("PRODUCER " + str_pid + " sent first input to " + str(dest_ip) + ":" + str(dest_port), flush=True)
            server_socket_at_res_recv, client_socket_at_res_recv, results = receive_text(num_bytes=1024, recv_port=recv_port, server_socket=server_socket_at_res_recv, client_socket=client_socket_at_res_recv)
            mx.nd.waitall()
            elapsed_time = (time.time() - start_time) * 1000

            inference_execs += 1
            print("\nPRODUCER " + str_pid + " executed inference! Current executions: " + str(inference_execs))
            print("PRODUCER: Inference took: " + str(elapsed_time) + " ms")
            print("PRODUCER: Results:\n" + results)

    except KeyboardInterrupt:
        print("Exiting PRODUCER (" + str_pid + ") and closing sockets...")

    finally:

        # Free sockets
        close_socket(client_socket_at_send)
        close_socket(server_socket_at_res_recv)
        close_socket(client_socket_at_res_recv)


# Only in parent process
if __name__ == "__main__":

    producer_deployed = False  # does this script have the producer deployed?


    # Reads current node's component info and starts child processes to actually deploy them
    # Reads inference queue which transfers signals (to producer component) to initiate inference
    def deploy_components(C, infq):

        global producer_deployed
        producer_deployed = False

        model_name = C.get("model")
        model = None
        components = C.get("components")

        print("Number of components: " + str(len(components)))
        print(components)

        for c in components:

            component_type = c.get("type")
            process = None

            # Prepare child process' command depending on component type

            if component_type == "block":

                # Load model (only first time entering here)
                if model is None:
                    model = core.load_model(model_name)

                # Create block object
                block_obj = core.generate_model_partition(
                    flattened_model=model,
                    from_layer=c.get("from_layer"), to_layer=c.get("to_layer"),
                    from_filter=c.get("from_filter"), to_filter=c.get("to_filter"),
                    from_weight=c.get("from_weight"), to_weight=c.get("to_weight")
                    )

                block_obj = pickle.dumps(block_obj, protocol=pickle.HIGHEST_PROTOCOL)  # serialize it

                process = multiprocessing.Process(target=block_component, args=(
                    block_obj,
                    c.get("assym_pad"),
                    c.get("block_order"),
                    c.get("recv_port"),
                    c.get("dest_ip"),
                    c.get("dest_port")
                    )
                )

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
                    c.get("send_results")
                    )
                )

            elif component_type == "producer":

                producer_deployed = True

                process = multiprocessing.Process(target=producer_component, args=(
                    c.get("dest_ip"),
                    c.get("dest_port"),
                    c.get("recv_port"),
                    infq
                    )
                )
               
            else:
                assert False  # incorrect type

            # Start process for current component
            process.start()
            print("Started child process " + str(process.pid) + " for component type " + str(component_type), flush=True)


    # Terminates already deployed component processes except the process with the pid to ignore
    def terminate_old_components(parent_process, pid_to_ignore):

        # Get all child processes
        children = parent_process.children(recursive=True)

        if len(children) > 1:
            print("\nTerminating deployed component processes. Re-deploying...\n")

            # Keep all children except the one with pid == pid_to_ignore
            children_to_terminate = []
            for child in children:
                if child.pid != pid_to_ignore:
                    children_to_terminate.append(child)
            assert children_to_terminate

            # Gracefully terminate children
            for child in children_to_terminate:
                child.terminate()

            # Wait for them to terminate
            gone, alive = psutil.wait_procs(children_to_terminate, timeout=5)

            # Force kill any processes that didn't terminate after 5 secs
            for child in alive:
                child.kill()


    # ==== MAIN ====

    multiprocessing.set_start_method('spawn', force=True)  # open child processes separate from the parent
    inference_queue = multiprocessing.Queue()  # will store "start inference" signals

    parent = psutil.Process(os.getpid())  # parent process
    children = parent.children(recursive=True)  # children processes 
    assert len(children) == 1  # ONLY queue at this moment
    queue_pid = children[0].pid  # keep queue process pid to ignore terminating it later 

    try:

        while True:

            print("Waiting for component deployment info...")

            ss = cs = None
            ss, cs, c_info = receive_text(num_bytes=1048576, recv_port=6999, server_socket=None, client_socket=None)  # receive component info
            close_socket(ss)
            close_socket(cs)

            terminate_old_components(parent_process=parent, pid_to_ignore=queue_pid)  # in case of re-deployment
            deploy_components(json.loads(c_info), inference_queue)  # deploy processes

            # If this script deployed a producer, wait for start inference signals
            i = None
            total_runs = 0
            while producer_deployed and total_runs < 16:
                print("LOCAL_DEPLOY parent queueing inference start signal #" + str(total_runs + 1), flush=True)
                inference_queue.put("s")
                total_runs += 1
                time.sleep(5)


    except KeyboardInterrupt:
        print("Exiting LOCAL_DEPLOY (parent process) and closing sockets...")

    finally:

        # Free sockets
        close_socket(ss)
        close_socket(cs)
