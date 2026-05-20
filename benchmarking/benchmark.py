import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import pickle
import mxnet as mx
import multiprocessing


# This function is called as a separate process to measure the memory consumption of an
# inference operation of a layer block (de-serialized from block_bytes) using a dummy input of size inp_shape
# The value is communicated to the parent process via mem_value_holder
def memory_profiler_proc(block_bytes, inp_shape, mem_value_holder):

    import gc
    import psutil
    import os

    # Measure memory before block loading and inference
    mem_before = psutil.Process(os.getpid()).memory_info().rss

    print(mem_before, " LALA\n")

    # De-serialize block bytes
    block = pickle.loads(block_bytes)

    # # Remove block's bytes from memory
    # # TODO: Maybe do the same in deployment?
    # del block_bytes
    # gc.collect()

    # Create dummy input
    inp = mx.nd.random.uniform(low=0, high=1, shape=inp_shape)

    # Run inference
    outp = core.manual_inference(block, inp)
    mx.nd.waitall()

    # Measure memory after inference
    mem_after = psutil.Process(os.getpid()).memory_info().rss

    # Store inference's memory footprint (difference) in MB to shared holder
    mem_value_holder.value = (mem_after - mem_before) / (1024 * 1024)


if __name__ == "__main__":

    import json
    import time
    import sys
    import copy
    from mxnet.gluon import nn
    import gluoncv
    import numpy as np
    from mxnet.gluon.model_zoo import vision


    # Helper function to initialize the profiler's dictionary to be returned
    def profiler_dict(filter_split_configs, input_split_configs, profile_model_specifics):

        keys_arrays_fs = {"fs_" + str(c): [] for c in filter_split_configs}
        keys_arrays_is = {"is_" + str(c): [] for c in input_split_configs}

        ret = {
            "model": "",
            "node": "",
            "exec_times": keys_arrays_fs
        }
        ret["exec_times"]["fs_1"] = keys_arrays_is

        if profile_model_specifics:
            ret["num_blocks"] = 0
            ret["inp_sizes"] = copy.deepcopy(keys_arrays_fs)
            ret["inp_sizes"]["fs_1"] = copy.deepcopy(keys_arrays_is)
            ret["out_sizes"] = copy.deepcopy(keys_arrays_fs)
            ret["out_sizes"]["fs_1"] = copy.deepcopy(keys_arrays_is)
            ret["mem_cons"] = copy.deepcopy(keys_arrays_fs)
            ret["mem_cons"]["fs_1"] = copy.deepcopy(keys_arrays_is)

        return ret


    # Receives a flattened model and returns a list with sequential layer block offsets (partitions,
    # according to pipelined model partitioning), the type and the # of filters/neurons of each partition
    def identify_partitioning_points(flattened_model):
        pp = []
        p_start = None
        num_filters = None
        p_type = None
        for i, layer_tuple in enumerate(flattened_model):
            layer = layer_tuple[0]

            # Partitions are groups of layers that start with a Conv2D, or contain a single FC layer
            if isinstance(layer, nn.Conv2D):
                p_next_type = "conv"
            elif isinstance(layer, nn.Dense):
                p_next_type = "dense"
            else:
                continue

            p_end = i
            pp.append({
                "p_start": p_start,
                "p_end": p_end,
                "num_filters": num_filters,
                "p_type": p_type
            })
            p_start = i
            p_type = p_next_type
            num_filters = list(layer.params.values())[0].shape[0]  # TODO: is this correct for all types of layers?

        # Manually add last partition
        pp.append({
            "p_start": p_start,
            "p_end": i + 1,
            "num_filters": num_filters,
            "p_type": "dense"  # TODO: is it always dense?
        })

        return pp[1:]  # exclude first (dummy) element


    block_ctr = 0


    # Receives a layer block, its input and an option (boolean) to profile model specific data
    # and outputs the block's memory consumption, execution time stats, its input and output size and its output
    # Note: memory consumption, input and output sizes are meaningful only if profile_model_specifics option in True
    def profile_block(block, inp, profile_model_specifics):

        # TODO DELETE: FOR DEBUGGING
        global block_ctr
        print(block_ctr)
        block_ctr += 1

        # Profile block's execution time after 'num_runs' runs
        num_runs = 6
        warm_up_runs = round(num_runs * 0.15)  # the first 15% of runs are for warm-up

        et_list = []  # keep exec times from all runs
        outp = None
        for run in range(num_runs):
            start = time.time()
            outp = core.manual_inference(block, inp)  # execute inference
            mx.nd.waitall()  # wait for all mxnet operations to finish
            end = time.time()
            if run < warm_up_runs:  # do not consider this measurement, it's a warm-up run
                continue

            et_list.append(1000 * (end - start))  # measure elapsed time (in ms) of partition execution

        et_ar = np.array(et_list)
        et = {
            'et_mean': np.mean(et_ar),
            'et_std': np.std(et_ar),
            'et_median': np.median(et_ar),
        }

        # Measure block's memory consumption, input and output size in bytes
        if profile_model_specifics:
            metadata_size = 24
            outs = outp.asnumpy().nbytes + metadata_size
            inps = inp.asnumpy().nbytes + metadata_size
            mem_value_holder = multiprocessing.Value('f', 0.0)
            mem_prof_proc = multiprocessing.Process(target=memory_profiler_proc, args=(pickle.dumps(block, protocol=pickle.HIGHEST_PROTOCOL), inp.shape, mem_value_holder))
            mem_prof_proc.start()
            mem_prof_proc.join()
            mem = mem_value_holder.value
        else:
            outs = inps = mem = -1

        return inps, mem, et, outs, outp


    # Receives a flattened model, its input, an option (boolean) to profile model specific data, model's name and profiling
    # node's name and produces a dictionary with the inference execution times, the input and output sizes and the memory
    # consumption values (if profile_model_specifics is True) of every sequential layer block of the model for the running
    # node, each one measured using different layer-level splitting strategies
    # Briefly, layer blocks are executed as follows:
    # a. no layer-level splitting: full block / full input
    # b. filter splitting: block partitioned in filter dimension / full input
    # c. input splitting: full block (or trimmed for FC) / portion of the input received
    def custom_profiler(flattened_model, first_input, profile_model_specifics, model_name, node_name):
        pp = identify_partitioning_points(flattened_model)  # layer groups to measure

        # Configurations to measure
        filter_split_configs = [0.25, 0.5, 0.75, 1]
        input_split_configs = [0.25, 0.5, 0.75, 1]

        inp = first_input
        outp = None

        # Init dict to be returned
        ret_data = profiler_dict(filter_split_configs, input_split_configs, profile_model_specifics)

        # Some first basic data
        if profile_model_specifics:
            ret_data['num_blocks'] = len(pp)
            ret_data['block_info'] = []
        ret_data['node'] = node_name
        ret_data['model'] = model_name

        for i in range(len(pp)):  # for every DNN layer block
            p = pp[i]

            # Store some info about the block to be used in deployment
            if profile_model_specifics:
                block_info_dict = {
                    "from_layer": p["p_start"],
                    "to_layer": p["p_end"],
                    "num_filters": p["num_filters"],
                    "num_weights_if_dense": None
                }
                ret_data['block_info'].append(block_info_dict)

            for fsc in filter_split_configs:  # for every filter split configuration

                if fsc == 1:  # for the no-filter-split configuration, measure the input split configurations

                    for isc in input_split_configs:

                        if isc < 1:  # input needs to split

                            if p["p_type"] == "dense":  # dense block

                                if len(inp.shape) != 2:  # input needs flattening
                                    inp = nn.Flatten()(inp)
                                weights_trim_offset_end = round(inp.shape[1] * isc)
                                split_inp = inp[:, 0:weights_trim_offset_end]  # split flattened input for fc block

                                # Generate layer block with trimmed dense weights
                                block_dense_inp_split = core.generate_model_partition(
                                    flattened_model,
                                    from_layer=p["p_start"],
                                    to_layer=p["p_end"],
                                    from_weight=0,
                                    to_weight=weights_trim_offset_end
                                )

                                # Produce measurements
                                inps, mem, et, outs, _ = profile_block(block_dense_inp_split, split_inp, profile_model_specifics)

                                # Write down measurements for dense block input split
                                ret_data['exec_times']['fs_1']['is_'+str(isc)].append([et])
                                if profile_model_specifics:
                                    ret_data['out_sizes']['fs_1']['is_'+str(isc)].append([outs])
                                    ret_data['inp_sizes']['fs_1']['is_' + str(isc)].append([inps])
                                    ret_data['mem_cons']['fs_1']['is_' + str(isc)].append([mem])
                                    if not ret_data['block_info'][-1]["num_weights_if_dense"]:
                                        ret_data['block_info'][-1]["num_weights_if_dense"] = inp.shape[1]

                            else:  # conv block

                                # Measuremets for input split block that grows each time

                                from_layer = p["p_start"]  # starting point
                                p_offset = 0

                                # Following lists will store measurements for each block depth
                                et_list = []
                                if profile_model_specifics:
                                    outs_list = []
                                    inps_list = []
                                    mem_list = []

                                while True:  # continue forever until a dense block is met or required input is too big

                                    end_block = pp[i + p_offset]  # grow measured block by one more layer group every time

                                    # We cannot include a dense layer in a deep input split configuration
                                    if end_block["p_type"] == "dense":
                                        break

                                    to_layer = end_block["p_end"]  # current ending point

                                    # Generate layer block without splitting (only input will split)
                                    block_inp_split = core.generate_model_partition(
                                        flattened_model,
                                        from_layer=from_layer,
                                        to_layer=to_layer
                                    )

                                    # Required width dimension size for block?
                                    inp_offsets = core.split_input_offsets(block_inp_split, [isc, 1 - isc])  # more cases here? TODO
                                    assert inp_offsets[0][0] == 0
                                    
                                    # We cannot let the suggested split limit to reach or exceed the width dimension of the original input
                                    if inp_offsets[0][1] >= inp.shape[3] or -inp_offsets[1][0] >= inp.shape[3]:
                                        break

                                    # Split input
                                    split_inp = inp[:, :, :, 0:inp_offsets[0][1]]

                                    # Produce measurements
                                    inps, mem, et, outs, _ = profile_block(block_inp_split, split_inp, profile_model_specifics)

                                    # Store measurements for current block depth
                                    et_list.append(et)
                                    if profile_model_specifics:
                                        outs_list.append(outs)
                                        inps_list.append(inps)
                                        mem_list.append(mem)

                                    p_offset += 1

                                # Write down all measurements for conv block input split
                                ret_data['exec_times']['fs_1']['is_'+str(isc)].append(et_list)
                                if profile_model_specifics:
                                    ret_data['out_sizes']['fs_1']['is_'+str(isc)].append(outs_list)
                                    ret_data['inp_sizes']['fs_1']['is_' + str(isc)].append(inps_list)
                                    ret_data['mem_cons']['fs_1']['is_' + str(isc)].append(mem_list)
                        
                        else:  # isc == 1 so no layer-level splitting (just use original input)

                            # Generate layer block without splitting
                            block_no_split = core.generate_model_partition(
                                flattened_model,
                                from_layer=p["p_start"],
                                to_layer=p["p_end"]
                            )

                            # Produce measurements
                            inps, mem, et, outs, outp = profile_block(block_no_split, inp, profile_model_specifics)

                            # Write down measurements
                            ret_data['exec_times']['fs_1']['is_'+str(isc)].append(et)
                            if profile_model_specifics:
                                ret_data['out_sizes']['fs_1']['is_'+str(isc)].append(outs)
                                ret_data['inp_sizes']['fs_1']['is_' + str(isc)].append(inps)
                                ret_data['mem_cons']['fs_1']['is_' + str(isc)].append(mem)

                    inp = outp  # get ready for the next partition (new input)

                else:  # fsc < 1

                    # Generate layer block, splitting in the filter dimension
                    block_filter_split = core.generate_model_partition(
                        flattened_model,
                        from_layer=p["p_start"],
                        to_layer=p["p_end"],
                        from_filter=0,
                        to_filter=int(round(p["num_filters"] * fsc))
                    )

                    # Produce measurements
                    inps, mem, et, outs, _ = profile_block(block_filter_split, inp, profile_model_specifics)

                    # Write down measurements
                    ret_data['exec_times']['fs_'+str(fsc)].append(et)
                    if profile_model_specifics:
                        ret_data['out_sizes']['fs_'+str(fsc)].append(outs)
                        ret_data['inp_sizes']['fs_' + str(fsc)].append(inps)
                        ret_data['mem_cons']['fs_' + str(fsc)].append(mem)

        return ret_data


    # ---- MAIN ----

    # To run script:
    # python benchmark.py <node_name> <model_name> <profile_model_specifics>
    # e.g. python benchmark.py iot_A1 vgg16 True


    # Read CL args
    assert len(sys.argv) == 4
    node_name = sys.argv[1]
    model_name = sys.argv[2]
    if model_name == "vgg16":
        model = core.flatten_vgg16()
        first_inp = mx.nd.random.uniform(low=0, high=1, shape=(1, 3, 224, 224))
    else:
        assert False # todo add more models
    profile_model_specifics = sys.argv[3]
    if profile_model_specifics == "True":
        profile_model_specifics = True
        multiprocessing.set_start_method('spawn', force=True)  # child process that measures memory consumption should be separate from parent
    else:
        profile_model_specifics = False

    # Run profiler
    start_time = time.time()

    profiling_data = custom_profiler(model, first_inp, profile_model_specifics, model_name, node_name)

    mx.nd.waitall()
    print("BENCH TOOK", time.time()-start_time, "seconds!")

    # Write profiling data to file
    with open(node_name + '.json', 'w') as out_file:
        json.dump(profiling_data, out_file)
        