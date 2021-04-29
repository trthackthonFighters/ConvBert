from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import modeling
from run_finetuning import FinetuningModel
from util import training_utils
from datetime import datetime
from finetune import task_builder
import configure_finetuning
import numpy as np
import tensorflow as tf
import os
import time

import sys
i_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import tf2onnx
import ctypes
import onnx_graphsurgeon as gs
import onnx

ctypes.cdll.LoadLibrary('./OnehotPlugin.so')
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)



#Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]




def construct_model():
    tf.compat.v1.disable_eager_execution()
    batch_size = 1
    iterations = 10
    tf_datatype = tf.int32
    np_datatype = np.int32

    graph_features = {}
    graph_features["input_ids"] = tf.placeholder(dtype=tf_datatype, shape=[batch_size, 128], name="input_ids")
    graph_features["input_mask"] = tf.placeholder(dtype=tf_datatype, shape=[batch_size, 128], name="input_mask")
    graph_features["segment_ids"] = tf.placeholder(dtype=tf_datatype, shape=[batch_size, 128], name="token_type_ids")

    #graph_features["task_id"] = tf.placeholder(dtype=tf_datatype, shape=(batch_size,), name="task_id")
    #graph_features["cola_label_ids"] = tf.placeholder(dtype=tf_datatype, shape=(batch_size,), name="cola_label_ids")
    #graph_features["cola_eid"] = tf.placeholder(dtype=tf_datatype, shape=(batch_size,), name="cola_eid")


    features = {}
    features["input_ids"] = np.random.rand(batch_size, 128).astype(np_datatype)
    features["input_mask"] = np.random.rand(batch_size, 128).astype(np_datatype)
    features["segment_ids"] = np.random.rand(batch_size, 128).astype(np_datatype)
    #features["task_id"] = np.random.rand(batch_size).astype(np_datatype)
    #features["cola_label_ids"] = np.random.rand(batch_size).astype(np_datatype)
    #features["cola_eid"] = np.random.rand(batch_size).astype(np_datatype)

    
    features_feed_dict = {graph_features[key] : features[key] for key in graph_features}
    #print(features_feed_dict)
    print(features)

    time.sleep(10)
    features_use = graph_features
    features_use["task_id"] = tf.constant(0, dtype=tf_datatype, shape=(batch_size,), name="task_id")
    features_use["cola_label_ids"] = tf.constant(0, dtype=tf_datatype, shape=(batch_size,), name="cola_label_ids")
    features_use["cola_eid"] = tf.constant(0, dtype=tf_datatype, shape=(batch_size,), name="cola_eid")

    param = {"model_size": "medium-small", "task_names": ["cola"]}
    config = configure_finetuning.FinetuningConfig(model_name="convbert_medium-small",
                                                   data_dir="./",
                                                   **param)
    task = task_builder.get_tasks(config)
    print("Getting tasks:".format(task))
    is_training = False
    nums_steps = 0

    model = FinetuningModel(config, task, is_training, features_use, nums_steps)
    outputs = model.outputs

    out_dict = {}
    for tks in task:
        out_dict = outputs[tks.name]

    output_names = []
    for key in out_dict:
        output_names.append(out_dict[key].name)
        print(out_dict[key].name)

    run_op_list = []
    outputs_names_with_port = output_names
    outputs_names_without_port = [name.split(":")[0] for name in outputs_names_with_port]
    for index in range(len(outputs_names_without_port)):
        run_op_list.append(outputs_names_without_port[index])
    print(run_op_list)

    inputs_names_with_port = [graph_features[key].name for key in graph_features]
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        time_sum = 0
        a = datetime.now()
        for i in range(iterations):
          tf_result = sess.run(run_op_list, feed_dict=features_feed_dict)
        b = datetime.now()
        time_sum = (b - a).total_seconds()
        tf_time = "[INFO] TF  execution time " + str(
            time_sum * 1000 / iterations) + " ms"
        print(tf_time)
        print(tf_result)


    print("finished tf inferencing")
    time.sleep(10)


    inputs_names_with_port.remove("task_id:0")

    for engine_file_path in ['ConvBert_onehot.trt']:
        if not os.path.exists(engine_file_path):
            print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
            exit(1)
    print('====', engine_file_path, '===')


    tf.set_random_seed(1234)
    np.random.seed(0)
    iterations = 100
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    model_file = "./ConvBert_onehot.onnx"
    # build trt model by onnx model
    cuda.Device(0).make_context()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size

        with open(model_file, 'rb') as model:
            # parse onnx model
            parser.parse(model.read())
            for i in range(parser.num_errors):
                print(parser.get_error(i))

        engine = builder.build_cuda_engine(network)
        if engine == None:
            print("[ERROR] engine is None")
            exit(-1)
        else:
            print("[INFO] engine is built")
    print("follow") 


    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
            count = 0
            for key in features:
               input_data = features[key].ravel()
               np.copyto(inputs[count].host, input_data)

            time_sum = 0
            a = datetime.now()
            for i in range(iterations):
                #np.copyto(inputs[0].host, input_data)
                output = do_inference(context,
                                      bindings=bindings,
                                      inputs=inputs,
                                      outputs=outputs,
                                      stream=stream,
                                      batch_size=batch_size)
            b = datetime.now()
            time_sum = (b - a).total_seconds()
            trt_time = ("TRT execution time " +
                        str(time_sum * 1000 / iterations) + " ms")
            trt_result = output
            print(tf_result)
            print(trt_result)
            # comment out the print as tf_result is None. 
            """ 
            for i in range(len(trt_result)):
              print("trt cross_check output_%d " % i + str(np.allclose(tf_result[i].flatten(), trt_result[i], atol=1e-5)))
              print("max diff " + str(np.fabs(tf_result[i].flatten() - trt_result[i]).max()))
              print("min diff " + str(np.fabs(tf_result[i].flatten() - trt_result[i]).min()))
            """

            print("tf_time=",tf_time)
            print("trt_time=",trt_time)

            cuda.Context.pop()
    
 











if "__main__" == __name__:
    construct_model()

    

