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
    print(features_feed_dict)

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

        for i in range(iterations):
            sess.run(run_op_list, feed_dict=features_feed_dict)
        tf_time_sum = 0
        a = datetime.now()
        for i in range(iterations):
            tf_result = sess.run(run_op_list, feed_dict=features_feed_dict)
        b = datetime.now()
        tf_time_sum = (b - a).total_seconds()
        tf_time = "[INFO] TF  execution time: " + str(
            tf_time_sum * 1000 / iterations) + " ms"
        print(tf_time)

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, outputs_names_without_port)
        # frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
        # save frozen model
        with open("ConvBert.pb", "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())
    exit(0)

    inputs_names_with_port.remove("task_id:0")

    onnx_model_file = "ConvBert.onnx"
    command = "python3 -m tf2onnx.convert --input ConvBert.pb --output %s --fold_const --opset 12 --verbose" % onnx_model_file
    command += " --inputs "
    for name in inputs_names_with_port:
        command += "%s," % name
    command = command[:-1] + " --outputs "
    for name in outputs_names_with_port:
        command += "%s," % name
    command = command[:-1]
    os.system(command)
    print(command)
    #exit(0)

    #do not convert now, it needs to modify onehot layer
    #command = "trtexec --onnx=ConvBert.onnx --verbose"
    #os.system(command)
    print(command)


if "__main__" == __name__:
    construct_model()
