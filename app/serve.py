import tensorflow as tf
import numpy as np

def serve_unet_model():

    TFLITE_MODEL = "/app/UNet_25_Crack.tflite"

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    return tflite_interpreter, height, width, input_details, output_details

def serve_rcnn_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile("/app/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
