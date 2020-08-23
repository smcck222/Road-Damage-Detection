import flask
from flask import Flask
import numpy as np
import io
from PIL import Image, ImageDraw
import tensorflow as tf
import ops as utils_ops
import visualization_utils as vis_util

from serve import serve_unet_model
from serve import serve_rcnn_model

app = Flask(__name__)

def load_unet_model():
    global tflite_interpreter_c, height_c, width_c, input_details_c, output_details_c
    tflite_interpreter_c, height_c, width_c, input_details_c, output_details_c = serve_unet_model()

def load_rcnn_model():
    global detection_graph
    detection_graph = serve_rcnn_model()

load_unet_model()
load_rcnn_model()

def prepare_img(image, type):
    if type == "detect":
        return Image.open(image).resize((width, height))
    elif type == "segment":
        return Image.open(image).resize((width_c, height_c))

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

@app.route("/detect/rcnn", methods=["POST"])
def detect_rcnn():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = Image.open(flask.request.files["image"])
            image_np = load_image_into_numpy_array(image)
            #image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            category_index = {0:{"name":"pothole"},1:{"name":"pothole"}}
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8,
                skip_scores=True,
                skip_labels=True )
            img = Image.fromarray(image_np.astype("uint8"))
            img = img.resize((128,128))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            return rawBytes.getvalue()
        else:
            return "Could not find image"
    return "Please use POST method"

@app.route("/segment", methods=["POST"])
def segment():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            img = prepare_img(flask.request.files["image"], "segment")


            input_data = np.expand_dims(img, axis=0)
            input_data = np.float32(input_data)/255.0
            tflite_interpreter_c.set_tensor(input_details_c[0]['index'], input_data)
            tflite_interpreter_c.invoke()
            result = tflite_interpreter_c.get_tensor(output_details_c[0]['index'])
            result = result > 0.5
            result = result*255
            mask = np.squeeze(result)
            bg = np.asarray(img).copy()
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]>0:
                        bg[i][j][0] = 0
                        bg[i][j][1] = 0
                        bg[i][j][2] = 255

            img = Image.fromarray(bg.astype("uint8"))

            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            return rawBytes.getvalue()
        else:
            return "Could not find image"
    return "Please use POST method"

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


@app.route('/')
def index():
    return "Road Damage Detection"

if __name__ == "__main__":
    app.run()
