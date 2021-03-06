from __future__ import print_function
import grpc
import requests
import tensorflow as tf
import cv2
import time
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import label_map_util
import visualization_utils as vis_util

tf.app.flags.DEFINE_string('server', 'localhost:8500',  # 34.73.124.32   0.0.0.0,  localhost
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image',
                           'call_samsung_14252401402.png',
                           'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS
compress = True
width = 720
height = 1280


def main(_):
    # step 2: send a request
    options = [('grpc.max_send_message_length', 1000 * 1024 * 1024),
               ('grpc.max_receive_message_length', 1000 * 1024 * 1024)]
    channel = grpc.insecure_channel(FLAGS.server, options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'model'
    request.model_spec.signature_name = 'serving_default'

    # step 1: prepare input
    img = cv2.imread(FLAGS.image)
    h, w = img.shape[:2]
    if compress:
        ratio = w / h
        h1 = height
        w1 = round(h1 * ratio)
        scaled_img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
        tensor = tf.contrib.util.make_tensor_proto(scaled_img, shape=[1] + list(scaled_img.shape))
    else:
        tensor = tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape))

    request.inputs['inputs'].CopyFrom(tensor)
    start = time.time()

    # step 3: get the results
    result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
    result = result_future.result()

    stop = time.time()
    print('time is ', stop - start)

    NUM_CLASSES = 30
    label_map = label_map_util.load_labelmap('annotations/label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val

    result = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.reshape(boxes, [100, 4]),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imwrite('result.jpg', result)


if __name__ == '__main__':
    tf.app.run()
