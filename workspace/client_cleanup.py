from __future__ import print_function
import grpc
import tensorflow as tf
import cv2
import time
import numpy as np
import json
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', '0.0.0.0:8500',  # 34.73.124.32   0.0.0.0
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image',
                           '/home/dong/PycharmProjects/Icon_detector/workspace/Google_Pixel 4_Jun_30_2020_09_34_45.png',
                           'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS
min_score_thresh = .60
output_name = 'predict_out.json'
label_map = {"1": "call", "2": "message", "3": "n0", "4": "n1", "5": "n2",
             "6": "n3", "7": "n4", "8": "n5", "9": "n6", "10": "n7",
             "11": "n8", "12": "n9", "13": "star", "14": "n0", "15": "sign",
             "16": "videocall", "17": "keypad", "18": "hangup", "19": "search",
             "20": "tkeypad", "21": "add", "22": "tdial", "23": "tedit", "24": "edit",
             "25": "lte", "26": "signal", "27": "4g", "28": "send", "29": "tsend", "30": "wifi"}


def main(_):
    start = time.time()
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
    tensor = tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape))
    width = img.shape[1]
    height = img.shape[0]
    request.inputs['inputs'].CopyFrom(tensor)


    # step 3: get the results
    result_future = stub.Predict.future(request, 20.0)  # 20 secs timeout
    result = result_future.result()

    boxes = np.reshape(result.outputs['detection_boxes'].float_val, [100, 4])
    classes = np.squeeze(result.outputs['detection_classes'].float_val).astype(np.int32)
    scores = np.squeeze(result.outputs['detection_scores'].float_val)
    detection_out = {}
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = boxes[i].tolist()  # ymin, xmin, ymax, xmax
            class_name = label_map[str(classes[i])]
            score = round(100 * scores[i])
            if class_name in detection_out:
                if score > detection_out[class_name][4]:
                    # [left, right, top, bottom, score]
                    detection_out[class_name] = [box[1] * width, box[3] * width, box[0] * height, box[2] * height, score]
            else:
                # [left, right, top, bottom, score]
                detection_out[class_name] = [box[1] * width, box[3] * width, box[0] * height, box[2] * height, score]

    with open(output_name, 'w') as outfile:
        json.dump(detection_out, outfile)
    print('time is ', time.time() - start)


if __name__ == '__main__':
    tf.app.run()
