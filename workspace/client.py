from __future__ import print_function
import grpc
import requests
import tensorflow.compat.v1 as tf
import cv2
import time
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import label_map_util
import visualization_utils as vis_util

tf.app.flags.DEFINE_string('server', '****:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/images/test_images_processed/Google_Pixel 4_Jul_3_2020_22_47_47.png', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
    # 设置grpc
    options = [('grpc.max_send_message_length', 1000 * 1024 * 1024),
               ('grpc.max_receive_message_length', 1000 * 1024 * 1024)]
    channel = grpc.insecure_channel(FLAGS.server, options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'detection'
    request.model_spec.signature_name = 'serving_default'

    # 输入图片并进行请求
    img = cv2.imread(FLAGS.image)
    tensor = tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape))
    request.inputs['inputs'].CopyFrom(tensor)
    start = time.time()

    # 法一，速度较慢
    # result = stub.Predict(request, 10.0)  # 10 secs timeout

    # 法二，速度较快
    result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
    result = result_future.result()

    stop = time.time()
    print('time is ', stop - start)

    # 读取标签配置文件
    NUM_CLASSES = 30
    label_map = label_map_util.load_labelmap('annotations/label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # 可视化检测结果
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    result = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.reshape(boxes, [300, 4]),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # 保存结果图片
    cv2.imwrite('result.jpg', result)


if __name__ == '__main__':
    tf.app.run()