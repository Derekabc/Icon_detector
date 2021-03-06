from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os import walk
import cv2
import numpy as np
import tensorflow as tf

import label_map_util
import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using  # iPhone 11, Samsung_Galaxy S20, OnePlus_8, Google_Pixel 4
MODEL_NAME = 'output_inference_graph_v3/'

INPUT_DIR = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/images/test_images_processed'
if MODEL_NAME == 'output_inference_graph_v1/':
    OUT_DIR = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/images/Inference_output_v3/'
else:
    OUT_DIR = '/home/dong/PycharmProjects/Intern/Icon_Detector/TensorFlow/images/Inference_output_v4/'
os.makedirs(OUT_DIR, exist_ok=True)
# Grab path to current working directory
CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'annotations','label_map.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 30

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load images
# get all the pictures in directory
images = []
ext = (".jpeg", ".jpg", ".png", ".PNG")

for (dirpath, dirnames, filenames) in walk(INPUT_DIR):
    for filename in filenames:
        if filename.endswith(ext):
            images.append(os.path.join(dirpath, filename))

print("Working...")
for img in images:
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(img)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # All the results have been drawn on image. Now display the image.
    # cv2.imshow('Object detector', image)
    cv2.imwrite(os.path.join(OUT_DIR, img.split('/')[-1]), image)
    # Press any key to close the image
    # cv2.waitKey(0)

    # Clean up
    # cv2.destroyAllWindows()
