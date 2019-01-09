from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")

PATH_TO_CKPT = './model/frozen_inference_graph.pb'

PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def count_nonblack_np(img):
    return img.any(axis=-1).sum()


def detect_team(image, show=False):
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),
        ([25, 146, 190], [96, 174, 250])
    ]
    i = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix / tot_pix

        if ratio > 0.01 and i == 0:
            return 'red'
        elif ratio > 0.01 and i == 1:
            return 'blue'

        i += 1

        if show == True:
            cv2.imshow("images", np.hstack([image, output]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
    return 'not_sure'


filename = 'teams.jpg'
image = cv2.imread(filename)
resize = cv2.resize(image, (640, 360))
detect_team(resize, show=True)

out = cv2.VideoWriter('hacettepereddeers.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 360))

filename = 'hacettepevsbogazici.mp4'
cap = cv2.VideoCapture(filename)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        counter = 0
        while (True):
            ret, image_np = cap.read()
            counter += 1
            if ret:
                h = image_np.shape[0]
                w = image_np.shape[1]

            if not ret:
                break
            if counter % 1 == 0:
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=0.6)

                frame_number = counter
                loc = {}
                for n in range(len(scores[0])):
                    if scores[0][n] > 0.60:
                        ymin = int(boxes[0][n][0] * h)
                        xmin = int(boxes[0][n][1] * w)
                        ymax = int(boxes[0][n][2] * h)
                        xmax = int(boxes[0][n][3] * w)

                        for cat in categories:
                            if cat['id'] == classes[0][n]:
                                label = cat['name']

                        if label == 'person':
                            crop_img = image_np[ymin:ymax, xmin:xmax]
                            color = detect_team(crop_img)
                            if color != 'not_sure':
                                coords = (xmin, ymin)
                                if color == 'red':
                                    loc[coords] = 'HACETTEPE RED DEERS'
                                else:
                                    loc[coords] = 'BOGAZICI SULTANS'

                for key in loc.keys():
                    text_pos = str(loc[key])
                    cv2.putText(image_np, text_pos, (key[0], key[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                                2)

            cv2.imshow('image', image_np)
            out.write(image_np)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                break

