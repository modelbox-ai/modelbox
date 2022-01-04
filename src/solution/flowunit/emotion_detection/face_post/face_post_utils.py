#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import cv2
import numpy as np
from itertools import product

def get_priors(image_size):
    steps = [8, 16, 32, 64]
    min_sizes_list = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    feature_maps = [[math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)] for step in steps]
    anchors = []
    for index, map in enumerate(feature_maps):
        min_sizes = min_sizes_list[index]
        for map_y, map_x in product(range(map[0]), range(map[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[index] / image_size[1] for x in [map_x + 0.5]]
                dense_cy = [y * steps[index] / image_size[0] for y in [map_y + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    
    output = np.reshape(np.float32(anchors), (-1, 4))
    return output

def decode(loc, priors, variances):
    a = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    b = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes = np.concatenate((a, b), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def postprocess(image_size, loc, conf, scale, resize):
    confidence_threshold = 0.3
    top_k = 100
    nms_threshold = 0.4

    priors = get_priors(image_size)
    boxes = decode(loc, priors, [0.1, 0.2])
    boxes = boxes * scale / resize
    scores = conf[:, 1]

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]

    return dets
