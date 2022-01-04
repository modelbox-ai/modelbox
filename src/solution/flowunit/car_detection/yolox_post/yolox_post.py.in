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

import _flowunit as modelbox
import numpy as np 
import json 
import cv2 

from yolox_utils import *

class YoloXPost(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.net_h = config.get_int('net_h', 288)
        self.net_w = config.get_int('net_w', 512)
        self.num_classes = config.get_int('num_classes', 80)
        self.num_grids = int((self.net_h / 32) * (self.net_w / 32)) * (1 + 2*2 + 4*4)
        self.conf_thre = config.get_float('conf_threshold', 0.3)
        self.nms_thre = config.get_float('iou_threshold', 0.4)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_image = data_context.input("in_image")
        in_feat = data_context.input("in_feat")

        out_data = data_context.output("out_data")

        for buffer_img, buffer_feat in zip(in_image, in_feat):
            width = buffer_img.get('width')
            height = buffer_img.get('height')
            channel = buffer_img.get('channel')
            frame_index = buffer_img.get('index')
            modelbox.debug("get frame index: {}".format(frame_index))

            img_data = np.array(buffer_img.as_object(), copy=False)
            img_data = img_data.reshape((height, width, channel))

            feat_data = np.array(buffer_feat.as_object(), copy=False)
            feat_data = feat_data.reshape((self.num_classes + 5, self.num_grids)).transpose()

            ratio = min(self.net_h / height, self.net_w / width)
            bboxes = postprocess(feat_data, (self.net_h, self.net_w), self.num_classes, self.conf_thre, self.nms_thre, ratio)
            if bboxes is not None:
                img_out = draw_bbox(img_data, bboxes)
                add_buffer = modelbox.Buffer(self.get_bind_device(), img_out)
                add_buffer.copy_meta(buffer_img)
                out_data.push_back(add_buffer)
            else:
                out_data.push_back(buffer_img)
            
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
