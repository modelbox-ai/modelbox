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
import cv2

class CustomResize(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.max_edge = config.get_int("max_edge", 320)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_image_list = data_context.input("in_image")
        out_image_list = data_context.output("out_image")

        for buffer_img in in_image_list:
            width = buffer_img.get("width")
            height = buffer_img.get("height")
            channel = buffer_img.get("channel")

            img_data = np.array(buffer_img.as_object(), dtype=np.uint8)
            img_data = img_data.reshape(height, width, channel)

            im_size_min = np.min([height, width])
            im_size_max = np.max([height, width])
            resize = self.max_edge / float(im_size_min)
            if np.round(resize * im_size_max) > self.max_edge:
                resize = self.max_edge / float(im_size_max)

            resize_img = cv2.resize(img_data, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            resize_img_height, resize_img_width, _ = resize_img.shape

            add_buffer = modelbox.Buffer(self.get_bind_device(), resize_img)
            add_buffer.copy_meta(buffer_img)
            add_buffer.set("width", resize_img_width)
            add_buffer.set("height", resize_img_height)
            add_buffer.set("width_stride", resize_img_width)
            add_buffer.set("height_stride", resize_img_height)
            out_image_list.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
