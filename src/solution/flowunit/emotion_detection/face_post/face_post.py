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

from face_post_utils import postprocess

class FacePost(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.max_edge = config.get_int("max_edge", 320)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_image_list = data_context.input("in_image")
        in_loc_list = data_context.input("in_loc")
        in_conf_list = data_context.input("in_conf")

        has_face_list = data_context.output("has_face")
        no_face_list = data_context.output("no_face")

        for buffer_img, buffer_loc, buffer_conf in zip(in_image_list, in_loc_list, in_conf_list):
            width = buffer_img.get("width")
            height = buffer_img.get("height")
            channel = buffer_img.get("channel")

            img_data = np.array(buffer_img.as_object())
            img_data = img_data.reshape(height, width, channel)

            im_size_min = np.min([height, width])
            im_size_max = np.max([height, width])
            resize = self.max_edge / float(im_size_min)
            if np.round(resize * im_size_max) > self.max_edge:
                resize = self.max_edge / float(im_size_max)
            resize_height = int(height * resize)
            resize_width = int(width * resize)
            scale = np.array([resize_width, resize_height, resize_width, resize_height])

            loc = np.array(buffer_loc.as_object())
            conf = np.array(buffer_conf.as_object())
            loc = np.reshape(loc, (-1, 4))
            conf = np.reshape(conf, (-1, 2))

            bboxes = postprocess((resize_height, resize_width), loc, conf, scale, resize)
            if bboxes is None or bboxes.size == 0:
                no_face_list.push_back(buffer_img)
            else:
                bboxes = np.delete(bboxes, -1, axis=1).astype(int)
                buffer_img.set("bboxes", bboxes.flatten().tolist())
                has_face_list.push_back(buffer_img)
                
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
