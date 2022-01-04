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

class ExpandBox(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data_list = data_context.input("in_data")
        out_image_list = data_context.output("roi_image")

        for in_buffer in in_data_list:
            width = in_buffer.get("width")
            height = in_buffer.get("height")
            channel = in_buffer.get("channel")

            img = np.array(in_buffer.as_object(), dtype=np.uint8)
            img = img.reshape(height, width, channel)

            bboxes = in_buffer.get("bboxes")
            bboxes = np.array(bboxes).reshape(-1, 4)
            for box in bboxes:
                img_roi = img[box[1]:box[3], box[0]:box[2]]
                img_roi = img_roi[:, :, ::-1]

                img_roi = img_roi.flatten()
                add_buffer = modelbox.Buffer(self.get_bind_device(), img_roi)
                add_buffer.copy_meta(in_buffer)
                add_buffer.set("pix_fmt", "rgb")
                add_buffer.set("width", int(box[2] - box[0]))
                add_buffer.set("height", int(box[3] - box[1]))
                add_buffer.set("width_stride", int(box[2] - box[0]))
                add_buffer.set("height_stride", int(box[3] - box[1]))
                out_image_list.push_back(add_buffer)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
