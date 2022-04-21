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

class DrawEmotion(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_face_list = data_context.input("in_face")
        in_emotion_list = data_context.input("in_emotion")
        out_data_list = data_context.output("out_data")

        for image, emotion in zip(in_face_list, in_emotion_list):
            bboxes = image.get("bboxes")
            bboxes = np.array(bboxes).reshape(-1, 4)

            width = image.get("width")
            height = image.get("height")
            channel = image.get("channel")

            out_img = np.array(image.as_object(), dtype=np.uint8)
            out_img = out_img.reshape(height, width, channel)

            emotion = emotion.as_object().split(",")
            for box, emo in zip(bboxes, emotion):
                cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(out_img, emo, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            add_buffer = modelbox.Buffer(self.get_bind_device(), out_img)
            add_buffer.copy_meta(image)
            out_data_list.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
