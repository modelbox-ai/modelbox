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

class CollapseEmotion(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.emotion_list = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        confidence_list = data_context.input("confidence")
        predicts_list = data_context.input("predicts")
        out_data_list = data_context.output("out_data")

        emotion_result = ""
        for conf, predict in zip(confidence_list, predicts_list):
            conf = np.array(conf.as_object(), dtype=np.float32)
            predict = np.array(predict.as_object(), dtype=np.float32)

            res = "NoEmotion"
            if conf > 0.7:
                res = self.emotion_list[np.argmax(predict)]
            if len(emotion_result) > 0:
                emotion_result += ","
            emotion_result += res

        emotion_buffer = modelbox.Buffer(self.get_bind_device(), emotion_result)
        out_data_list.push_back(emotion_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()
    
    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
