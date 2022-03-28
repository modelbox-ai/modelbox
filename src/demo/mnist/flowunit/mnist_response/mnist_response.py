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

class MnistResponseFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data = data_context.input("In_1")
        out_data = data_context.output("Out_1")

        if data_context.has_error():
            exception = data_context.get_error()
            exception_desc = exception.get_description()
            result = {
                "error_msg": str(exception_desc)
            }
            result_str = (json.dumps(result) + chr(0)).encode('utf-8').strip()
            add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
            out_data.push_back(add_buffer)
        else:
            for buffer in in_data:
                max_index = np.argmax(buffer.as_object())
                result = {
                    "predict_result": str(max_index)
                }
                result_str = (json.dumps(result) + chr(0)).encode('utf-8').strip()
                add_buffer = modelbox.Buffer(self.get_bind_device(), result_str)
                out_data.push_back(add_buffer)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()

    def data_group_pre(self, data_context):
        return modelbox.Status()

    def data_group_post(self, data_context):
        return modelbox.Status()