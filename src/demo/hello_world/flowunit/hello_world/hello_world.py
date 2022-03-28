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
import json
import time

def addTimestamp(msg):
    local_time = time.asctime(time.localtime(time.time()))
    return '{} {}'.format(msg, str(local_time))

class HelloWorld(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        in_data = data_context.input("In_1")
        out_data = data_context.output("Out_1")

        for buffer in in_data:
            request_body = json.loads(buffer.as_object().strip(chr(0)))
            msg = request_body.get("msg")
            msg = msg.title()
            msg = addTimestamp(msg)

            out_string = msg + chr(0)
            out_buffer = modelbox.Buffer(self.get_bind_device(), out_string.encode('utf-8').strip())
            out_data.push_back(out_buffer)

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
