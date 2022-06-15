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

from time import sleep
import unittest
import modelbox
import numpy as np
import os
import threading
from test import test_config
from PIL import Image
from PIL import ImageChops
import cv2

def callback_func(ctx):
    input = ctx.input("in1")
    input1 = ctx.input("in2")
    output = ctx.output("out1")
    buffer1 = input[0]
    buffer2 = input1[0]
    array1 = np.array(buffer1)
    array2 = np.array(buffer2)
    output.push_back(array1 + array2)
    return modelbox.Status.StatusCode.STATUS_SUCCESS

class TestAPIMode(unittest.TestCase):
    def setUp(self):
        flow_cfg = modelbox.FlowConfig()
        flow_cfg.set_queue_size(32)
        flow_cfg.set_batch_size(8)
        flow_cfg.set_skip_default_drivers(True)
        flow_cfg.set_drivers_dir([test_config.TEST_DRIVER_DIR])
        self.graph_desc = modelbox.FlowGraphDesc()
        self.graph_desc.init(flow_cfg)

    def tearDown(self):
        pass

    def test_add_node(self):
        source_url = test_config.TEST_ASSETS + "/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4"
        
        input = self.graph_desc.add_input("input1")
        video_demuxer = self.graph_desc.add_node("video_demuxer", "cpu", input)
        self.graph_desc.add_output("output1", video_demuxer)
        
        flow = modelbox.Flow()
        flow.init(self.graph_desc)
        flow.start_run()

    def test_add_function(self):
        input1 = self.graph_desc.add_input("input1")
        input2 = self.graph_desc.add_input("input2")
        func_node = self.graph_desc.add_function(callback_func, ["in1", "in2"], ["out1"], {"in1": input1[0], "in2": input2[0]})
        self.graph_desc.add_output("output1", func_node)

        flow = modelbox.Flow()
        flow.init(self.graph_desc)
        flow.start_run()

        data = np.array([1, 1])
        data_map = flow.create_external_data_map()
        buffer_list = data_map.create_buffer_list()
        buffer_list.push_back(data)
        data_map.send("input1", buffer_list)
        data_map.send("input2", buffer_list)
        result = modelbox.ExtOutputBufferList()
        data_map.recv(result)
        buffer_list = result.get_buffer_list("output1")
        buffer = buffer_list[0]
        out_data = np.array(buffer)
        self.assertEqual(out_data[0], 2)
        self.assertEqual(out_data[1], 2)


if __name__ == '__main__':
    unittest.main()