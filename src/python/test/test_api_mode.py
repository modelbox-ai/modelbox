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
    input = ctx.Input("In_1")
    output = ctx.output("Out_1")
    buffer = input[0]
    output.push_back(buffer)
    return modelbox.Status.StatusCode.STATUS_SUCCESS

class TestDynamicGraph(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_api_graph(self):
        engine = modelbox.FlowGraphDesc()
        self.assertNotEqual(engine, None)
        config = modelbox.Configuration()
        config.set("graph.queue_size","32")
        config.set("graph.queue_size_external","1000")
        config.set("graph.batch_size","16")
        config.set("drivers.skip-default", "true")
        config.set("drivers.dir", [test_config.TEST_DRIVER_DIR])
        engine.init(config)
        video_demuxer_output = engine.addnode("video_demuxer",{},{})
        
        engine.bindinput(video_demuxer_output,"in_video_url")
        engine.bindoutput(video_demuxer_output,"out_video_packet")
        flow = modelbox.Flow()
        flow.init(engine)
        flow.build()
        flow.run_async()
      
        retval = modelbox.Status()
        ret = flow.wait(1000, retval)
    
    def test_callback(self):
        desc = modelbox.FlowGraphDesc()
        self.assertNotEqual(desc, None)
        config = modelbox.Configuration()
        config.set("graph.queue_size","32")
        config.set("graph.queue_size_external","1000")
        config.set("graph.batch_size","16")
        config.set("drivers.skip-default", "true")
        config.set("drivers.dir", [test_config.TEST_DRIVER_DIR])
        desc.init(config)

        resize_output = desc.addnode("resize", {"width": "256", "height": "256"},{})
        out_image = resize_output.get_nodedesc("out_image")
        callback_output = desc.addnode(callback_func, ["In_1"],["Out_1"],{"In_1":out_image})
        desc.bindinput(resize_output,"__default__inport__")
        desc.bindoutput(callback_output,"__default__outport__")
        flow = modelbox.Flow()
        flow.init(desc)
        flow.build()
        flow.run_async()
        retval = modelbox.Status()
        flow.wait(1000, retval)
        
if __name__ == '__main__':
    unittest.main()