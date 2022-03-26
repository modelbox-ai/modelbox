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

    def test_dynamic_graph(self):
        engine = modelbox.ModelBoxEngine()
        self.assertNotEqual(engine, None)
        config = modelbox.Configuration()
        config.set("graph.queue_size","32")
        config.set("graph.queue_size_external","1000")
        config.set("graph.batch_size","16")
        config.set("drivers.skip-default", "true")
        config.set("drivers.dir", [test_config.TEST_DRIVER_DIR])
        engine.init(config)
        source_url = f'{test_config.TEST_ASSETS}/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4'
        video_demuxer_output = engine.execute("video_demuxer",{},{})
        
        engine.bindinput(video_demuxer_output,"in_video_url")
        engine.bindoutput(video_demuxer_output,"out_video_packet")
        engine.run()
        engine.close()
    
    def test_callback(self):
        engine = modelbox.ModelBoxEngine()
        self.assertNotEqual(engine, None)
        config = modelbox.Configuration()
        config.set("graph.queue_size","32")
        config.set("graph.queue_size_external","1000")
        config.set("graph.batch_size","16")
        config.set("drivers.skip-default", "true")
        config.set("drivers.dir", [test_config.TEST_DRIVER_DIR])
        engine.init(config)

        resize_output = engine.execute("resize", {"width": "256", "height": "256"},{})
        sadas = resize_output.get_datahandler("out_image")
        callback_output = engine.execute(callback_func, ["In_1"],["Out_1"],{"In_1":sadas})
        engine.bindinput(resize_output,"__default__inport__")
        engine.bindoutput(callback_output,"__default__outport__")
        engine.run()
        engine.close()
        
            
    

if __name__ == '__main__':
    unittest.main()