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
import sys
import threading
import modelbox
import inspect
from test import test_config

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
        input_stream =  engine.create_input({"input"})
        source_url = f'{test_config.TEST_ASSETS}/video/jpeg_5s_480x320_24fps_yuv444_8bit.mp4'
        input_stream.setmeta("source_url",source_url)
        input_stream.close()

        video_demuxer_output = engine.execute("video_demuxer",{},input_stream)
        frame_num = 0
        for packet in video_demuxer_output:
            frame_num = frame_num + 1
        
        engine.shutdown()

if __name__ == '__main__':
    unittest.main()