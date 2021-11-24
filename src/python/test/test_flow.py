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


class TestFlow(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_build_failed(self):
        conf_file = test_config.TEST_DATA_DIR + "/py_config.toml"
        driver_dir = test_config.TEST_DRIVER_DIR
        
        txt = r"""
[driver]
dir=["{}"]
skip-default=true
[log]
level="ERROR"
[graph]
graphconf = '''digraph demo {{                                                                            
    notexist[type=flowunit, flowunit=notexist, device=cpu]                                                                                                                
}}'''
format = "graphviz"
""".format(driver_dir)

        conf = modelbox.Configuration()
        flow = modelbox.Flow()
        ret = flow.init("graph", txt)
        if ret == False:
            self.assertTrue(ret)
            
        self.assertTrue(ret)
        ret = flow.build()
        if ret == False:
            modelbox.error(ret)
        self.assertFalse(ret)
        
    @unittest.skip("disable thread for no delete")
    def test_flow_op(self):
        conf_file = test_config.TEST_DATA_DIR + "/py_op_config.toml"
        driver_dir = test_config.TEST_DRIVER_DIR
        with open(conf_file, "w") as out:
            txt = r"""
[driver]
dir=["{}", "{}"]
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                            
    python_image[type=flowunit, flowunit=python_image, device=cpu, deviceid=0, label="<image_out/out_1>", batch_size = 10]   
    python_resize[type=flowunit, flowunit=python_resize, device=cpu, deviceid=0, label="<resize_in> | <resize_out>"]   
    python_brightness[type=flowunit, flowunit=python_brightness, device=cpu, deviceid=0, label="<brightness_in> | <brightness_out>", brightness = 0.1]  
    python_show[type=flowunit, flowunit=python_show, device=cpu, deviceid=0, label="<show_in>", is_save = true]    
    python_image:"image_out/out_1" -> python_resize:resize_in
    python_resize:resize_out -> python_brightness:brightness_in
    python_brightness:brightness_out -> python_show:show_in                                                                                              
}}'''
format = "graphviz"
""".format(driver_dir, test_config.TEST_DATA_DIR + "/python_op")
            out.write(txt)

        conf = modelbox.Configuration()
        flow = modelbox.Flow()
        ret = flow.init(conf_file)
        os.remove(conf_file)
        if ret == False:
            modelbox.error(ret)
        self.assertTrue(ret)
        ret = flow.build()
        self.assertTrue(ret)
        ret = flow.run_async()
        self.assertTrue(ret)
        retval = modelbox.Status()
        ret = flow.wait(0, retval)
        self.assertEqual(retval, modelbox.Status.StatusCode.STATUS_STOP)

    def test_flow_op_thread(self):
        conf_file = test_config.TEST_DATA_DIR + "/py_op_config.toml"
        driver_dir = test_config.TEST_DRIVER_DIR
        with open(conf_file, "w") as out:
            txt = r"""
[driver]
dir=["{}", "{}"]
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                            
    python_image[type=flowunit, flowunit=python_image, device=cpu, deviceid=0, label="<image_out/out_1>", batch_size = 10]   
    python_resize[type=flowunit, flowunit=python_resize, device=cpu, deviceid=0, label="<resize_in> | <resize_out>"]   
    python_brightness[type=flowunit, flowunit=python_brightness, device=cpu, deviceid=0, label="<brightness_in> | <brightness_out>", brightness = 0.1]  
    python_show[type=flowunit, flowunit=python_show, device=cpu, deviceid=0, label="<show_in>", is_save = false]    
    python_image:"image_out/out_1" -> python_resize:resize_in
    python_resize:resize_out -> python_brightness:brightness_in
    python_brightness:brightness_out -> python_show:show_in                                                                                              
}}'''
format = "graphviz"
""".format(driver_dir, test_config.TEST_DATA_DIR + "/python_op")
            out.write(txt)

        conf = modelbox.Configuration()
        flow1 = modelbox.Flow()
        flow2 = modelbox.Flow()
        t1 = threading.Thread(target=self.thread_func,
                              args=(flow1, conf_file,))
        t2 = threading.Thread(target=self.thread_func,
                              args=(flow2, conf_file,))
        t1.setDaemon(True)
        t2.setDaemon(True)
        
        t1.start()
        t2.start()

        t1.join()
        t2.join()
        os.remove(conf_file)


    def thread_func(self, flow, conf_file):
        ret = flow.init(conf_file)
        if ret == False:
            modelbox.error(ret)
        self.assertTrue(ret)
        ret = flow.build()
        self.assertTrue(ret)
        ret = flow.run_async()
        self.assertTrue(ret)
        retval = modelbox.Status()
        ret = flow.wait(0, retval)
        self.assertEqual(retval, modelbox.Status.StatusCode.STATUS_STOP)

    def test_flow_op_ext(self):
        conf_file = test_config.TEST_DATA_DIR + "/py_op_config.toml"
        driver_dir = test_config.TEST_DRIVER_DIR
        with open(conf_file, "w") as out:
            txt = r"""
[driver]
dir=["{}", "{}"]
skip-default=true
[log]
level="INFO"
[graph]
graphconf = '''digraph demo {{                                                                            
    input1[type=input]   
    python_resize[type=flowunit, flowunit=python_resize, device=cpu, deviceid=0, label="<resize_in> | <resize_out>"]   
    python_brightness[type=flowunit, flowunit=python_brightness, device=cpu, deviceid=0, label="<brightness_in> | <brightness_out>", brightness = 0.1]  
    output1[type=output]   
    input1 -> python_resize:resize_in
    python_resize:resize_out -> python_brightness:brightness_in
    python_brightness:brightness_out -> output1                                                                                             
}}'''
format = "graphviz"
""".format(driver_dir, test_config.TEST_DATA_DIR + "/python_op")
            out.write(txt)

        conf = modelbox.Configuration()
        flow = modelbox.Flow()
        ret = flow.init(conf_file)
        os.remove(conf_file)
        if ret == False:
            modelbox.error(ret)
        self.assertTrue(ret)
        ret = flow.build()
        self.assertTrue(ret)
        ret = flow.run_async()
        self.assertTrue(ret)

        img = cv2.imread(test_config.TEST_SOURCE_DIR + "/../src/python/test/data/liu-x-160.jpg")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        extern_data_map = flow.create_external_data_map()
        buffer_list = extern_data_map.create_buffer_list()
        im_array = np.asarray(img_rgb[:,:])      
        buffer_list.push_back(im_array)
        extern_data_map.send("input1", buffer_list)
        extern_data_map.shutdown()

        buffer_list_map = modelbox.ExtOutputBufferList()
        ret = extern_data_map.recv(buffer_list_map)
        self.assertTrue(ret)
        
        result_buffer_list = buffer_list_map.get_buffer_list("output1")

        for i in range(result_buffer_list.size()):
            aa = result_buffer_list[i]
            np_image = np.array(aa, copy= False)
            image = Image.fromarray(np_image)
            with Image.open(test_config.TEST_SOURCE_DIR + "/../src/python/test/data/python_test_show_out.png") as check_image:
                try:
                    diff = ImageChops.difference(image, check_image)
                    self.assertEqual(diff.getbbox(), None)
                except ValueError as e:
                    self.assertTrue(False)
        flow.stop()


if __name__ == '__main__':
    unittest.main()
