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


class TestBuffer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_flow_for_buffer(self):
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
    python_buffer[type=flowunit, flowunit=python_buffer, device=cpu, deviceid=0, label="<buffer_in> | <buffer_out>", buffer_config = 0.2]  
    output1[type=output]   
    input1 -> python_buffer:buffer_in
    python_buffer:buffer_out -> output1                                                                                             
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
        with Image.open(test_config.TEST_SOURCE_DIR + "/../src/python/test/data/liu-x-160.jpg") as img:
            img_np = np.array(img)

        extern_data_map = flow.create_external_data_map()
        buffer_list = extern_data_map.create_buffer_list()
        buffer_list.push_back(img_np)
        extern_data_map.send("input1", buffer_list)
        extern_data_map.shutdown()
        
        buffer_list_map = modelbox.ExtOutputBufferList()
        ret = extern_data_map.recv(buffer_list_map)
        self.assertTrue(ret)

        result_buffer_list = buffer_list_map.get_buffer_list("output1")

        for i in range(result_buffer_list.size()):
            buffer = result_buffer_list[i]
            np_image = np.array(buffer, copy= False)
            image = Image.fromarray(np_image)
            with Image.open(test_config.TEST_SOURCE_DIR + "/../src/python/test/data/liu-x-160.jpg") as check_image:
                try:
                    check_image_np = np.array(check_image)
                    diff = ImageChops.difference(image, Image.fromarray(check_image_np))
                    self.assertEqual(diff.getbbox(), None)
                except ValueError as e:
                    flow.stop()
                    self.assertTrue(False)

            data_type = buffer.get("type")
            if data_type != modelbox.Buffer.ModelBoxDataType.UINT8:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid type")

            float_test = buffer.get("float_test")
            if float_test != 0.5:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid float test")

            string_test = buffer.get("string_test")
            if string_test != "TEST":
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid string test")

            int_test = buffer.get("int_test")
            if int_test != 100:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid int test")

            bool_test = buffer.get("bool_test")
            if bool_test != False:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid bool test")

            int_list = buffer.get("list_int_test")
            if int_list != [1, 1, 1]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid int list")

            float_list = buffer.get("list_float_test")
            if float_list != [0.1, 0.2, 0.3]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid float list")

            bool_list = buffer.get("list_bool_test")
            if bool_list != [False, False, True]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid bool list")

            string_list = buffer.get("list_string_test")
            if string_list != ["TEST1", "TEST2", "TEST3"]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid string list")

            int_list2 = buffer.get("list2_int_test")
            if int_list2 != [[1, 2], [3, 4]]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid 2D int list")

            float_list2 = buffer.get("list2_float_test")
            if float_list2 != [[1.1, 2.2], [3.3, 4.4]]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid 2D float list")

            bool_list2 = buffer.get("list2_bool_test")
            if bool_list2 != [[True, False], [False, True]]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid 2D bool list")

            string_list2 = buffer.get("list2_string_test")
            if string_list2 != [["hello", "world"], ["good", "bad"]]:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid 2D string list")

            np_set_test = np.array([[1, 2 ,3], [11, 12, 13]])
            np_get_test = buffer.get("np_test")
            if not (np_set_test == np_get_test).all():
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid np test")

            try:
                dict_test = buffer.get("map_test")
            except ValueError as err:
                modelbox.info(str(err))
            else:
                flow.stop()
                self.assertTrue(False)

        flow.stop()

if __name__ == '__main__':
    unittest.main()
