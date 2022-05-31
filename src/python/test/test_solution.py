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
import os
import cv2
import unittest
import modelbox
import numpy as np
from test import test_config
from PIL import Image
from PIL import ImageChops


class TestSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution_name = "solution_test"
        cls.solution_file_name = "solution_test.toml"
        cls.solution_default_dir = "/usr/local/share/modelbox/solutions/graphs"
        cls.src_image = os.path.join(os.path.dirname(__file__), "data/liu-x-160.jpg")
        cls.dst_image = os.path.join(os.path.dirname(__file__), "data/python_test_show_out.png")

    def test_solution_set_name(self):
        solution = modelbox.Solution(self.solution_name)
        solution_name = solution.get_solution_name()
        self.assertEqual(solution_name, self.solution_name)

    def test_solution_get_default_dir(self):
        solution = modelbox.Solution(self.solution_name)
        solution_dir = solution.get_solution_dir()
        self.assertEqual(solution_dir, self.solution_default_dir)
        
    def test_solution_set_dir(self):
        solution = modelbox.Solution(self.solution_name)
        solution.set_solution_dir(test_config.TEST_DATA_DIR)
        solution_dir = solution.get_solution_dir()
        self.assertEqual(solution_dir, test_config.TEST_DATA_DIR)

    def test_solution_op_exec(self):
        toml_file_path = os.path.join(test_config.TEST_DATA_DIR, self.solution_file_name)
        txt = r"""
[driver]
dir=["{}", "{}"]
skip-default=true
[flow]
name = "{}"
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
""".format(test_config.TEST_DRIVER_DIR, os.path.join(test_config.TEST_DATA_DIR, "python_op"), self.solution_name)
        with open(toml_file_path, "w") as fp:
            fp.write(txt)

        solution = modelbox.Solution(self.solution_name)
        solution.set_solution_dir(test_config.TEST_DATA_DIR)
        flow = modelbox.Flow()
        ret = flow.init(solution)
        self.assertTrue(ret)
        ret = flow.build()
        self.assertTrue(ret)
        ret = flow.run_async()
        self.assertTrue(ret)
        
        os.remove(toml_file_path)
        
        img = cv2.imread(self.src_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        extern_data_map = flow.create_external_data_map()
        buffer_list = extern_data_map.create_buffer_list()
        im_array = np.asarray(img_rgb[:, :])
        buffer_list.push_back(im_array)
        extern_data_map.send("input1", buffer_list)
        extern_data_map.shutdown()

        buffer_list_map = modelbox.ExtOutputBufferList()
        ret = extern_data_map.recv(buffer_list_map)
        self.assertTrue(ret)

        result_buffer_list = buffer_list_map.get_buffer_list("output1")
        self.assertTrue(result_buffer_list.size() == 1)
        np_image = np.array(result_buffer_list[0], copy=False)
        image = Image.fromarray(np_image)
        with Image.open(self.dst_image) as check_image:
            try:
                diff = ImageChops.difference(image, check_image)
                self.assertEqual(diff.getbbox(), None)
            except ValueError as e:
                self.assertTrue(False)
        flow.stop()


if __name__ == '__main__':
    unittest.main()
