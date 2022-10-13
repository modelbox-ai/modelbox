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

from test import test_config


class TestPyModel(unittest.TestCase):
    def setUp(self):
        self.model = modelbox.Model(test_config.TEST_DATA_DIR + "/python_op", "python_brightness", 1, "cpu", "0")
        self.model.add_path(test_config.TEST_DRIVER_DIR)
        ret = self.model.start()
        self.assertEqual(ret.code(), modelbox.Status.StatusCode.STATUS_SUCCESS)

    def tearDown(self):
        self.model.stop()

    def test_model_infer(self):
        for i in range(0, 10):
            data = np.zeros((32, 32, 3), dtype=np.uint8)
            result = self.model.infer([data])
            self.assertEqual(len(result), 1)
            self.assertEqual(np.array(result[0]).shape, (32, 32, 3))

    def test_model_infer_batch(self):
        data = []
        batch_size = 10
        for i in range(0, batch_size):
            data.append(np.zeros((32, 32, 3), dtype=np.uint8))
        result = self.model.infer_batch([data])
        self.assertEqual(len(result), 1)
        port0_result = result[0]
        self.assertEqual(len(port0_result), batch_size)
        for i in range(0, batch_size):
            self.assertEqual(np.array(port0_result[0]).shape, (32, 32, 3))

if __name__ == '__main__':
    unittest.main()
