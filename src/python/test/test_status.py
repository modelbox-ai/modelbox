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


class TestStatus(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_status_cond(self):
        s1 = modelbox.Status()
        s2 = modelbox.Status()
        self.assertEqual(s1, s2)

        s1 = modelbox.Status(modelbox.Status.StatusCode.STATUS_SUCCESS)
        s2 = modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT)
        self.assertNotEqual(s1, s2)

        self.assertTrue(s1)
        self.assertFalse(s2)

    def test_status_message(self):
        m = "py log message"
        s = modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT, m)
        expect_msg = "code: " + s.str_code() + ", errmsg: " + m
        self.assertEqual(s.__str__(), expect_msg)


if __name__ == '__main__':
    unittest.main()
