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


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_config_set_get(self):
        c = modelbox.Configuration()
        c.set("1", 1)
        c.set("2", 1.2)
        c.set("3", 1.3)
        c.set("4", False)
        c.set("5", True)
        c.set("6", "test")

        self.assertEqual(c.get_int("1"), 1)
        self.assertEqual(c.get_float("2"), 1.2)
        self.assertEqual(c.get_float("3"), 1.3)
        self.assertEqual(c.get_bool("4"), False)
        self.assertEqual(c.get_bool("5"), True)
        self.assertEqual(c.get_string("6"), "test")


if __name__ == '__main__':
    unittest.main()
