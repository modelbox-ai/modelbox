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


class TestSessionContext(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_session_set_get(self):
        session_context = modelbox.SessionContext()
        session_context.set_private("int", 1)
        session_context.set_private("double", 11.2)
        session_context.set_private("bool_true", True)
        session_context.set_private("bool_false", False)
        session_context.set_private("str", "string test")

        session_context.set_private("list1_int", [1, 2, 3, 4])
        session_context.set_private("list1_double", [1.1, 2.2, 3.3, 4.4])
        session_context.set_private("list1_bool", [True, False, False, True])
        session_context.set_private("list1_str", ["hello", "world", "!"])

        session_context.set_private("list2_int", [[1, 2], [3, 4]])
        session_context.set_private("list2_double", [[1.1, 2.2], [3.3, 4.4]])
        session_context.set_private("list2_bool", [[True, False], [False, True]])
        session_context.set_private("list2_str", [["hello", "world"], ["good", "bad"]])

        session_context.set_private("dict", {"1":1, "2":2})

        np_test = np.random.random((2, 3))
        session_context.set_private("np_test", np_test)

        self.assertEqual(session_context.get_private("int"), 1)
        self.assertEqual(session_context.get_private("double"), 11.2)
        self.assertEqual(session_context.get_private("bool_true"), True)
        self.assertEqual(session_context.get_private("bool_false"), False)
        self.assertEqual(session_context.get_private("str"), "string test")

        self.assertEqual(session_context.get_private("list1_int"), [1, 2, 3, 4])
        self.assertEqual(session_context.get_private("list1_double"), [1.1, 2.2, 3.3, 4.4])
        self.assertEqual(session_context.get_private("list1_bool"), [True, False, False, True])
        self.assertEqual(session_context.get_private("list1_str"), ["hello", "world", "!"])

        self.assertEqual(session_context.get_private("list2_int"), [[1, 2], [3, 4]])
        self.assertEqual(session_context.get_private("list2_double"), [[1.1, 2.2], [3.3, 4.4]])
        self.assertEqual(session_context.get_private("list2_bool"), [[True, False], [False, True]])
        self.assertEqual(session_context.get_private("list2_str"), [["hello", "world"], ["good", "bad"]])

        self.assertEqual(session_context.get_private("dict"), {"1":1, "2":2})

        self.assertTrue((session_context.get_private("np_test") == np_test).all())

if __name__ == '__main__':
    unittest.main()
