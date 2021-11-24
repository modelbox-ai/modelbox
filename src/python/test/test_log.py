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
import os


class TestLog(unittest.TestCase):
    def setUp(self):
        self._lock = threading.Lock()
        self._count = 0
        self._log = modelbox.Log()
        self._oldlogger = self._log.get_logger()
        self._log.reg(self.Log)
        self._log.set_log_level(modelbox.Log.Level.INFO)

    def tearDown(self):
        self._log.set_logger(self._oldlogger)
        pass

    def Log(self, level, file, lineno, func, msg):
        with self._lock:
            self._count += 1

        frame = inspect.currentframe()
        frame = frame.f_back
        info = inspect.getframeinfo(frame)

        self._msg = msg
        self.assertEqual(file, os.path.basename(info.filename))
        self.assertEqual(lineno, info.lineno)
        self.assertEqual(func, info.function)

    def test_LogPrint(self):
        msg = "Hello, world"
        modelbox.info(msg)
        self.assertEqual(msg, self._msg)

    def test_LogPrintExt(self):
        msg = "Hello, world"
        frame = inspect.currentframe()
        info = inspect.getframeinfo(frame)
        self._log.print_ext(modelbox.Log.Level.INFO, os.path.basename(info.filename),
                            info.lineno + 2, info.function, msg)
        self.assertEqual(msg, self._msg)

    def test_LogSetLevel(self):
        not_set_msg = "NOT SET"
        self._msg = not_set_msg
        msg = "Hello, world"
        self._log.set_log_level(modelbox.Log.Level.ERROR)
        modelbox.info(msg)
        self.assertNotEqual(msg, self._msg)
        self.assertEqual(not_set_msg, self._msg)

    def test_LogDefaultNoOutput(self):
        msg = "Hello, world"
        modelbox.info(msg)

    def threadtest(self, index, loop):
        for i in range(loop):
            modelbox.info("loop" +
                        str(index) + ": " + str(i))

    def test_LogMultiThread(self):
        msg = "Hello, world"
        l = []
        num = 100
        loop = 100

        for i in range(100):
            t = threading.Thread(target=self.threadtest, args=(i, loop))
            t.start()
            l.append(t)

        for t in l:
            t.join()

        self.assertEqual(loop * num, self._count)


if __name__ == '__main__':
    unittest.main()
