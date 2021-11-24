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


import modelbox
import datetime

__log = modelbox.Log()


def LogCallback(level, file, lineno, func, msg):
    print("[{time}][{level}][{file:>20}:{lineno:>4}] {msg}".format(
        time=datetime.datetime.now(), level=level,
        file=file, lineno=lineno, msg=msg
    ))


def RegLog():
    __log.reg(LogCallback)
    __log.set_log_level(modelbox.Log.Level.INFO)


def SetLogLevel(level):
    __log.set_log_level(level)

RegLog()
