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

cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "custom_resize")

set(FLOWUNIT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py)
set(FLOWUNIT_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml)

install(FILES 
    ${FLOWUNIT_FILE}
    ${FLOWUNIT_CONFIG}
    DESTINATION ${SOLUTION_FLOWUNIT_EMOTION_DIR}/${FLOWUNIT_NAME}
    COMPONENT solution
)
