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

set(UNIT_NAME "yolov3_inference")

set(MODEL_TOML ${CMAKE_CURRENT_BINARY_DIR}/${UNIT_NAME}.toml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}.toml.in ${MODEL_TOML} @ONLY)
install(FILES 
    ${MODEL_TOML} DESTINATION ${SOLUTION_FLOWUNIT_COMMON_DIR}
    COMPONENT solution
)

# driver test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}.toml.in ${TEST_SOLUTION_DRIVERS_DIR}/${UNIT_NAME}.toml @ONLY)