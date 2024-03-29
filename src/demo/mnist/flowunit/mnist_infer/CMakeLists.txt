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

set(FLOWUNIT_NAME "mnist_infer")

set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
set(MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.pb)
set(TRAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/train.py)
set(TRAIN_SH_FILE ${CMAKE_CURRENT_SOURCE_DIR}/train.sh)
set(FLOWUNIT_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml)

configure_file(${MODEL_FILE} ${FLOWUNIT_PATH}/mnist_model.pb COPYONLY)
configure_file(${TRAIN_FILE} ${FLOWUNIT_PATH}/train.py COPYONLY)
configure_file(${TRAIN_SH_FILE} ${FLOWUNIT_PATH}/train.sh COPYONLY)
configure_file(${FLOWUNIT_CONFIG} ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)
 
install(DIRECTORY
    ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_MNIST_FLOWUNIT_DIR}
    COMPONENT demo
)
