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

file(GLOB SOURCES *.cpp *.cc *.c)

set(INCLUDE ${CMAKE_CURRENT_LIST_DIR})
SET(HEADER ${CMAKE_CURRENT_LIST_DIR}/model_decrypt_interface.h)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_CONFIG_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})

install(FILES ${HEADER} 
    DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/modelbox/inference 
    COMPONENT cpu-device-flowunit-devel
    )

set(LIBRARY modelbox-common-inference-object)
add_library(${LIBRARY} STATIC ${SOURCES})
set_property(TARGET ${LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(MODELBOX_COMMON_INFERENCE_LIBRARY ${LIBRARY} CACHE INTERNAL "")
set(MODELBOX_COMMON_INFERENCE_INCLUDE ${INCLUDE} CACHE INTERNAL "")