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

set(UNIT_NAME "tensorrt-inference-plugin")
project(modelbox-flowunit-${UNIT_NAME})

file(GLOB PLUGIN_SOURCE *.cpp *.cc *.c)

if (NOT TENSORRT_FOUND) 
    message(STATUS "Not found tensorrt, disable ${UNIT_NAME} flowunit")
    return()
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_CONFIG_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${TENSORRT_INCLUDE_DIR})


set(PLUGIN_SHARED tensorrt-plugin)
add_library(${PLUGIN_SHARED} SHARED ${PLUGIN_SOURCE})

target_link_libraries(${PLUGIN_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${PLUGIN_SHARED} ${TENSORRT_LIBRARIES})

set_target_properties(${PLUGIN_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${PLUGIN_SHARED}")

list(APPEND DRIVER_UNIT_TEST_TARGET ${PLUGIN_SHARED})
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")
