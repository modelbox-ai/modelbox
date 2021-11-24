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

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

file(GLOB MOCK_DRIVER_SOURCE *.cpp *.cc *.c)
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(device_mockdevice)
add_subdirectory(flowunit_mockflowunit)
add_subdirectory(graph_conf_mockgraphconf)

add_definitions(-DTEST_DEVICE_MOCKDEVICE_PATH="${LIBMODELBOX_DEVICE_MOCKDEVICE_SO}")
add_definitions(-DTEST_FLOWUNIT_MOCKFLOWUNIT_PATH="${LIBMODELBOX_FLOWUNIT_MOCKFLOWUNIT_SO}")
add_definitions(-DTEST_GRAPHCONF_MOCKGRAPHCONF_PATH="${LIBMODELBOX_GRAPHCONF_MOCKGRAPHCONF_SO}")

set(MOCK_DRIVER_CTRL_LIB mock-driver-ctrl-lib)
set(MOCK_DRIVER_CTRL_LIB ${MOCK_DRIVER_CTRL_LIB} CACHE INTERNAL "")
set(MOCK_DRIVER_CTRL_INCLUDE ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

add_library(${MOCK_DRIVER_CTRL_LIB} ${MOCK_DRIVER_SOURCE})
target_link_libraries(${MOCK_DRIVER_CTRL_LIB} dl)

add_dependencies(${MOCK_DRIVER_CTRL_LIB} ${LIBMODELBOX_DEVICE_MOCKDEVICE_SHARED})
add_dependencies(${MOCK_DRIVER_CTRL_LIB} ${LIBMODELBOX_FLOWUNIT_MOCKFLOWUNIT_SHARED})
add_dependencies(${MOCK_DRIVER_CTRL_LIB} ${LIBMODELBOX_GRAPHCONF_MOCKGRAPHCONF_SHARED})

