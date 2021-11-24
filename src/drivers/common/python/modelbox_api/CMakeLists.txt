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

file(GLOB_RECURSE SOURCES *.cpp *.cc *.c)

if(NOT ${PYTHONLIBS_FOUND})
    message(STATUS "Not found python, disable python api")
    return()
endif()

set(INCLUDE ${CMAKE_CURRENT_LIST_DIR})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_CONFIG_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIRS})

set(LIBRARY modelbox-common-modelbox-api-object)
add_library(${LIBRARY} STATIC ${SOURCES})
target_compile_options(${LIBRARY} PUBLIC ${PYBIND11_CPP_STANDARD})
target_compile_options(${LIBRARY} PUBLIC -fvisibility=hidden)
set_property(TARGET ${LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

set(MODELBOX_COMMON_MODELBOX_API_LIBRARY ${LIBRARY} CACHE INTERNAL "")
set(MODELBOX_COMMON_MODELBOX_API_INCLUDE ${INCLUDE} CACHE INTERNAL "")
