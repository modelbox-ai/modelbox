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

project(output_broker_plugin)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if(OBS_FOUND)
    add_subdirectory(obs_output_broker)
else()
    message(STATUS "Not found obs library, disable obs output broker plugin")
endif()

if(DIS_FOUND)
    add_subdirectory(dis_output_broker)
else()
    message(STATUS "Not found dis library, disable dis output broker plugin")
endif()

if(CPPREST_FOUND)
    add_subdirectory(webhook_output_broker)
else()
    message(STATUS "Not found cpprest library, disable webhook output broker plugin")
endif()
