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

project(modelbox)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if (STANDALONE)
    set(MODELBOX_ROOT_VAR "\${MODELBOX_ROOT}")
endif()

add_subdirectory(common)
add_subdirectory(manager)
add_subdirectory(server)
add_subdirectory(serving)
add_subdirectory(tool)

set(MODELBOX_PROG_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
set(MODELBOX_PROG_INCLUDE 
    ${MODELBOX_PROG_INCLUDE} 
    CACHE INTERNAL "")
