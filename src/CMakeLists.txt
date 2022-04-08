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

if (DEFINED ENV{USER_ROOT} AND ANDROID_ABI)
    include_directories($ENV{USER_ROOT}/${ANDROID_ABI}/usr/include)
    link_directories($ENV{USER_ROOT}/${ANDROID_ABI}/usr/lib)

    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
endif()

add_subdirectory(libmodelbox)
add_subdirectory(drivers)
add_subdirectory(develop)
add_subdirectory(python)
add_subdirectory(java)

set(MODELBOX_DEMO_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/demo")

add_subdirectory(modelbox)

if (${WITH_DEMO})
    add_subdirectory(demo)
else()
    message(STATUS "Skip buiding demos")
endif()