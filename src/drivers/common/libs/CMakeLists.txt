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

project(modelbox-drivers-comm-lib)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

set(HEADER ${CMAKE_CURRENT_LIST_DIR}/include)

include_directories(${HEADER})

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

install(DIRECTORY ${HEADER}/modelbox
    DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR} 
    COMPONENT cpu-device-flowunit-devel
    )

set(LIBMODELBOX_DRIVER_COMMON_LIB_INCLUDE ${HEADER} CACHE INTERNAL "")
list(APPEND DRIVER_UNIT_TEST_INCLUDE ${HEADER})
set(DRIVER_UNIT_TEST_INCLUDE ${DRIVER_UNIT_TEST_INCLUDE} CACHE INTERNAL "")
