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

project(modelbox-drivers-cuda-flowunit)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

list(APPEND DRIVER_UNIT_TEST_INCLUDE ${ACL_INCLUDE_DIR})
list(APPEND DRIVER_UNIT_TEST_INCLUDE ${DSMI_INCLUDE_DIR})
set(DRIVER_UNIT_TEST_INCLUDE ${DRIVER_UNIT_TEST_INCLUDE} CACHE INTERNAL "")
list(APPEND DRIVER_UNIT_TEST_LINK_LIBRARIES ${ACL_LIBRARIES})
list(APPEND DRIVER_UNIT_TEST_LINK_LIBRARIES ${DSMI_LIBRARIES})
set(DRIVER_UNIT_TEST_LINK_LIBRARIES ${DRIVER_UNIT_TEST_LINK_LIBRARIES} CACHE INTERNAL "")