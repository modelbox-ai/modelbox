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

set(UNIT_DEVICE "cpu")
set(UNIT_NAME "face_draw")

project(modelbox-unit-${UNIT_NAME}-${UNIT_DEVICE})

find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(STATUS "Not found opencv, disable face_draw flowunit")
    return()
endif()

file(GLOB_RECURSE MODELBOX_UNIT_SOURCE *.cpp *.cc *.c)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})

set(MODELBOX_UNIT_LINK_LIBRARY ${OpenCV_LIBS})
set(MODELBOX_UNIT_SHARED libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-shared)
add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})

target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_UNIT_LINK_LIBRARY})
target_link_libraries(${MODELBOX_UNIT_SHARED} modelbox)
target_link_libraries(${MODELBOX_UNIT_SHARED} modelbox-device-${UNIT_DEVICE})
target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)

if ( NOT "${LIBMODELBOX_SHARED}" STREQUAL "")
    add_dependencies(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
    add_dependencies(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_DEVICE_CPU_SHARED})
endif()
set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-${UNIT_NAME}")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT solution
        RUNTIME DESTINATION ${SOLUTION_FLOWUNIT_FACE_DIR}
        LIBRARY DESTINATION ${SOLUTION_FLOWUNIT_FACE_DIR}
        OPTIONAL
        )


# driver test
list(APPEND DRIVER_SOLUTION_TEST_TARGET ${MODELBOX_UNIT_SHARED})
set(DRIVER_SOLUTION_TEST_TARGET ${DRIVER_SOLUTION_TEST_TARGET} CACHE INTERNAL "")
