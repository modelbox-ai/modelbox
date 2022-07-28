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


cmake_minimum_required(VERSION 3.10.2)

project(modelbox-inference-mindspore-lite)
if (NOT MINDSPORE_LITE_FOUND) 
    message(STATUS "Not found mindspore-lite, disable mindsopre-lite flowunit")
    return()
endif()
set(CMAKE_CXX_STANDARD 17)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

set(MINDSPORE_ENGINE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB_RECURSE UNIT_SOURCE ${MINDSPORE_ENGINE_SRC_DIR}/*.cpp 
    ${MINDSPORE_ENGINE_SRC_DIR}/*.cc 
    ${MINDSPORE_ENGINE_SRC_DIR}/*.c)
group_source_test_files(MODELBOX_UNIT_SOURCE MODELBOX_UNIT_TEST_SOURCE "_test.c*" ${UNIT_SOURCE})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${MINDSPORE_ENGINE_SRC_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ASCEND_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${MINDSPORE_LITE_INCLUDE_DIR})
include_directories(${MODELBOX_COMMON_INFERENCE_INCLUDE})

set(MODELBOX_UNIT_SHARED libcommon-engine-mindspore-lite-shared)
list(APPEND MODELBOX_UNIT_SOURCE_INCLUDE ${MINDSPORE_ENGINE_SRC_DIR})
list(APPEND MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})
set(LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_SHARED ${MODELBOX_UNIT_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES 
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

set_target_properties(${MODELBOX_UNIT_SHARED}  PROPERTIES LINK_FLAGS "-Wl,-rpath,${MINDSOPRE_LITE_LIB_DIR}")

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MINDSPORE_LITE_LIBRARIES})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_COMMON_INFERENCE_LIBRARY})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "common-engine-mindspore-lite")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT cpu-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL)
install(DIRECTORY ${HEADER} 
        DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        COMPONENT cpu-device-flowunit-devel)

set(LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libcommon-engine-mindspore-lite.so CACHE INTERNAL "")

list(APPEND DRIVER_UNIT_TEST_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_UNIT_SHARED})
set(DRIVER_UNIT_TEST_INCLUDE ${DRIVER_UNIT_TEST_INCLUDE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")
