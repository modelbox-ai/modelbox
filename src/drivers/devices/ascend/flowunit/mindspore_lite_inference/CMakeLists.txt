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

set(UNIT_DEVICE "ascend")
set(UNIT_NAME "mindspore-lite-inference")

project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME})

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if (NOT MINDSPORE_LITE_FOUND) 
    message(STATUS "Not found mindspore-lite, disable ${UNIT_NAME} flowunit")
    return()
endif()
set(CMAKE_CXX_STANDARD 17)

set(MINDSPORE_FLOWUNIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UNIT_SOURCE 
    ${MINDSPORE_FLOWUNIT_SOURCE_DIR}/*.cpp 
    ${MINDSPORE_FLOWUNIT_SOURCE_DIR}/*.cc 
    ${MINDSPORE_FLOWUNIT_SOURCE_DIR}/*.c)
group_source_test_files(MODELBOX_UNIT_SOURCE MODELBOX_UNIT_TEST_SOURCE "_test.c*" ${UNIT_SOURCE})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${MINDSPORE_ENGINE_SRC_DIR})
include_directories(${MINDSPORE_FLOWUNIT_SOURCE_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ASCEND_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${MINDSPORE_LITE_INCLUDE_DIR})
include_directories(${ACL_INCLUDE_DIR})
include_directories(${LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_INCLUDE})
include_directories(${MODELBOX_COMMON_INFERENCE_INCLUDE})

set(MODELBOX_UNIT_SHARED libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-shared)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${MINDSPORE_FLOWUNIT_SOURCE_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_toml/modelbox.test.mindspore_lite.ascend.inference.in ${CMAKE_BINARY_DIR}/test/test-working-dir/data/virtual_mindspore_lite_ascend_infer_test.toml @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_toml/modelbox.test.mindspore_lite.ascend.inference.encrypt.in ${CMAKE_BINARY_DIR}/test/test-working-dir/data/virtual_mindspore_lite_ascend_infer_test_en.toml @ONLY)


add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})
set(LIBMODELBOX_FLOWUNIT_INFERENCE_ASCEND_SHARED ${MODELBOX_UNIT_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES 
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_compile_options(${MODELBOX_UNIT_SHARED} PUBLIC -fvisibility=hidden)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_DEVICE_ASCEND_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_FLOWUNIT_INFERENCE_MINDSPORE_LITE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_COMMON_INFERENCE_LIBRARY})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT ascend-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL)
install(DIRECTORY ${HEADER} 
        DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        COMPONENT ascend-device-flowunit-devel)

set(LIBMODELBOX_FLOWUNIT_MINDSPORE_LITE_ASCEND_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_MINDSPORE_LITE_ASCEND_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_MINDSPORE_LITE_ASCEND_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_MINDSPORE_LITE_ASCEND_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}.so CACHE INTERNAL "")

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_UNIT_SHARED})
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")