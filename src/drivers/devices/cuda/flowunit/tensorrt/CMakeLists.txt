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

set(UNIT_DEVICE "cuda")
set(UNIT_NAME "tensorrt_inference")

project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME})

if (NOT TENSORRT_FOUND) 
    message(STATUS "Not found tensorrt, disable ${UNIT_NAME} flowunit")
    return()
endif()

if (${TENSORRT_VERSION} STREQUAL "7.1.3")
    message("found tensorrt version is " ${TENSORRT_VERSION} " define tensorrt7")
    ADD_DEFINITIONS(-DTENSORRT7)
endif()

if (${TENSORRT_VERSION} STRGREATER_EQUAL "8.0.0" AND ${TENSORRT_VERSION} STRLESS "9.0.0")
    message("found tensorrt version is " ${TENSORRT_VERSION} " define tensorrt8")
    add_definitions(-DTENSORRT8)
    set(TRT_VERSION8 TRUE)
endif()

file(GLOB UNIT_SOURCE *.cpp *.cc *.c *.cu)
if (NOT TRT_VERSION8)
    file(GLOB NV_PLUGIN_FILES ./nvplugin/*.cc ./nvplugin/*.cpp ./nvplugin/*.cu)
endif()
list(APPEND UNIT_SOURCE ${NV_PLUGIN_FILES})
SET(HEADER tensorrt_inference_plugin.h)

group_source_test_files(MODELBOX_UNIT_SOURCE MODELBOX_UNIT_TEST_SOURCE "_test.c*" ${UNIT_SOURCE})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_toml/modelbox.test.tensorrt.in ${CMAKE_BINARY_DIR}/test/test-working-dir/data/virtual_tensorrt_test.toml @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_toml/modelbox.test.tensorrt.encrypt.in ${CMAKE_BINARY_DIR}/test/test-working-dir/data/virtual_tensorrt_encrypt_test.toml @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_toml/modelbox.test.plugin.tensorrt.in ${CMAKE_BINARY_DIR}/test/test-working-dir/data/virtual_plugin_tensorrt_test.toml @ONLY)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CUDA_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${MODELBOX_COMMON_INFERENCE_INCLUDE})

# for supress c++ compile wanring
set(MODELBOX_UNIT_SHARED libmodelbox_unit_${UNIT_DEVICE}_${UNIT_NAME}_shared)
set(MODELBOX_UNIT_SHARED_OUTPUT_NAME libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-shared)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

cuda_add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})
set(LIBMODELBOX_FLOWUNIT_INFERENCE_CUDA_SHARED ${MODELBOX_UNIT_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES 
    OUTPUT_NAME ${MODELBOX_UNIT_SHARED_OUTPUT_NAME}
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_DEVICE_CUDA_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${TENSORRT_LIBRARIES})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${CUDA_CUDART_LIBRARY})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_COMMON_INFERENCE_LIBRARY})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}")

install(TARGETS ${MODELBOX_UNIT_SHARED}
        COMPONENT cuda-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL)
install(FILES ${HEADER} 
        DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/modelbox/drivers/devices/cuda/flowunit/tensorrt
        COMPONENT cuda-device-flowunit-devel)

set(LIBMODELBOX_FLOWUNIT_INFERENCE_CUDA_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_CUDA_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}.so CACHE INTERNAL "")

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_UNIT_SHARED})
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")

add_subdirectory(test_plugin)