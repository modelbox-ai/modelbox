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

project(modelbox-inference-tensorflow)

if (NOT TENSORFLOW_FOUND) 
    message(STATUS "Not found tensorflow, disable common flowunit")
    return()
endif()

file(GLOB_RECURSE MODELBOX_UNIT_SOURCE *.cpp *.cc *.c)
file(GLOB PLUGIN_FILES ./test_plugin/*.cc ./test_plugin/*.cpp ./test_plugin/*.c)
list(REMOVE_ITEM MODELBOX_UNIT_SOURCE ${PLUGIN_FILES})
set(HEADER tensorflow_inference_plugin.h)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_INCLUDE})
include_directories(${TENSORFLOW_INCLUDE_DIR})
include_directories(${MODELBOX_COMMON_INFERENCE_INCLUDE})

set(MODELBOX_UNIT_SHARED libmodelbox-engine-tensorflow)
set(MODELBOX_UNIT_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_UNIT_SHARED} SHARED ${MODELBOX_UNIT_SOURCE})
set(LIBMODELBOX_FLOWUNIT_INFERENCE_TENSORFLOW_SHARED ${MODELBOX_UNIT_SHARED})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES 
SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_UNIT_SHARED} pthread)
target_link_libraries(${MODELBOX_UNIT_SHARED} rt)
target_link_libraries(${MODELBOX_UNIT_SHARED} dl)
target_link_libraries(${MODELBOX_UNIT_SHARED} ${TENSORFLOW_LIBRARIES})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
target_link_libraries(${MODELBOX_UNIT_SHARED} ${MODELBOX_COMMON_INFERENCE_LIBRARY})

set_target_properties(${MODELBOX_UNIT_SHARED} PROPERTIES OUTPUT_NAME "modelbox-engine-tensorflow")

install(FILES ${HEADER} 
        COMPONENT cpu-device-flowunit-devel
        DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/modelbox)


install(TARGETS ${MODELBOX_UNIT_SHARED} 
        COMPONENT cpu-device-flowunit
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        OPTIONAL
        )

set(LIBMODELBOX_FLOWUNIT_INFERENCE_TENSORFLOW_SHARED ${MODELBOX_UNIT_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_TENSORFLOW_INCLUDE ${MODELBOX_UNIT_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_TENSORFLOW_SOURCES ${MODELBOX_UNIT_SOURCE} CACHE INTERNAL "")
set(LIBMODELBOX_FLOWUNIT_INFERENCE_TENSORFLOW_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-engine-tensorflow.so CACHE INTERNAL "")

add_subdirectory(test_plugin)
