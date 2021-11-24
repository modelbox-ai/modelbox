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

set(PLUGIN_NAME "vcn_source")

project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME}-${PLUGIN_NAME})

file(GLOB_RECURSE PLUGIN_SOURCE *.cpp *.cc *.c)
group_source_test_files(MODELBOX_PLUGIN_SOURCE MODELBOX_PLUGIN_TEST_SOURCE "_test.c*" ${PLUGIN_SOURCE})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CPU_INCLUDE})
include_directories(${MODELBOX_COMMON_INCLUDE})
include_directories(${NLOHMANN_INCLUDE_DIR})
include_directories(${VCN_INCLUDE_DIR})
include_directories(${MODELBOX_COMMON_DRIVER_UTIL_INCLUDE})

set(MODELBOX_PLUGIN_SHARED modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-${PLUGIN_NAME}-shared)
set(MODELBOX_PLUGIN_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_PLUGIN_SHARED} SHARED ${MODELBOX_PLUGIN_SOURCE})
 
set_target_properties(${MODELBOX_PLUGIN_SHARED} PROPERTIES 
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_PLUGIN_SHARED} ${LIBMODELBOX_DEVICE_CPU_SHARED})
target_link_libraries(${MODELBOX_PLUGIN_SHARED} ${MODELBOX_COMMON_LIBRARY})
target_link_libraries(${MODELBOX_PLUGIN_SHARED} pthread)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} rt)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} dl)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} ${VCN_LIBRARIES})
target_link_libraries(${MODELBOX_PLUGIN_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_PLUGIN_SHARED} ${MODELBOX_COMMON_SOURCE_CONTEXT_LIBRARY})
set_target_properties(${MODELBOX_PLUGIN_SHARED} PROPERTIES OUTPUT_NAME "modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-${PLUGIN_NAME}")

install(TARGETS ${MODELBOX_PLUGIN_SHARED} 
    COMPONENT cpu-device-flowunit
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )


install(DIRECTORY ${HEADER} 
    DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR} 
    COMPONENT cpu-device-flowunit-devel
    )

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_PLUGIN_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_PLUGIN_SHARED})
list(APPEND DRIVER_UNIT_TEST_LINK_LIBRARIES ${MODELBOX_PLUGIN_SHARED})
list(APPEND DRIVER_UNIT_TEST_INCLUDE ${MODELBOX_PLUGIN_SOURCE_INCLUDE})
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_LINK_LIBRARIES ${DRIVER_UNIT_TEST_LINK_LIBRARIES} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_INCLUDE ${DRIVER_UNIT_TEST_INCLUDE} CACHE INTERNAL "")

