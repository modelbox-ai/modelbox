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

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "cannot build the project in the source directory! Out-of-source build is enforced!")
endif()
 
set(PLUGIN_NAME "example")

if(NOT DEFINED MODELBOX_PROJECT_VERSION_MAJOR)
    # build from flowunit cmakelists, not from project cmakelists
    set(MODELBOX_PROJECT_VERSION_MAJOR 0)
    set(MODELBOX_PROJECT_VERSION_MINOR 0)
    set(MODELBOX_PROJECT_VERSION_PATCH 1)
    set(RELEASE_PACKAGE_DIR_ROOT /opt/modelbox/project/default )
    set(RELEASE_PACKAGE_DIR_LIB ${RELEASE_PACKAGE_DIR_ROOT}/lib ) 
    set(RELEASE_PACKAGE_DIR_BIN ${RELEASE_PACKAGE_DIR_ROOT}/bin ) 
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fno-strict-aliasing")
 
file(GLOB_RECURSE UNIT_SOURCE *.cpp *.cc *.c)
 
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(UNIT_SHARED ${CMAKE_PROJECT_NAME}-${PLUGIN_NAME}-plugin)
add_library(${UNIT_SHARED} SHARED ${UNIT_SOURCE})
 
set_target_properties(
  ${UNIT_SHARED} PROPERTIES
  SOVERSION ${MODELBOX_PROJECT_VERSION_MAJOR}
  VERSION ${MODELBOX_PROJECT_VERSION_MAJOR}.${MODELBOX_PROJECT_VERSION_MINOR}.${MODELBOX_PROJECT_VERSION_PATCH}
)

target_link_libraries(${UNIT_SHARED} pthread)
target_link_libraries(${UNIT_SHARED} rt)
target_link_libraries(${UNIT_SHARED} dl)

install(TARGETS ${UNIT_SHARED} 
    COMPONENT ${UNIT_COMPONENT}
    RUNTIME DESTINATION ${RELEASE_PACKAGE_DIR_BIN}
    LIBRARY DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    ARCHIVE DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    OPTIONAL
    )