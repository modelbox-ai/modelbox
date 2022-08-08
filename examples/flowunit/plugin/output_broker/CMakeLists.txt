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

set(PLUGIN_NAME "example")
set(UNIT_DEVICE "cpu")
set(UNIT_NAME "output_broker")

project(modelbox-flowunit-${UNIT_DEVICE}-${UNIT_NAME}-${PLUGIN_NAME})

if(NOT DEFINED MODELBOX_PROJECT_VERSION_MAJOR)
    # build from flowunit cmakelists, not from project cmakelists
    set(MODELBOX_PROJECT_VERSION_MAJOR 0)
    set(MODELBOX_PROJECT_VERSION_MINOR 0)
    set(MODELBOX_PROJECT_VERSION_PATCH 1)
    set(RELEASE_PACKAGE_DIR_ROOT /opt/modelbox/flowunit/${UNIT_SECT} )
    set(RELEASE_PACKAGE_DIR_LIB ${RELEASE_PACKAGE_DIR_ROOT} ) 
    set(RELEASE_PACKAGE_DIR_BIN ${RELEASE_PACKAGE_DIR_ROOT} ) 
endif()

file(GLOB_RECURSE MODELBOX_PLUGIN_SOURCE *.cpp *.cc *.c)

set(MODELBOX_PLUGIN_SHARED modelbox-unit-${UNIT_DEVICE}-${UNIT_NAME}-${PLUGIN_NAME})

add_library(${MODELBOX_PLUGIN_SHARED} SHARED ${MODELBOX_PLUGIN_SOURCE})

set_target_properties(${MODELBOX_PLUGIN_SHARED} PROPERTIES 
    SOVERSION ${MODELBOX_PROJECT_VERSION_MAJOR}
    VERSION ${MODELBOX_PROJECT_VERSION_MAJOR}.${MODELBOX_PROJECT_VERSION_MINOR}.${MODELBOX_PROJECT_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_PLUGIN_SHARED} pthread)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} rt)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} dl)
target_link_libraries(${MODELBOX_PLUGIN_SHARED} modelbox)

install(TARGETS ${MODELBOX_PLUGIN_SHARED} 
    COMPONENT ${UNIT_COMPONENT}
    RUNTIME DESTINATION ${RELEASE_PACKAGE_DIR_BIN}
    LIBRARY DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    ARCHIVE DESTINATION ${RELEASE_PACKAGE_DIR_LIB}
    OPTIONAL
    )

