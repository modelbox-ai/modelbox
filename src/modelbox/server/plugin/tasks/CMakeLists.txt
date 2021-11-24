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

set(UNIT_NAME "plugin")

project(modelbox-${UNIT_NAME})

file(GLOB_RECURSE MODELBOX_UNIT_SOURCE *.cpp *.cc *.c)
exclude_files_from_dir_in_list(MODELBOX_UNIT_SOURCE "${MODELBOX_UNIT_SOURCE}" "${CMAKE_BINARY_DIR}/")

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${NLOHMANN_INCLUDE_DIR})

set(MODELBOX_SERVER_PLUGIN modelbox-plugin)

add_library(${MODELBOX_SERVER_PLUGIN} SHARED ${MODELBOX_UNIT_SOURCE})

set_target_properties(${MODELBOX_SERVER_PLUGIN} PROPERTIES 
    OUTPUT_NAME "modelbox-plugin"
    PREFIX ""
    SUFFIX ".so")

install(TARGETS ${MODELBOX_SERVER_PLUGIN} 
    COMPONENT server
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    )

set(MODELBOX_PLUGIN_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/${MODELBOX_SERVER_PLUGIN}.so)

target_link_libraries(${MODELBOX_SERVER_PLUGIN} pthread)
target_link_libraries(${MODELBOX_SERVER_PLUGIN} rt)

set(MODELBOX_SERVER_PLUGIN ${MODELBOX_SERVER_PLUGIN} CACHE INTERNAL "")
set(MODELBOX_PLUGIN_SO_PATH ${MODELBOX_PLUGIN_SO_PATH} CACHE INTERNAL "")
