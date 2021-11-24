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

project(modelbox)

find_package(Doxygen)
if(NOT DOXYGEN_FOUND)
    message(STATUS "Disable document building.")
    return()
endif()

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

if (NOT DOXYGEN_API_INSTALL_DIR)
    set(DOXYGEN_API_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/www")
endif()

string(REPLACE ";" " " DOXYGEN_LIBMODELBOX_INCLUDES "${LIBMODELBOX_INCLUDE}")
string(REPLACE ";" " " DOXYGEN_MODELBOX_SERVER_INCLUDES "${MODELBOX_SERVER_INCLUDE}")

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

add_custom_target(api-docs ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/ DESTINATION ${DOXYGEN_API_INSTALL_DIR}/api
    COMPONENT document
)
