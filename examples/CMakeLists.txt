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


subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")
file(GLOB_RECURSE EXAMPLE_BIN_FILES ${CMAKE_CURRENT_LIST_DIR}/bin/*)

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${MODELBOX_SERVER_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CPU_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CUDA_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ASCEND_INCLUDE})

link_directories(${LIBMODELBOX_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CPU_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CUDA_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_ASCEND_BINARY_DIR})

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

set(MODELBOX_TEMPLATE_DIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/project-template")
configure_file(${CMAKE_CURRENT_LIST_DIR}/misc/modelbox-template-cmd.json.in ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-template-cmd.json @ONLY)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flowunit DESTINATION ${MODELBOX_TEMPLATE_DIR}
    COMPONENT document
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/plugin DESTINATION ${MODELBOX_TEMPLATE_DIR}
    COMPONENT document
)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/project DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${MODELBOX_TOP_DIR}/CMake/FindACL.cmake   ${CMAKE_CURRENT_BINARY_DIR}/project/CMake/FindACL.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/FindDSMI.cmake   ${CMAKE_CURRENT_BINARY_DIR}/project/CMake/FindDSMI.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/Function.cmake   ${CMAKE_CURRENT_BINARY_DIR}/project/CMake/Function.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/Options.cmake   ${CMAKE_CURRENT_BINARY_DIR}/project/CMake/Options.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/test/test_main.cc   ${CMAKE_CURRENT_BINARY_DIR}/project/test/test_main.cc  @ONLY)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/project DESTINATION ${MODELBOX_TEMPLATE_DIR}
    COMPONENT document
)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-template-cmd.json
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT document
    )

install(PROGRAMS ${EXAMPLE_BIN_FILES}
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT document)

