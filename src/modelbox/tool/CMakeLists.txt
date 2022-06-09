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

file(GLOB_RECURSE MODELBOX_TOOL_SOURCES *.cpp *.cc *.c)
set(MODELBOX_TOOL_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)

list(APPEND MODELBOX_TOOL_INCLUDES ${MODELBOX_TOOL_INCLUDE})
list(APPEND MODELBOX_TOOL_INCLUDES ${LIBMODELBOX_INCLUDE})
list(APPEND MODELBOX_TOOL_INCLUDES ${LIBMODELBOX_BASE_INCLUDE})
list(APPEND MODELBOX_TOOL_INCLUDES ${TLOG_INCLUDE})
list(APPEND MODELBOX_TOOL_INCLUDES ${NLOHMANN_INCLUDE_DIR})
list(APPEND MODELBOX_TOOL_INCLUDES ${MODELBOX_COMMON_INCLUDE})
list(APPEND MODELBOX_TOOL_INCLUDES ${HUAWEI_SECURE_C_INCLUDE_DIR})
list(REMOVE_DUPLICATES MODELBOX_TOOL_INCLUDES)

include_directories(${MODELBOX_TOOL_INCLUDES})

set(HEADER 
    ${MODELBOX_TOOL_INCLUDE}/modelbox
    )

add_executable(modelbox-tool ${MODELBOX_TOOL_SOURCES})
target_link_libraries(modelbox-tool ${TLOG_STATIC_LIBRARIES})
target_link_libraries(modelbox-tool ${LIBMODELBOX_SHARED})
target_link_libraries(modelbox-tool ${MODELBOX_COMMON_LIBRARY})
target_link_libraries(modelbox-tool pthread)
target_link_libraries(modelbox-tool rt)
target_link_libraries(modelbox-tool ${HUAWEI_SECURE_C_LIBRARIES})

add_dependencies(modelbox-tool ${LIBMODELBOX_SHARED})

set(PYTHON_DEBUG_BIN ${CMAKE_CURRENT_LIST_DIR}/bin/modelbox-python-debug)
set_target_properties(modelbox-tool PROPERTIES OUTPUT_NAME "modelbox-tool")

if (STANDALONE)
    set_target_properties(modelbox-tool PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

install(TARGETS modelbox-tool
    COMPONENT libmodelbox
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    )

set(MODELBOX_TOOL_INCLUDE 
    ${MODELBOX_TOOL_INCLUDE} 
    ${TLOG_INCLUDE}
    CACHE INTERNAL "")
    
install(PROGRAMS ${PYTHON_DEBUG_BIN}
    DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    COMPONENT libmodelbox)

set(MODELBOX_TOOL_SOURCES ${MODELBOX_TOOL_SOURCES} CACHE INTERNAL "")
set(MODELBOX_TOOL_LINK_LIBRARIES 
    ${TLOG_STATIC_LIBRARIES}
    ${HUAWEI_SECURE_C_LIBRARIES}
    CACHE INTERNAL "")