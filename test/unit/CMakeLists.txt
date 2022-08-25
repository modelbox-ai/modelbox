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

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

file(GLOB_RECURSE UNIT_TEST_SOURCE *.cpp *.cc *.c)
 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
 
include_directories(${TEST_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_MOCKDEVICE_INCLUDE})
include_directories(${LIBMODELBOX_FLOWUNIT_MOCKFLOWUNIT_INCLUDE})
include_directories(${LIBMODELBOX_GRAPHCONF_MOCKGRAPHCONF_INCLUDE})
include_directories(${LIBMODELBOX_VIRTUALDRIVER_PYTHON_INCLUDE})
include_directories(${MOCKFLOW_INCLUDE})
 
add_executable(unit EXCLUDE_FROM_ALL
    ${UNIT_TEST_SOURCE}
    ${TEST_SOURCE}
    ${TEST_MAIN_SOURCE}
)

set_target_properties(unit PROPERTIES ENABLE_EXPORTS 1)


if (TARGET ${MODELBOX_SERVER_PLUGIN_EDITOR})
add_dependencies(unit ${MODELBOX_SERVER_PLUGIN_EDITOR})
endif()

if (TARGET ${MODELBOX_SERVER_PLUGIN})
    add_dependencies(unit ${MODELBOX_SERVER_PLUGIN})
    add_custom_command(TARGET unit POST_BUILD
        COMMAND rm -fr ${TEST_WORKING_LIB_DIR}/*
        COMMAND cp $<TARGET_FILE:${MODELBOX_SERVER_PLUGIN}> ${TEST_WORKING_LIB_DIR}/
    )
endif()

if (${LIBMODELBOX_VIRTUALDRIVER_PYTHON_SHARED})
add_dependencies(unit ${LIBMODELBOX_VIRTUALDRIVER_PYTHON_SHARED})
endif()

add_dependencies(unit ${LIBMODELBOX_VIRTUALDRIVER_INFERENCE_SHARED})
add_dependencies(unit ${LIBMODELBOX_DEVICE_CPU_SHARED})

if(${PYTHONLIBS_FOUND})
    add_dependencies(unit ${LIBMODELBOX_FLOWUNIT_PYTHON_SHARED})
endif()

target_link_libraries(unit pthread)
target_link_libraries(unit rt)
target_link_libraries(unit dl)
target_link_libraries(unit gtest_main)
target_link_libraries(unit gmock_main)
target_link_libraries(unit ${MOCKFLOW_LIB})
target_link_libraries(unit ${TEST_LINK_LIBRARIES})
target_link_libraries(unit ${LIBMODELBOX_SHARED})
target_link_libraries(unit ${DUKTAPE_LIBRARIES})

add_custom_target(unittest-modelbox
	COMMAND ${TEST_RUNNER_LIST} ${CMAKE_CURRENT_BINARY_DIR}/unit
	DEPENDS  unit
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run modelbox-unit Test..."
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS unit)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS unittest-modelbox)
set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")

 
