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

file(GLOB MANAGER_TEST_SOURCE *.cpp *.cc *.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MOCK_DRIVER_CTRL_INCLUDE})
include_directories(${TEST_INCLUDE})
include_directories(${DRIVER_UNIT_TEST_INCLUDE})

set(MANAGER_TEST_SOURCE ${MANAGER_TEST_SOURCE} CACHE INTERNAL "")

add_executable(manager-unit EXCLUDE_FROM_ALL
	${MANAGER_TEST_SOURCE}
	${MODELBOX_MANAGER_SOURCES}
)

target_link_libraries(manager-unit pthread)
target_link_libraries(manager-unit rt)
target_link_libraries(manager-unit gtest_main)
target_link_libraries(manager-unit gmock_main)
target_link_libraries(manager-unit ${HUAWEI_SECURE_C_LIBRARIES})
target_link_libraries(manager-unit ${TLOG_STATIC_LIBRARIES})

add_custom_target(unittest-manager
	COMMAND ${TEST_RUNNER_LIST} ${CMAKE_CURRENT_BINARY_DIR}/manager-unit
	DEPENDS manager-unit
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run manager-unit Test..."
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS manager)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS unittest-manager)
set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")
