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

file(GLOB FUNCTION_TEST_SOURCE *.cpp *.cc *.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MOCK_DRIVER_CTRL_INCLUDE})
include_directories(${TEST_INCLUDE})
include_directories(${DRIVER_UNIT_TEST_INCLUDE})

set(FUNCTION_TEST_SOURCE ${FUNCTION_TEST_SOURCE} CACHE INTERNAL "")

add_subdirectory(car_detection)

add_executable(function EXCLUDE_FROM_ALL
	${FUNCTION_TEST_SOURCE}
	${TEST_MAIN_SOURCE}
)
set(MODELBOX_UNIT_LINK_LIBRARY ${OpenCV_LIBS})
add_dependencies(function ${MODELBOX_DRIVERS_TEST_TARGET})
target_link_libraries(function pthread)
target_link_libraries(function rt)
target_link_libraries(function gtest_main)
target_link_libraries(function gmock_main)
target_link_libraries(function ${LIBMODELBOX_SHARED})
target_link_libraries(function ${MOCK_DRIVER_CTRL_LIB})
target_link_libraries(function ${HUAWEI_SECURE_C_LIBRARIES})
target_link_libraries(function ${MODELBOX_UNIT_LINK_LIBRARY})

add_custom_target(functiontest
	COMMAND ${TEST_RUNNER_LIST} ${CMAKE_CURRENT_BINARY_DIR}/function
	DEPENDS function
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run Function Test..."
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS function)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS functiontest)
set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")

foreach (ITR ${DRIVER_SOLUTION_TEST_TARGET} ${list_var})
	add_dependencies(function ${ITR})
	add_custom_command(TARGET function PRE_BUILD
	COMMAND cp $<TARGET_FILE:${ITR}> ${TEST_SOLUTION_DRIVERS_DIR}/
	)
endforeach(ITR) 
set(DRIVER_SOLUTION_TEST_TARGET "" CACHE INTERNAL "")