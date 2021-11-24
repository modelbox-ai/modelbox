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

 
cmake_minimum_required(VERSION 3.3)
 
file(GLOB_RECURSE UNIT_TEST_SOURCE *.cpp *.cc *.c)
 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MOCK_DRIVER_CTRL_INCLUDE})
include_directories(${TEST_INCLUDE})
include_directories(${MODELBOX_COMMON_DRIVER_UTIL_INCLUDE})

list(REMOVE_DUPLICATES DRIVER_UNIT_TEST_INCLUDE)
include_directories(${DRIVER_UNIT_TEST_INCLUDE})

add_executable(drivers-unit EXCLUDE_FROM_ALL
	${UNIT_TEST_SOURCE}
	${TEST_MAIN_SOURCE}
	${DRIVER_UNIT_TEST_SOURCE}
)
set(DRIVER_UNIT_TEST_SOURCE "" CACHE INTERNAL "")

add_custom_target(all-drivers)
add_custom_command(TARGET all-drivers PRE_BUILD
	COMMAND rm -fr ${TEST_WORKING_DRIVERS_DIR}/*
)

foreach (ITR ${DRIVER_UNIT_TEST_TARGET} ${list_var})
	add_dependencies(all-drivers ${ITR})
	add_custom_command(TARGET all-drivers POST_BUILD
	COMMAND cp $<TARGET_FILE:${ITR}> ${TEST_WORKING_DRIVERS_DIR}/
	)
endforeach(ITR) 
set(DRIVER_UNIT_TEST_TARGET "" CACHE INTERNAL "")

add_custom_command(TARGET all-drivers POST_BUILD
	COMMAND ldconfig ${TEST_WORKING_DRIVERS_DIR} -n
)

foreach (ITR ${DRIVER_UNIT_TEST_LINK_LIBRARIES} ${list_var})
	target_link_libraries(drivers-unit ${ITR})
endforeach(ITR) 
set(DRIVER_UNIT_TEST_LINK_LIBRARIES "" CACHE INTERNAL "")

target_link_libraries(drivers-unit pthread)
target_link_libraries(drivers-unit rt)
target_link_libraries(drivers-unit dl)
target_link_libraries(drivers-unit gtest_main)
target_link_libraries(drivers-unit gmock_main)
target_link_libraries(drivers-unit ${LIBMODELBOX_SHARED})
target_link_libraries(drivers-unit ${MOCKFLOW_LIB})
target_link_libraries(drivers-unit pthread)
add_dependencies(drivers-unit all-drivers)

add_custom_target(unittest-drivers
	COMMAND ${TEST_RUNNER_LIST} ${CMAKE_CURRENT_BINARY_DIR}/drivers-unit
	DEPENDS  drivers-unit
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run drivers Test..."
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS drivers-unit)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS unittest-drivers)
set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")

set(MODELBOX_DRIVERS_TEST_TARGET drivers-unit CACHE INTERNAL "")

 