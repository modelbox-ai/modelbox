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

add_definitions(-DBUILD_TEST)
set(TEST_MAIN_SOURCE ${CMAKE_CURRENT_LIST_DIR}/test_main.cc)
configure_file(${CMAKE_CURRENT_LIST_DIR}/test_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test_config.h @ONLY)

list(APPEND TEST_INCLUDE ${LIBMODELBOX_INCLUDE})
list(APPEND TEST_INCLUDE ${LIBMODELBOX_SOURCES_DIR})
list(APPEND TEST_INCLUDE ${MODELBOX_PROG_INCLUDE} )
list(APPEND TEST_INCLUDE ${MODELBOX_SERVER_INCLUDE})
list(APPEND TEST_INCLUDE ${LIBMODELBOX_BASE_INCLUDE})
list(APPEND TEST_INCLUDE ${TOML_INCLUDE_DIR})
list(APPEND TEST_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND TEST_INCLUDE ${MODELBOX_TOP_DIR})
list(APPEND TEST_INCLUDE ${MODELBOX_MANAGER_INCLUDE})
list(REMOVE_DUPLICATES TEST_INCLUDE)

if(DUKTAPE_FOUND)
    add_definitions(-DENABLE_JS_PLUGIN)
endif()

set(TEST_SOURCE 
    ${MODELBOX_SERVER_SOURCES} 
	${MODELBOX_TOOL_SOURCES}
    ${MODELBOX_SERVING_SOURCES}
)

set(TEST_LINK_LIBRARIES
    ${MODELBOX_SERVER_LINK_LIBRARIES}
	${MODELBOX_TOOL_LINK_LIBRARIES}
	${LIBMODELBOX_LINK_SOURCES} 
	${MODELBOX_MANAGER_LINK_LIBRARIES}
)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${NLOHMANN_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-gnu-unique")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
add_subdirectory(mock)
if(NOT DISABLE_MODELBOX_TEST)
	add_subdirectory(unit)
	add_subdirectory(drivers)
	add_subdirectory(function)
	add_subdirectory(manager)
endif()
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_OLD})

if(NOT DISABLE_MODELBOX_TEST)
list(REMOVE_DUPLICATES MODELBOX_UNIT_TEST_TARGETS)
add_custom_target(build-test
	DEPENDS ${MODELBOX_UNIT_TEST_TARGETS}
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run Build Test..."
)

if (TEST_COVERAGE)
add_custom_target(coverage
	COMMAND make unittest
	COMMAND lcov -d ${CMAKE_BINARY_DIR}/src -d ${CMAKE_BINARY_DIR}/test -c -o ${CMAKE_BINARY_DIR}/coverage.info
	COMMAND lcov -r ${CMAKE_BINARY_DIR}/coverage.info  '/usr/*' 'thirdparty/download/*' 'build/*' -o ${CMAKE_BINARY_DIR}/coverage-strip.info
	COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage-strip.info -o ${CMAKE_BINARY_DIR}/coverage/
	COMMAND rm ${CMAKE_BINARY_DIR}/coverage-strip.info
	DEPENDS build-test
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage.info ${CMAKE_BINARY_DIR}/coverage
	COMMENT "Run Coverate Test..."
)
endif()

list(REMOVE_DUPLICATES MODELBOX_UNIT_TEST_RUN_TARGETS)
add_custom_target(unittest
	DEPENDS ${MODELBOX_UNIT_TEST_RUN_TARGETS}
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run Unit Test..."
)
endif()

# replace all test config files
list(LENGTH MODELBOX_UNIT_TEST_CONFIG_IN CONFIG_FILE_NUMBER)
if ( ${CONFIG_FILE_NUMBER} GREATER 0)
	math(EXPR CONFIG_FILE_NUMBER "${CONFIG_FILE_NUMBER}-1")
	foreach(i RANGE 0 ${CONFIG_FILE_NUMBER}-1 1)
		list(GET MODELBOX_UNIT_TEST_CONFIG_IN ${i} CONFIG_IN)
		list(GET MODELBOX_UNIT_TEST_CONFIG_OUT ${i} CONFIG_OUT)
		configure_file(${CONFIG_IN} ${CONFIG_OUT} @ONLY)
	endforeach(i)
endif()

set(MODELBOX_UNIT_TEST_CONFIG_IN "" CACHE INTERNAL "")
set(MODELBOX_UNIT_TEST_CONFIG_OUT "" CACHE INTERNAL "")
set(MODELBOX_UNIT_TEST_TARGETS "" CACHE INTERNAL "")
set(MODELBOX_UNIT_TEST_RUN_TARGETS "" CACHE INTERNAL "")
