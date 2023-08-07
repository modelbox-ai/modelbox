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

project(modelbox-python)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if(NOT ${PYTHONLIBS_FOUND})
    message(STATUS "Disable python pip package")
    return()
endif()

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${PYBIND11_INCLUDE_DIR})
include_directories(${MODELBOX_COMMON_MODELBOX_API_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB_RECURSE LIBMODELBOX_PYHON_SOURCES *.cpp *.cc *.c)
set(EMPTY_SOURCE_FILE ${CMAKE_BINARY_DIR}/empty.cc)
if (NOT EXISTS ${EMPTY_SOURCE_FILE})
    file(WRITE ${EMPTY_SOURCE_FILE})
endif()
add_library(modelbox-python-static STATIC ${LIBMODELBOX_PYHON_SOURCES})
set_property(TARGET modelbox-python-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(modelbox-python-static PRIVATE pybind11::module)
target_link_libraries(modelbox-python-static PRIVATE ${MODELBOX_COMMON_MODELBOX_API_LIBRARY})
target_link_libraries(modelbox-python-static PRIVATE ${LIBMODELBOX_SHARED})
add_dependencies(modelbox-python-static ${MODELBOX_COMMON_MODELBOX_API_LIBRARY})

# so for test, with build RPATH
set(MODELBOX_PYTHON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)
add_library(modelbox-python-build SHARED ${EMPTY_SOURCE_FILE})
target_link_libraries(modelbox-python-build -Wl,--whole-archive modelbox-python-static -Wl,--no-whole-archive)
set_target_properties(modelbox-python-build PROPERTIES 
    OUTPUT_NAME "_modelbox"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

# so for release, without RPATH
set(MODELBOX_PYTHON_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
add_library(modelbox-python SHARED ${EMPTY_SOURCE_FILE})
target_link_libraries(modelbox-python -Wl,--whole-archive modelbox-python-static -Wl,--no-whole-archive)
set_target_properties(modelbox-python PROPERTIES 
    OUTPUT_NAME "_modelbox"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(modelbox-python PROPERTIES SKIP_BUILD_RPATH ON)

# copy build version to build directory
set_target_properties(modelbox-python-build
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_BINARY_DIR}/modelbox"
    LIBRARY_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_BINARY_DIR}/modelbox"
    RUNTIME_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_BINARY_DIR}/bin"
)

# copy release version to release directory
set_target_properties(modelbox-python
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_TARGET_DIR}/modelbox"
    LIBRARY_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_TARGET_DIR}/modelbox"
    RUNTIME_OUTPUT_DIRECTORY "${MODELBOX_PYTHON_TARGET_DIR}/bin"
)

set(MODELBOX_PYTHON_RELEASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/release)
configure_file(${CMAKE_CURRENT_LIST_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
set(PYTHON_CONFIG_IN ${CMAKE_CURRENT_LIST_DIR}/test/test_config.py.in)
set(PYTHON_CONFIG_OUT ${CMAKE_CURRENT_BINARY_DIR}/test/test_config.py)

add_custom_target(python-files)
# copy file to target
add_custom_command(TARGET python-files PRE_BUILD COMMAND rm -fr ${MODELBOX_PYTHON_TARGET_DIR}/*)
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${CMAKE_CURRENT_LIST_DIR}/modelbox ${MODELBOX_PYTHON_TARGET_DIR})
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${LICENSE_FILE} ${MODELBOX_PYTHON_TARGET_DIR}/ )
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${README_FILE} ${MODELBOX_PYTHON_TARGET_DIR}/ )
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${CMAKE_CURRENT_BINARY_DIR}/setup.py ${MODELBOX_PYTHON_TARGET_DIR}/ )
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${CMAKE_CURRENT_LIST_DIR}/MANIFEST.in ${MODELBOX_PYTHON_TARGET_DIR}/)

# copy file to build, for testing
add_custom_command(TARGET python-files PRE_BUILD COMMAND rm -fr ${MODELBOX_PYTHON_BINARY_DIR})
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${MODELBOX_PYTHON_TARGET_DIR} ${MODELBOX_PYTHON_BINARY_DIR})
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${CMAKE_CURRENT_LIST_DIR}/test ${MODELBOX_PYTHON_BINARY_DIR}/ )
add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af ${CMAKE_CURRENT_BINARY_DIR}/test/* ${MODELBOX_PYTHON_BINARY_DIR}/test/ )
add_custom_command(TARGET python-files PRE_BUILD COMMAND rm -fr ${CMAKE_CURRENT_BINARY_DIR}/test/*.in)
add_dependencies(modelbox-python-build python-files)
add_dependencies(modelbox-python modelbox-python-build)
add_custom_command(TARGET modelbox-python 
    WORKING_DIRECTORY ${MODELBOX_PYTHON_TARGET_DIR}
    POST_BUILD 
    COMMENT "Building python wheel package.."
    COMMAND strip -s ${MODELBOX_PYTHON_TARGET_DIR}/modelbox/*.so
    COMMAND umask 0022 && chmod -R a+rX . && ${PYTHON_EXECUTABLE} setup.py -q sdist -d ${MODELBOX_PYTHON_RELEASE_DIR} bdist_wheel -d ${MODELBOX_PYTHON_RELEASE_DIR})

install(CODE 
    "file(COPY ${MODELBOX_PYTHON_RELEASE_DIR}/ DESTINATION ${RELEASE_PACKAGE_DIR}/python)"
)

set(MODELBOX_PYTHON_BINARY_DIR ${MODELBOX_PYTHON_BINARY_DIR} CACHE INTERNAL "")

if(NOT DISABLE_MODELBOX_TEST)
    add_custom_target(unittest-python
        COMMAND PYTHONPATH=${MODELBOX_PYTHON_BINARY_DIR} ${PYTHON_EXECUTABLE} -m unittest discover ${MODELBOX_PYTHON_BINARY_DIR} -v
        DEPENDS modelbox-python
        WORKING_DIRECTORY ${TEST_WORKING_DIR}
        COMMENT "Run python Test..."
    )

    # run single test case
    # usage:
    #   cmake -DPYTHON_TEST_CASE=test_log.py ..
    #   make unittest-python-case
    if (PYTHON_TEST_CASE)
        message(STATUS "run test case for python: ${PYTHON_TEST_CASE}")
        add_custom_target(unittest-python-case
            COMMAND PYTHONPATH=${MODELBOX_PYTHON_BINARY_DIR} ${PYTHON_EXECUTABLE} ${MODELBOX_PYTHON_BINARY_DIR}/test/${PYTHON_TEST_CASE};
            DEPENDS modelbox-python
            WORKING_DIRECTORY ${TEST_WORKING_DIR}
            COMMENT "Run python Test..."
        )
        add_dependencies(unittest-python-case all-drivers)
    endif()

    add_dependencies(unittest-python all-drivers)

    list(APPEND MODELBOX_UNIT_TEST_TARGETS modelbox-python)
    set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

    list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS unittest-python)
    set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")
endif()

# update test config file
list(APPEND MODELBOX_UNIT_TEST_CONFIG_IN ${PYTHON_CONFIG_IN})
set(MODELBOX_UNIT_TEST_CONFIG_IN ${MODELBOX_UNIT_TEST_CONFIG_IN} CACHE INTERNAL "")
list(APPEND MODELBOX_UNIT_TEST_CONFIG_OUT ${PYTHON_CONFIG_OUT})
set(MODELBOX_UNIT_TEST_CONFIG_OUT ${MODELBOX_UNIT_TEST_CONFIG_OUT} CACHE INTERNAL "")
list(APPEND DRIVER_UNIT_TEST_TARGET modelbox-python-build)
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")

set(MODELBOX_PYTHON_OP_DIR ${CMAKE_SOURCE_DIR}/src/python/test/op/ CACHE INTERNAL "")

FILE(GLOB OP_CHILDRENS RELATIVE ${MODELBOX_PYTHON_OP_DIR} ${MODELBOX_PYTHON_OP_DIR}/*)
foreach(OP_CHILDREN ${OP_CHILDRENS})
    if(IS_DIRECTORY ${MODELBOX_PYTHON_OP_DIR}/${OP_CHILDREN})
        file(GLOB OP_CONFIG_FILES ${MODELBOX_PYTHON_OP_DIR}/${OP_CHILDREN}/*.in ${MODELBOX_PYTHON_OP_DIR}/${OP_CHILDREN}/*.py)
        foreach(OP_CONFIG_FILE ${OP_CONFIG_FILES})
            if(OP_CONFIG_FILE MATCHES ".*.in$")
                string(REGEX MATCH "op/*/.*.in$"  RELATIVE_PATH ${OP_CONFIG_FILE})
                string(REGEX REPLACE "op/.*/" "" TARGET_NAME ${RELATIVE_PATH})
                string(REGEX REPLACE ".in" "_test.toml" TARGET_NAME ${TARGET_NAME})
                configure_file(${OP_CONFIG_FILE} ${CMAKE_BINARY_DIR}/test/test-working-dir/data/python_op/${OP_CHILDREN}/${TARGET_NAME} @ONLY)
            elseif(OP_CONFIG_FILE MATCHES ".*.py")
                add_custom_command(TARGET python-files PRE_BUILD COMMAND cp -af  ${OP_CONFIG_FILE} ${CMAKE_BINARY_DIR}/test/test-working-dir/data/python_op/${OP_CHILDREN}/)
            endif()
        endforeach()
    endif()
endforeach()