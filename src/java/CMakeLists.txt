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

project(modelbox-java)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if (NOT WITH_JAVA) 
    message(STATUS "java support is disabled")
    return()
endif()

find_program(MVN mvn)
if (NOT MVN) 
    message(STATUS "not found maven, not build modelbox java")
    return()
endif()

if(NOT ${JNI_FOUND})
    message(STATUS "not found jdk, not build modelbox jar")
    return()
endif()

if(NOT ${JAVA_FOUND})
    message(STATUS "not found jdk, not build modelbox jar")
    return()
endif()

set(TEST_CONFIG_JSON_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/test/java/com/modelbox/TestConfig.json)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/test/java/com/modelbox/TestConfig.json.in ${TEST_CONFIG_JSON_FILE})
set(MODELBOX_JNI_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

add_subdirectory(jni)

add_custom_target(
    modelbox-java
    COMMAND mvn package -DskipTests -DbuildDirectory=${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


add_custom_target(
    unittest-java
    ${CMAKE_COMMAND} -E env "TEST_CONFIG_JSON_FILE=${TEST_CONFIG_JSON_FILE}"
    ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${JNI_LIBRARY_DIR}" mvn test -DbuildDirectory=${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS modelbox-java)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")
list(APPEND MODELBOX_UNIT_TEST_RUN_TARGETS unittest-java)
set(MODELBOX_UNIT_TEST_RUN_TARGETS ${MODELBOX_UNIT_TEST_RUN_TARGETS} CACHE INTERNAL "")

file(GLOB JAVA_RELEASE_FILE ${CMAKE_CURRENT_BINARY_DIR}/*.jar)
install(CODE 
    "file(COPY ${JAVA_RELEASE_FILE} DESTINATION ${RELEASE_PACKAGE_DIR}/java)"
)

add_dependencies(unittest-java modelbox-java)
add_dependencies(unittest-java modelbox-jni)