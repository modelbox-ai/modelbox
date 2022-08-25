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

if(NOT ${JNI_FOUND})
    message(STATUS "not found jdk, not build modelbox jni")
    return()
endif()

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${JNI_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${MODELBOX_JNI_HEADER_DIR})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})

set(MODELBOX_JNI "modelbox-jni")
file(GLOB_RECURSE LIBMODELBOX_JAVA_JNI_SOURCES *.cpp *.cc *.c)
add_library(${MODELBOX_JNI} SHARED ${LIBMODELBOX_JAVA_JNI_SOURCES})
target_compile_options(${MODELBOX_JNI} PUBLIC -fvisibility=hidden)
add_dependencies(${MODELBOX_JNI} modelbox-java-jni-header)

target_link_libraries(${MODELBOX_JNI} ${LIBMODELBOX_SHARED})
set(JNI_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
set(JNI_PACKAGE_INSTALLDIR "/usr/java/packages/lib")

install(TARGETS ${MODELBOX_JNI} 
    COMPONENT cpu-device-flowunit
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${JNI_PACKAGE_INSTALLDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )

