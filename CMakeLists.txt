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

project(modelbox)

if (NOT MODELBOX_VERSION_MAJOR)
    set(MODELBOX_VERSION_MAJOR 1)
endif()

if (NOT MODELBOX_VERSION_MINOR)
    set(MODELBOX_VERSION_MINOR 0)
endif()

if (NOT MODELBOX_VERSION_PATCH)
    set(MODELBOX_VERSION_PATCH 0)
endif()

set(MODELBOX_VERSION_STRING "${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}")
set(MODELBOX_VERSION_API 1)

set(CMAKE_PROJECT_VERSION_MAJOR ${MODELBOX_VERSION_MAJOR})
set(CMAKE_PROJECT_VERSION_MINOR ${MODELBOX_VERSION_MINOR})
set(CMAKE_PROJECT_VERSION_PATCH ${MODELBOX_VERSION_PATCH})

set(MODELBOX_AUTHOR "Huawei Technologies Co., Ltd.")
set(MODELBOX_AUTHOR_EMAIL "")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

include(Options)
include(SecureCompilerOption)
include(Function)
include(FindPkgConfig)
include(CMakeDependentOption)
include(GNUInstallDirs)
include(CPackComponent)
include(ProjectEnvVars)
include(JavaJDK)

if (NOT STANDALONE)
    PKG_GET_VARIABLE(SYSTEMDSYSTEMUNITDIR systemd systemdsystemunitdir)
endif()

set(LICENSE_FILE ${CMAKE_CURRENT_LIST_DIR}/LICENSE)
set(README_FILE ${CMAKE_CURRENT_LIST_DIR}/README.md)
set(RELEASE_PACKAGE_DIR ${CMAKE_BINARY_DIR}/release)
set(CUSTOM_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs)
set(MODELBOX_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CPPREST)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS system)
find_package(CUDA 10.0)
find_package(CUDACUDA)
find_package(TENSORRT)
find_package(TENSORFLOW)
find_package(OBS)
find_package(DIS)
find_package(VCN)
find_package(NVCUVID)
find_package(FFMPEG)
find_package(ACL)
find_package(DSMI)
find_package(OpenCV)
find_package(DUKTAPE)
find_package(MINDSPORE)
find_package(MINDSPORE_LITE)
find_package(FUSE)
find_package(JNI)
find_package(Java)

add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(test EXCLUDE_FROM_ALL)
add_subdirectory(docs)
add_subdirectory(package)
