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


if(UNIX AND NOT APPLE)
    file(GLOB LIBMODELBOX_LINUX_OS_SOURCES "./linux/*.cc" "*.cc")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        file(GLOB LIBMODELBOX_CPU_SOURCES "./linux/x86_64/*.cc" "*.cc")
        set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/linux/x86_64/base64_simd.cc APPEND PROPERTY COMPILE_FLAGS "-mavx512f -mavx512bw -fPIC")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        file(GLOB LIBMODELBOX_CPU_SOURCES "./linux/aarch64/*.cc" "*.cc")
    else()
         file(GLOB LIBMODELBOX_CPU_SOURCES "./linux/default/*.cc" "*.cc")
    endif()
elseif(ANDROID)
    file(GLOB LIBMODELBOX_LINUX_OS_SOURCES "./android/*.cc" "*.cc")
endif()

list(APPEND LIBMODELBOX_BASE_SOURCES ${LIBMODELBOX_LINUX_OS_SOURCES})
set(LIBMODELBOX_BASE_SOURCES ${LIBMODELBOX_BASE_SOURCES} PARENT_SCOPE)

set(LIBMODELBOX_ARCH_CPU_LIBRARY modelbox-arch-cpu)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
add_library(LIBMODELBOX_ARCH_CPU_OBJECTS OBJECT ${LIBMODELBOX_CPU_SOURCES})
set_property(TARGET ${LIBMODELBOX_ARCH_CPU_OBJECTS} PROPERTY POSITION_INDEPENDENT_CODE ON)
set(LIBMODELBOX_ARCH_CPU_OBJECTS ${LIBMODELBOX_ARCH_CPU_OBJECTS} CACHE INTERNAL "")