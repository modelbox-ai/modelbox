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

file(GLOB_RECURSE LIBMODELBOX_BASE_SOURCES *.cpp *.cc *.c)
exclude_files_from_dir_in_list(LIBMODELBOX_BASE_SOURCES "${LIBMODELBOX_BASE_SOURCES}" "${CMAKE_CURRENT_LIST_DIR}/arch/")

add_subdirectory(arch)

set(LIBMODELBOX_BASE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_CONFIG_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${TOML_INCLUDE_DIR})
set(LIBMODELBOX_BASE_INCLUDE ${LIBMODELBOX_BASE_INCLUDE})

add_library(LIBMODELBOX_BASE_OBJECTS OBJECT ${LIBMODELBOX_BASE_SOURCES})
set_property(TARGET LIBMODELBOX_BASE_OBJECTS PROPERTY POSITION_INDEPENDENT_CODE ON)

set(LIBMODELBOX_BASE_LIBRARY modelbox-base)
add_library(${LIBMODELBOX_BASE_LIBRARY} STATIC $<TARGET_OBJECTS:LIBMODELBOX_BASE_OBJECTS>)
set_property(TARGET ${LIBMODELBOX_BASE_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIBMODELBOX_BASE_LIBRARY} ${HUAWEI_SECURE_C_LIBRARIES})
target_link_libraries(${LIBMODELBOX_BASE_LIBRARY} pthread)
target_link_libraries(${LIBMODELBOX_BASE_LIBRARY} rt)
target_link_libraries(${LIBMODELBOX_BASE_LIBRARY} dl)

set_target_properties(${LIBMODELBOX_BASE_LIBRARY} PROPERTIES OUTPUT_NAME "modelbox-base")

set(LIBMODELBOX_BASE_LIBRARY ${LIBMODELBOX_BASE_LIBRARY} CACHE INTERNAL "")
set(LIBMODELBOX_BASE_INCLUDE ${LIBMODELBOX_BASE_INCLUDE} CACHE INTERNAL "" )
set(LIBMODELBOX_BASE_SOURCES ${LIBMODELBOX_BASE_SOURCES} CACHE INTERNAL "" )
set(LIBMODELBOX_BASE_OBJECTS LIBMODELBOX_BASE_OBJECTS CACHE INTERNAL "" )
