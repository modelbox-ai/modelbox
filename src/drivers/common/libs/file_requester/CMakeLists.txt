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


if (NOT CPPREST_FOUND) 
    message(STATUS "Not found cpprest, disable file_server library")
    return()
endif()

if (NOT OBS_FOUND)
message(STATUS "Not found obs, disable file_requester library")
    return()
endif()

file(GLOB_RECURSE UNIT_SOURCE *.cpp *.cc *.c)
group_source_test_files(MODELBOX_COMMON_LIB_SOURCE MODELBOX_UNIT_TEST_SOURCE "_test.c*" ${UNIT_SOURCE})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DRIVER_COMMON_LIB_INCLUDE})
include_directories(${MODELBOX_COMMON_OBS_CLIENT_INCLUDE})
include_directories(${CPPREST_INCLUDE_DIR})
include_directories(${INCLUDE})

set(MODELBOX_COMMON_LIB_SHARED modelbox-drivers-common-fileserver)
set(MODELBOX_COMMON_LIB_SOURCE_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_library(${MODELBOX_COMMON_LIB_SHARED} SHARED ${MODELBOX_COMMON_LIB_SOURCE})

set(LIBMODELBOX_DRIVER_COMMON_LIB_FILE_SERVER ${MODELBOX_COMMON_LIB_SHARED})
set_target_properties(${MODELBOX_COMMON_LIB_SHARED} PROPERTIES 
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} ${LIBMODELBOX_DEVICE_CPU_SHARED})
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} ${MODELBOX_COMMON_OBS_CLIENT_LIBRARY})
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} ${OBS_LIBRARIES})
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} pthread)
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} rt)
target_link_libraries(${MODELBOX_COMMON_LIB_SHARED} ${CPPREST_LIBRARIES})

install(TARGETS ${MODELBOX_COMMON_LIB_SHARED} 
    COMPONENT cpu-device-flowunit
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )


set(LIBMODELBOX_DRIVER_COMMON_LIB_FILE_SERVER ${MODELBOX_COMMON_LIB_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_DRIVER_COMMON_LIB_FILE_SERVER_INCLUDE ${MODELBOX_COMMON_LIB_SOURCE_INCLUDE} CACHE INTERNAL "")
set(LIBMODELBOX_DRIVER_COMMON_LIB_FILE_SERVER_SOURCES ${MODELBOX_COMMON_LIB_SOURCE} CACHE INTERNAL "")

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${MODELBOX_COMMON_LIB_SHARED})
list(APPEND DRIVER_UNIT_TEST_LINK_LIBRARIES ${MODELBOX_COMMON_LIB_SHARED})
set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_LINK_LIBRARIES ${DRIVER_UNIT_TEST_LINK_LIBRARIES} CACHE INTERNAL "")

