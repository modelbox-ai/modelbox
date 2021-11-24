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

file(GLOB_RECURSE SOURCES *.cpp *.cc *.c)

if (NOT Boost_FOUND) 
    message(STATUS "Not found boost, skip build IAMAuth")
    return()
endif()

if (NOT CPPREST_FOUND) 
    message(STATUS "Not found cpprest, skip build IAMAuth")
    return()
endif()

set(INCLUDE ${CMAKE_CURRENT_LIST_DIR})

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${APIGW_CPP_INCLUDE_DIR})
include_directories(${CPPREST_INCLUDE_DIR})
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${NLOHMANN_INCLUDE_DIR})

set(LIBRARY modelbox-common-cpu-iam_auth)  
add_library(${LIBRARY} SHARED ${SOURCES})
set_property(TARGET ${LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${LIBRARY} ${CPPREST_LIBRARIES} ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES})
target_link_libraries(${LIBRARY} ${APIGW_CPP_LIBRARIES})
target_link_libraries(${LIBRARY} ${LIBMODELBOX_SHARED})

install(FILES 
    $<TARGET_FILE:${APIGW_CPP_LIBRARIES}> DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    COMPONENT cpu-device-flowunit
    )

install(TARGETS ${LIBRARY} 
    COMPONENT cpu-device-flowunit
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )

set(MODELBOX_COMMON_IAM_AUTH_LIBRARY ${LIBRARY} CACHE INTERNAL "")
set(MODELBOX_COMMON_IAM_AUTH_INCLUDE ${INCLUDE} CACHE INTERNAL "")
list(APPEND DRIVER_UNIT_TEST_INCLUDE ${MODELBOX_COMMON_IAM_AUTH_INCLUDE})
set(DRIVER_UNIT_TEST_INCLUDE ${DRIVER_UNIT_TEST_INCLUDE} CACHE INTERNAL "")
list(APPEND DRIVER_UNIT_TEST_LINK_LIBRARIES ${MODELBOX_COMMON_IAM_AUTH_LIBRARY})
set(DRIVER_UNIT_TEST_LINK_LIBRARIES ${DRIVER_UNIT_TEST_LINK_LIBRARIES} CACHE INTERNAL "")
