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

set(DEVICE_NAME "rockchip")
project(modelbox-devices-${DEVICE_NAME})

file(GLOB_RECURSE LIBMODELBOX_DEVICE_SOURCES *.cpp *.cc *.c)
set(LIBMODELBOX_DEVICE_ROCKCHIP_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)

include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${ROCKCHIP_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ROCKCHIP_INCLUDE})
include_directories(${ROCKCHIP_MPP_INCLUDE})
include_directories(${RKNN_INCLUDE_DIR})
include_directories(${ROCKCHIP_RGA_INCLUDE})

set(HEADER
    ${LIBMODELBOX_DEVICE_ROCKCHIP_INCLUDE}/modelbox
)

set(LIBMODELBOX_DEVICE_ROCKCHIP_STATIC libmodelbox-device-${DEVICE_NAME}-static)
set(LIBMODELBOX_DEVICE_ROCKCHIP_SHARED libmodelbox-device-${DEVICE_NAME}-shared)

add_library(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} STATIC ${LIBMODELBOX_DEVICE_SOURCES})
add_library(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} SHARED ${LIBMODELBOX_DEVICE_SOURCES})

set_target_properties(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} PROPERTIES
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} ${LIBMODELBOX_SHARED})
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} pthread)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} rt)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} dl)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} ${RKMPP_LIBRARIES})
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} ${RKRGA_LIBRARY})

target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} ${LIBMODELBOX_SHARED})
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} pthread)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} rt)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} dl)
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} ${RKMPP_LIBRARIES})
target_link_libraries(${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} ${RKRGA_LIBRARY})

set_target_properties(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} ${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED}
    PROPERTIES OUTPUT_NAME "modelbox-device-${DEVICE_NAME}"
)
set_target_properties(${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} ${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TEST_WORKING_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TEST_WORKING_BIN_DIR}"
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/libmodelbox-device-${DEVICE_NAME}.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-device-${DEVICE_NAME}.pc @ONLY)

install(TARGETS ${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED}
    COMPONENT rockchip-device-flowunit
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL)

install(TARGETS ${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC}
    COMPONENT rockchip-device-flowunit-devel
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL)

install(DIRECTORY
    ${HEADER}
    DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
    COMPONENT rockchip-device-flowunit-devel
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-device-${DEVICE_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT rockchip-device-flowunit-devel
)

set(LIBMODELBOX_DEVICE_ROCKCHIP_STATIC ${LIBMODELBOX_DEVICE_ROCKCHIP_STATIC} CACHE INTERNAL "")
set(LIBMODELBOX_DEVICE_ROCKCHIP_SHARED ${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_DEVICE_ROCKCHIP_INCLUDE ${LIBMODELBOX_DEVICE_ROCKCHIP_INCLUDE} CACHE INTERNAL "")

set(LIBMODELBOX_DEVICE_LIBMODELBOX_DEVICE_ROCKCHIP_SHARED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
set(LIBMODELBOX_DEVICE_SOURCES ${LIBMODELBOX_DEVICE_SOURCES} CACHE INTERNAL "")
set(LIBMODELBOX_DEVICE_ROCKCHIP_SO_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox-device-${DEVICE_NAME}.so CACHE INTERNAL "")

# driver test
list(APPEND DRIVER_UNIT_TEST_SOURCE ${MODELBOX_UNIT_TEST_SOURCE})
list(APPEND DRIVER_UNIT_TEST_TARGET ${LIBMODELBOX_DEVICE_ROCKCHIP_SHARED})

set(DRIVER_UNIT_TEST_SOURCE ${DRIVER_UNIT_TEST_SOURCE} CACHE INTERNAL "")
set(DRIVER_UNIT_TEST_TARGET ${DRIVER_UNIT_TEST_TARGET} CACHE INTERNAL "")