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

file(GLOB_RECURSE LIBMODELBOX_SOURCES *.cpp *.cc *.c)
exclude_files_from_dir_in_list(LIBMODELBOX_SOURCES "${LIBMODELBOX_SOURCES}" "${CMAKE_CURRENT_LIST_DIR}/base/")
set(LIBMODELBOX_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)
set(LIBMODELBOX_ENGINE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/engine)
set(LIBMODELBOX_CONFIG_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/base/include)

include_directories(${LIBMODELBOX_ENGINE_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${NLOHMANN_INCLUDE_DIR})
include_directories(${LIBMODELBOX_CONFIG_INCLUDE})

add_subdirectory(base)
include_directories(${LIBMODELBOX_BASE_INCLUDE})

set(HEADER 
    ${LIBMODELBOX_INCLUDE}/modelbox
    ${LIBMODELBOX_BASE_INCLUDE}/modelbox
    ${LIBMODELBOX_CONFIG_INCLUDE}/modelbox
    )

add_library(libmodelbox-static STATIC ${LIBMODELBOX_SOURCES} $<TARGET_OBJECTS:LIBMODELBOX_BASE_OBJECTS> $<TARGET_OBJECTS:LIBMODELBOX_ARCH_CPU_OBJECTS>)
set(LIBMODELBOX_STATIC libmodelbox-static)

add_library(libmodelbox-shared SHARED ${LIBMODELBOX_SOURCES} $<TARGET_OBJECTS:LIBMODELBOX_BASE_OBJECTS> $<TARGET_OBJECTS:LIBMODELBOX_ARCH_CPU_OBJECTS>)
set(LIBMODELBOX_SHARED libmodelbox-shared)

set_target_properties(libmodelbox-shared PROPERTIES 
    SOVERSION ${MODELBOX_VERSION_MAJOR}
    VERSION ${MODELBOX_VERSION_MAJOR}.${MODELBOX_VERSION_MINOR}.${MODELBOX_VERSION_PATCH}
)

target_link_libraries(libmodelbox-static ssl)
target_link_libraries(libmodelbox-static crypto)
target_link_libraries(libmodelbox-static pthread)
target_link_libraries(libmodelbox-static rt)
target_link_libraries(libmodelbox-static dl)
target_link_libraries(libmodelbox-static atomic)
target_link_libraries(libmodelbox-static ${HUAWEI_SECURE_C_LIBRARIES})

target_link_libraries(libmodelbox-shared ssl)
target_link_libraries(libmodelbox-shared crypto)
target_link_libraries(libmodelbox-shared pthread)
target_link_libraries(libmodelbox-shared rt)
target_link_libraries(libmodelbox-shared dl)
target_link_libraries(libmodelbox-shared atomic)
target_link_libraries(libmodelbox-shared ${HUAWEI_SECURE_C_LIBRARIES})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ATOMIC_LINK_LIBRARIES, "atomic")
    target_link_libraries(libmodelbox-static ${ATOMIC_LINK_LIBRARIES})
    target_link_libraries(libmodelbox-shared ${ATOMIC_LINK_LIBRARIES})
endif()

set_target_properties(libmodelbox-static PROPERTIES OUTPUT_NAME "modelbox")
set_target_properties(libmodelbox-shared PROPERTIES OUTPUT_NAME "modelbox")

configure_file(${CMAKE_CURRENT_LIST_DIR}/libmodelbox.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox.pc @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${LIBMODELBOX_CONFIG_INCLUDE}/modelbox/base/config.h @ONLY)

install(TARGETS libmodelbox-shared 
    COMPONENT libmodelbox
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )

install(TARGETS libmodelbox-static 
    COMPONENT libmodelbox-devel
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    OPTIONAL
    )

install(DIRECTORY 
    ${HEADER} DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
    COMPONENT libmodelbox-devel
    )

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/libmodelbox.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT libmodelbox-devel
    )

set(LIBMODELBOX_STATIC ${LIBMODELBOX_STATIC} CACHE INTERNAL "")
set(LIBMODELBOX_SHARED ${LIBMODELBOX_SHARED} CACHE INTERNAL "")
set(LIBMODELBOX_INCLUDE 
    ${LIBMODELBOX_INCLUDE} 
    ${LIBMODELBOX_CONFIG_INCLUDE}
    CACHE INTERNAL "")
set(LIBMODELBOX_SOURCES ${LIBMODELBOX_SOURCES} CACHE INTERNAL "")
set(LIBMODELBOX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
set(LIBMODELBOX_SOURCES_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")
