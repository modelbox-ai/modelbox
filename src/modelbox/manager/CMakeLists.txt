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

file(GLOB_RECURSE MODELBOX_SOURCES *.cpp *.cc *.c)
set(MODELBOX_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB_RECURSE MANAGER_BIN_FILES ${CMAKE_CURRENT_LIST_DIR}/bin/*)
set(MODELBOX_MANAGER_CLIENT_SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/manager_monitor_client.c")

exclude_files_from_dir_in_list(MODELBOX_SOURCES "${MODELBOX_SOURCES}" "${MODELBOX_MANAGER_CLIENT_SOURCE}")

list(APPEND MODELBOX_MANAGER_INCLUDES ${MODELBOX_INCLUDE})
list(APPEND MANAGER_INCLUDES ${MODELBOX_INCLUDE})
list(APPEND MANAGER_INCLUDES ${MODELBOX_MANAGER_INCLUDES})
list(APPEND MANAGER_INCLUDES ${CMAKE_CURRENT_LIST_DIR}/src)
list(APPEND MANAGER_INCLUDES ${TLOG_INCLUDE})
list(REMOVE_DUPLICATES MANAGER_INCLUDES)

include_directories(${MANAGER_INCLUDES})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})

set(HEADER 
    ${MODELBOX_INCLUDE}/modelbox
    ${MODELBOX_COMMON_INCLUDE}/modelbox
    )

add_executable(modelbox-manager ${MODELBOX_SOURCES})
target_link_libraries(modelbox-manager pthread)
target_link_libraries(modelbox-manager rt)
target_link_libraries(modelbox-manager ${TLOG_STATIC_LIBRARIES})
target_link_libraries(modelbox-manager ${HUAWEI_SECURE_C_LIBRARIES})

add_library(manager-client SHARED ${MODELBOX_MANAGER_CLIENT_SOURCE})

if (NOT MODELBOX_MANAGER_CONFIG_FILE_NAME)
    set(MODELBOX_MANAGER_CONFIG_FILE_NAME "manager.conf")
endif()

if (STANDALONE)
    set_target_properties(modelbox-manager PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/manager-opts ${CMAKE_CURRENT_BINARY_DIR}/etc/manager-opts @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/init.d/modelbox-manager.in ${CMAKE_CURRENT_BINARY_DIR}/etc/init.d/modelbox-manager @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/manager.conf.in ${CMAKE_CURRENT_BINARY_DIR}/etc/${MODELBOX_MANAGER_CONFIG_FILE_NAME} @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/modelbox-manager.service.in ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-manager.service @ONLY)

install(TARGETS modelbox-manager 
    COMPONENT server
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    )

install(TARGETS manager-client
    COMPONENT server
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    )

install(DIRECTORY 
    ${HEADER} 
    DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
    COMPONENT server-devel
    )

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/etc/init.d/modelbox-manager
    DESTINATION /etc/init.d
    PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE 
    COMPONENT server
    )

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/etc/${MODELBOX_MANAGER_CONFIG_FILE_NAME}
    ${CMAKE_CURRENT_BINARY_DIR}/etc/manager-opts
    DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/modelbox/ 
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    COMPONENT server
    )

install(DIRECTORY 
    DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/modelbox/init-script/modelbox-manager
    DIRECTORY_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    COMPONENT server
    )

install(DIRECTORY 
    DESTINATION /${CMAKE_INSTALL_RUNSTATEDIR}/modelbox/
    COMPONENT server
)

install(DIRECTORY 
    DESTINATION /var/log/modelbox
    COMPONENT server
)


install(PROGRAMS ${MANAGER_BIN_FILES}
    DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    COMPONENT server)

if (SYSTEMDSYSTEMUNITDIR) 
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-manager.service 
    DESTINATION ${SYSTEMDSYSTEMUNITDIR} 
    COMPONENT server
    )
else()
message("Skip install systemd unit")
endif()

set(MODELBOX_MANAGER_INCLUDE 
    ${MANAGER_INCLUDES} 
    CACHE INTERNAL "")
    
set(MODELBOX_MANAGER_SOURCES ${MODELBOX_SOURCES} ${MODELBOX_MANAGER_CLIENT_SOURCE} CACHE INTERNAL "")
set(MODELBOX_MANAGER_LINK_LIBRARIES 
    ${HUAWEI_SECURE_C_LIBRARIES}
    CACHE INTERNAL "")