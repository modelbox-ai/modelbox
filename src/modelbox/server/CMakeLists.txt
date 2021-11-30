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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")

if(DUKTAPE_FOUND)
    add_definitions(-DENABLE_JS_PLUGIN)
endif()

file(GLOB_RECURSE MODELBOX_SOURCES *.cpp *.cc *.c)
exclude_files_from_dir_in_list(MODELBOX_SOURCES "${MODELBOX_SOURCES}" "${CMAKE_CURRENT_LIST_DIR}/plugin/")
exclude_files_from_dir_in_list(MODELBOX_SOURCES "${MODELBOX_SOURCES}" "${CMAKE_CURRENT_LIST_DIR}/iva_plugin/")
set(MODELBOX_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB_RECURSE SERVER_BIN_FILES ${CMAKE_CURRENT_LIST_DIR}/bin/*)

list(APPEND MODELBOX_SERVER_INCLUDES ${MODELBOX_INCLUDE})
list(APPEND MODELBOX_SERVER_INCLUDES ${MODELBOX_COMMON_INCLUDE})
list(APPEND MODELBOX_SERVER_INCLUDES ${LIB_HTTPLIB_INCLUDE_DIR})
list(APPEND SERVER_INCLUDES ${LIBMODELBOX_INCLUDE})
list(APPEND SERVER_INCLUDES ${LIBMODELBOX_BASE_INCLUDE})
list(APPEND SERVER_INCLUDES ${MODELBOX_INCLUDE})
list(APPEND SERVER_INCLUDES ${OPENSSL_INCLUDE_DIR})
list(APPEND SERVER_INCLUDES ${MODELBOX_SERVER_INCLUDES})
list(APPEND SERVER_INCLUDES ${MODELBOX_MANAGER_INCLUDE})
list(REMOVE_DUPLICATES SERVER_INCLUDES)

include_directories(${SERVER_INCLUDES})
include_directories(${TOML_INCLUDE_DIR})
include_directories(${DUKTAPE_INCLUDE_DIR})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${HTTPLIB_INCLUDE_DIR})

set(HEADER 
    ${MODELBOX_INCLUDE}/modelbox
    ${MODELBOX_COMMON_INCLUDE}/modelbox
    )

add_subdirectory(plugin)

add_executable(modelbox-server ${MODELBOX_SOURCES})
target_link_libraries(modelbox-server ${LIBMODELBOX_SHARED})
target_link_libraries(modelbox-server ${MODELBOX_COMMON_LIBRARY})
target_link_libraries(modelbox-server pthread)
target_link_libraries(modelbox-server rt)
target_link_libraries(modelbox-server ${DUKTAPE_LIBRARIES})
target_link_libraries(modelbox-server manager-client)
target_link_libraries(modelbox-server ${OPENSSL_LIBRARIES})

if(NOT CMAKE_INSTALL_RUNSTATEDIR)
    set(CMAKE_INSTALL_RUNSTATEDIR "var/run" CACHE INTERNAL "")
endif()

if (NOT MODELBOX_CONFIG_FILE_NAME)
    set(MODELBOX_CONFIG_FILE_NAME "modelbox.conf")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/modelbox-opts ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-opts @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/init.d/modelbox.in ${CMAKE_CURRENT_BINARY_DIR}/etc/init.d/modelbox @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/modelbox.conf.in ${CMAKE_CURRENT_BINARY_DIR}/etc/${MODELBOX_CONFIG_FILE_NAME} @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/modelbox-dev.conf.in ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-dev.conf.template @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/etc/modelbox.service.in ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox.service @ONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/misc/modelbox-server-cmd.json.in ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-server-cmd.json @ONLY)

add_dependencies(modelbox-server ${LIBMODELBOX_SHARED})

set_target_properties(modelbox-server PROPERTIES OUTPUT_NAME "modelbox")

install(TARGETS modelbox-server 
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
    ${CMAKE_CURRENT_BINARY_DIR}/etc/init.d/modelbox
    DESTINATION /etc/init.d
    PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE 
    COMPONENT server
    )


install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/etc/${MODELBOX_CONFIG_FILE_NAME}
    ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-dev.conf.template
    ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox-opts
    DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/modelbox/ 
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    COMPONENT server
    )

install(DIRECTORY 
    DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/modelbox/graph
    DIRECTORY_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    COMPONENT server
    )

install(DIRECTORY 
    DESTINATION ${CMAKE_INSTALL_FULL_SYSCONFDIR}/modelbox/conf.d
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

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-server-cmd.json
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT server
    )
install(PROGRAMS ${SERVER_BIN_FILES}
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT server)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/etc/modelbox.service 
    DESTINATION ${SYSTEMDSYSTEMUNITDIR} 
    COMPONENT server
    )

set(MODELBOX_SERVER_INCLUDE 
    ${MODELBOX_SERVER_INCLUDES} 
    CACHE INTERNAL "")
    
set(MODELBOX_SERVER_SOURCES ${MODELBOX_SOURCES} CACHE INTERNAL "")
set(MODELBOX_SERVER_LINK_LIBRARIES 
    ${HUAWEI_SECURE_C_LIBRARIES}
    ${BBOX_LIBRARIES}
    ${LINK_LIBRARIES}
    manager-client
    ${MODELBOX_COMMON_LIBRARY}
    CACHE INTERNAL "")
