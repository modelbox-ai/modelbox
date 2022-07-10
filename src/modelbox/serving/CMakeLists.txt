#
# Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

include_directories(${MODELBOX_INCLUDE})
include_directories(${HUAWEI_SECURE_C_INCLUDE_DIR})
include_directories(${LIBMODELBOX_BASE_INCLUDE})

add_executable(modelbox-serving ${MODELBOX_SOURCES})
target_link_libraries(modelbox-serving pthread)
target_link_libraries(modelbox-serving rt)
target_link_libraries(modelbox-serving ${LIBMODELBOX_SHARED})
target_link_libraries(modelbox-serving ${MODELBOX_COMMON_LIBRARY})
target_link_libraries(modelbox-serving ${HUAWEI_SECURE_C_LIBRARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_file/model.toml.in ${TEST_WORKING_DATA_DIR}/test_serving_model.toml @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_file/custom_service.py.in ${TEST_WORKING_DATA_DIR}/test_custom_service.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_file/test_client.py.in ${TEST_WORKING_DATA_DIR}/test_client.py @ONLY)

if (STANDALONE)
    set_target_properties(modelbox-serving PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

install(TARGETS modelbox-serving 
    COMPONENT server
    RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
    )

install(DIRECTORY 
    DESTINATION ${CMAKE_INSTALL_RUNSTATEDIR}/modelbox/
    COMPONENT server
)

set(MODELBOX_SERVING_SOURCES ${MODELBOX_SOURCES} CACHE INTERNAL "")