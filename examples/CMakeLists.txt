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

file(GLOB_RECURSE EXAMPLE_BIN_FILES ${CMAKE_CURRENT_LIST_DIR}/bin/*)

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${MODELBOX_SERVER_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CPU_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CUDA_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ASCEND_INCLUDE})

link_directories(${LIBMODELBOX_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CPU_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CUDA_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_ASCEND_BINARY_DIR})

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

set(MODELBOX_TEMPLATE_DIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/project-template")
set(MODELBOX_EXAMPLE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(PROJECT_BASE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/project-base")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/project-release")
file(COPY ${MODELBOX_EXAMPLE_DIR}/project/base/ DESTINATION ${PROJECT_BASE_BINARY_DIR})
configure_file(${MODELBOX_TOP_DIR}/CMake/FindACL.cmake   ${PROJECT_BASE_BINARY_DIR}/CMake/FindACL.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/FindDSMI.cmake   ${PROJECT_BASE_BINARY_DIR}/CMake/FindDSMI.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/Function.cmake   ${PROJECT_BASE_BINARY_DIR}/CMake/Function.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/CMake/Options.cmake   ${PROJECT_BASE_BINARY_DIR}/CMake/Options.cmake  @ONLY)
configure_file(${MODELBOX_TOP_DIR}/test/test_main.cc   ${PROJECT_BASE_BINARY_DIR}/test/test_main.cc  @ONLY)
file(COPY ${MODELBOX_TOP_DIR}/.gitignore DESTINATION ${PROJECT_BASE_BINARY_DIR})

file(COPY ${MODELBOX_EXAMPLE_DIR}/service-plugin DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${MODELBOX_EXAMPLE_DIR}/flowunit DESTINATION ${PROJECT_BINARY_DIR})

subdirlist(PROJECT_TEMPLATE ${MODELBOX_EXAMPLE_DIR}/project "")
list(REMOVE_ITEM PROJECT_TEMPLATE "base")

subdirlist(DEMO_LIST ${DEMO_SOURCE_DIR} "")

# Create template project
foreach (ITR ${PROJECT_TEMPLATE})
	set(PROJECT_TEMPLATE_DIR "${PROJECT_BINARY_DIR}/project/${ITR}")
	set(PROJECT_TEMPLATE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/project/${ITR}")
    set(PROJECT_DEMO_DIR "")
    file(COPY ${PROJECT_BASE_BINARY_DIR}/ DESTINATION ${PROJECT_TEMPLATE_DIR})

    # check if whether copy demo source
    foreach (ITR1 ${DEMO_LIST})
        if(${ITR} STREQUAL ${ITR1})
            file(COPY ${DEMO_SOURCE_DIR}/${ITR1}/ DESTINATION ${PROJECT_TEMPLATE_DIR}/src)
            set(PROJECT_DEMO_DIR "${DEMO_SOURCE_DIR}/${ITR1}")
        endif()
    endforeach(ITR1)
    file(COPY ${PROJECT_TEMPLATE_SRC_DIR}/ DESTINATION ${PROJECT_TEMPLATE_DIR})
    if(EXISTS ${PROJECT_TEMPLATE_SRC_DIR}/setup.sh)
        # run setup script in template directory
        execute_process(COMMAND sh ${PROJECT_TEMPLATE_SRC_DIR}/setup.sh "${PROJECT_TEMPLATE_DIR}" "${MODELBOX_EXAMPLE_DIR}" "${PROJECT_DEMO_DIR}"
            RESULT_VARIABLE COMMAND_RESULT
            WORKING_DIRECTORY ${PROJECT_TEMPLATE_DIR} 
        )

        if(COMMAND_RESULT)
            message(FATAL_ERROR "run setup script failed")
        endif()
        file(REMOVE ${PROJECT_TEMPLATE_DIR}/setup.sh)
    endif()
endforeach(ITR) 

install(DIRECTORY ${PROJECT_BINARY_DIR}/ DESTINATION ${MODELBOX_TEMPLATE_DIR}
    COMPONENT document
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/misc/modelbox-template-cmd.json.in ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-template-cmd.json @ONLY)
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/misc/modelbox-template-cmd.json
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT document
    )

install(PROGRAMS ${EXAMPLE_BIN_FILES}
    DESTINATION ${MODELBOX_TOOLS_PATH}
    COMPONENT document)

set(MODELBOX_TEMPLATE_BIN_DIR "${PROJECT_BINARY_DIR}" CACHE INTERNAL "")
set(MODELBOX_TEMPLATE_CMD_PATH "${CMAKE_CURRENT_LIST_DIR}/bin/template" CACHE INTERNAL "")
