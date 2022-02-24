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


if(NOT TEST_WORKING_DIR)
	set(TEST_WORKING_DIR ${CMAKE_BINARY_DIR}/test/test-working-dir)
	file(MAKE_DIRECTORY ${TEST_WORKING_DIR})
endif()
set(TEST_WORKING_DATA_DIR "${TEST_WORKING_DIR}/data")
file(MAKE_DIRECTORY ${TEST_WORKING_DATA_DIR})
set(TEST_WORKING_LIB_DIR "${TEST_WORKING_DIR}/lib")
file(MAKE_DIRECTORY ${TEST_WORKING_LIB_DIR})
set(TEST_WORKING_BIN_DIR "${TEST_WORKING_DIR}/bin")
file(MAKE_DIRECTORY ${TEST_WORKING_BIN_DIR})
set(TEST_WORKING_DRIVERS_DIR "${TEST_WORKING_DIR}/drivers")
file(MAKE_DIRECTORY ${TEST_WORKING_DRIVERS_DIR})
set(TEST_ASSETS ${CMAKE_SOURCE_DIR}/test/assets)
set(TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/test)
set(TEST_SOLUTION_DRIVERS_DIR "${TEST_WORKING_DIR}/solution")
file(MAKE_DIRECTORY ${TEST_SOLUTION_DRIVERS_DIR})

set(MODELBOX_TOOLS_PATH "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/tools")
set(MODELBOX_SOLUTION_DIR "/opt/modelbox/solution")
set(MODELBOX_WWW_DIR "${CMAKE_INSTALL_FULL_DATAROOTDIR}/modelbox/www")
