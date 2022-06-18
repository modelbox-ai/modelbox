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

project(data_source_parser_plugin)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

if(OBS_FOUND)
    add_subdirectory(obs_source_parser)
else()
    message(STATUS "Not found obs library, disable obs source parser plugin")
endif()

if(VCN_FOUND)
    add_subdirectory(vcn_source_parser)
else()
    message(STATUS "Not found vcn library, disable vcn source parser plugin")
endif()

if(CPPREST_FOUND)
    add_subdirectory(url_source_parser)
else()
    message(STATUS "Not found cpprest library, disable url output broker plugin")
endif()

if(CPPREST_FOUND)
    add_subdirectory(vis_source_parser)
else()
    message(STATUS "Not found cpprest library, disable vis output broker plugin")
endif()

if(CPPREST_FOUND)
    add_subdirectory(restful_source_parser)
else()
    message(STATUS "Not found cpprest library, disable restful output broker plugin")
endif()

add_subdirectory(vcn_restful_source_parser)