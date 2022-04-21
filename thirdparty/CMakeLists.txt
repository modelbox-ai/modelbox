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

include(ExternalProject)

set(THIRDPARTY_DOWNLOAD_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
set(THIRDPARTY_DOWNLOAD_WORKING_DIR ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/build)

# 预先下载的三方组件，当使用ADD_SUBDIRECTORY包含子项目时，采用此方式。
if (NOT LOCAL_PACKAGE_PATH) 
  set(THIRDPARTY_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
  configure_file(CMake/pre-download.in ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/CMakeLists.txt @ONLY)
else()
  set(THIRDPARTY_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
  configure_file(CMake/local-package.in ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/CMakeLists.txt @ONLY)
endif()

file(MAKE_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ..
  RESULT_VARIABLE COMMAND_RESULT
  WORKING_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR} 
)

if(COMMAND_RESULT)
  message(FATAL_ERROR "Download thirdparty failed: ${COMMAND_RESULT}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE COMMAND_RESULT
  WORKING_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR} 
)

if(COMMAND_RESULT)
  message(FATAL_ERROR "Download thirdparty failed: ${COMMAND_RESULT}")
endif()

set(CMAKE_CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(GOOGLETEST_SOURCE_DIR ${THIRDPARTY_DOWNLOAD_DIR}/googletest)
add_subdirectory(${GOOGLETEST_SOURCE_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/googletest EXCLUDE_FROM_ALL)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_OLD})

set(PYTHON_VER 3.5)
find_package(PythonInterp ${PYTHON_VER})
if(NOT ${PYTHONINTERP_FOUND})
  set(PYTHON_EXECUTABLE /usr/bin/python3)
  find_package(PythonInterp ${PYTHON_VER})
endif()

find_package(PythonLibs ${PYTHON_VER})
set(PYTHONLIBS_FOUND ${PYTHONLIBS_FOUND} CACHE INTERNAL "")

if (NOT ${PYTHONE_DISABLED} AND ${PYTHONLIBS_FOUND})
  set(PYBIND_SOURCE_DIR ${THIRDPARTY_DOWNLOAD_DIR}/pybind11)
  add_subdirectory(${PYBIND_SOURCE_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/pybind11 EXCLUDE_FROM_ALL)
  set(PYBIND11_PYTHON_VERSION ${PYBIND11_PYTHON_VERSION} CACHE INTERNAL "")
  set(PYBIND11_INCLUDE_DIRS ${PYBIND_SOURCE_DIR}/include CACHE INTERNAL "")
  set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE INTERNAL "")
  set(PYTHON_VERSION_STRING ${PYTHON_VERSION_STRING} CACHE INTERNAL "")
else()
  message(STATUS "Disable python")
  set(PYTHONLIBS_FOUND FALSE CACHE INTERNAL "")
endif()

# 安全C的CMake生成
set(HUAWEI_SECURE_C_DIR ${THIRDPARTY_DOWNLOAD_DIR}/Huawei_Secure_C)
configure_file(CMake/Huawei_Secure_C_CMakeList.in ${HUAWEI_SECURE_C_DIR}/CMakeLists.txt @ONLY)
add_subdirectory(${HUAWEI_SECURE_C_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/Huawei_Secure_C EXCLUDE_FROM_ALL)

if (WITH_SECURE_C)
  install(FILES 
      $<TARGET_FILE:${HUAWEI_SECURE_C_LIBRARIES}> DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
      COMPONENT libmodelbox
      )
endif()

set(TLOG_DIR ${THIRDPARTY_DOWNLOAD_DIR}/tinylog)
configure_file(CMake/tlog_cmakelist.in ${TLOG_DIR}/CMakeLists.txt @ONLY)
add_subdirectory(${TLOG_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/tinylog EXCLUDE_FROM_ALL)

set(TOML_INCLUDE_DIR ${THIRDPARTY_DOWNLOAD_DIR}/toml11 CACHE INTERNAL "")

set(NLOHMANN_INCLUDE_DIR ${THIRDPARTY_DOWNLOAD_DIR}/nlohmann/include CACHE INTERNAL "")

set(CPP_HTTPLIB_DIR ${THIRDPARTY_DOWNLOAD_DIR}/cpp-httplib/build)
configure_file(CMake/cpp_httplib_cmakelist.in ${CPP_HTTPLIB_DIR}/CMakeLists.txt @ONLY)
add_subdirectory(${CPP_HTTPLIB_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/cpp_httplib  EXCLUDE_FROM_ALL)

if (${WITH_ALL_DEMO})
  set(DEMO_DIR ${THIRDPARTY_DOWNLOAD_DIR}/demo)
  configure_file(CMake/Demo_Files_CMakeList.in ${DEMO_DIR}/CMakeLists.txt @ONLY)
  add_subdirectory(${DEMO_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/demo EXCLUDE_FROM_ALL)
endif()

# APIGW_CPP的签名库
set(APIGW_CPP_DIR ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/APIGW-cpp-sdk-1.0.2)
configure_file(CMake/APIGW_CPP_CMakeList.in ${APIGW_CPP_DIR}/CMakeLists.txt @ONLY)
add_subdirectory(${APIGW_CPP_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/APIGW-cpp-sdk-1.0.2 EXCLUDE_FROM_ALL)

if (WITH_WEBUI)
  set(MODELBOX_WEBUI ${THIRDPARTY_DOWNLOAD_DIR}/modelbox-webui)
  if (IS_DIRECTORY ${MODELBOX_WEBUI}) 
    add_subdirectory(${MODELBOX_WEBUI} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/modelbox-webui)
  else()
    message(STATUS "Skip build modelbox editor webui")
  endif()
endif()
