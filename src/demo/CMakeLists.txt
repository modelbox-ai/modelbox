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

if (WITH_ALL_DEMO)
    subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")
else()
    list(APPEND SUBDIRS "hello_world")
    list(APPEND SUBDIRS "mnist")
endif()

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CPU_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_CUDA_INCLUDE})
include_directories(${LIBMODELBOX_DEVICE_ASCEND_INCLUDE})

link_directories(${LIBMODELBOX_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CPU_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_CUDA_BINARY_DIR})
link_directories(${LIBMODELBOX_DEVICE_ASCEND_BINARY_DIR})

set(DEMO_MODEL_DIR "${MODELBOX_DEMO_DIR}/model")
set(DEMO_VIDEO_DIR "${MODELBOX_DEMO_DIR}/video")
set(DEMO_IMAGE_DIR "${MODELBOX_DEMO_DIR}/image")

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

add_custom_target(demo
    DEPENDS ${DRIVER_DEMO_TEST_TARGET}
) 
