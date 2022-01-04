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

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(GRAPH_NAME "emotion_detection.toml")
set(EMOTION_TEST_VIDEO "emotion_test_video.mp4")
set(SOLUTION_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${GRAPH_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${GRAPH_NAME}.in ${SOLUTION_GRAPH} @ONLY)
install(FILES 
    ${SOLUTION_GRAPH} DESTINATION ${SOLUTION_GRAPHS_SUB}
    COMPONENT solution
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/${EMOTION_TEST_VIDEO} DESTINATION ${SOLUTION_VIDEO_DIR}
    COMPONENT solution
)
