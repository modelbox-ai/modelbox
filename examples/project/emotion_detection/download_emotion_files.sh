#!/bin/sh
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

DOWNLOAD_EMOTION_FILES="https://gitee.com/modelbox/modelbox-binary/attach_files/1010735/download/emotion_demo_files.zip"
BASE_PATH=$(cd `dirname $0`; pwd)

main() {
    wget ${DOWNLOAD_EMOTION_FILES} -O ${BASE_PATH}/emotion_demo_files.zip
    if [ $? -ne 0 ]; then
        echo "download emotion_demo_files.zip failed"
        return 1
    fi

    unzip ${BASE_PATH}/emotion_demo_files.zip -d ${BASE_PATH}/emotion_demo_files/
    if [ $? -ne 0 ]; then
        echo "decompress emotion_demo_files.zip failed"
        return 1
    fi

    cp ${BASE_PATH}/emotion_demo_files/emotion.pt ${BASE_PATH}/src/flowunit/emotion_infer/emotion.pt -f
    if [ $? -ne 0 ]; then
        echo "copy emotion model failed"
        return 1
    fi
    
    cp ${BASE_PATH}/emotion_demo_files/face_detector.pt ${BASE_PATH}/src/flowunit/face_detect/face_detector.pt -f
    if [ $? -ne 0 ]; then
        echo "copy face detector model failed"
        return 1
    fi

    cp ${BASE_PATH}/emotion_demo_files/emotion_test_video.mp4 ${BASE_PATH}/src/graph/emotion_test_video.mp4 -f
    if [ $? -ne 0 ]; then
        echo "copy test video failed"
        return 1
    fi

    rm ${BASE_PATH}/emotion_demo_files -rf
    if [ $? -ne 0 ]; then
        echo "remove emotion_demo_files folder failed"
        return 1
    fi
}

main
