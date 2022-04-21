#!/bin/sh
# run this script during cmake prepare

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

TARGE_DIR=$1
EXAMPLE_DIR=$2
DEMO_DIR=$3

main() {
    for flowdir in ${TARGE_DIR}/src/flowunit/* ; do
        if [ ! -d "${flowdir}" ]; then
            continue;
        fi

        cp ${EXAMPLE_DIR}/flowunit/python/CMakeLists.txt ${flowdir}/CMakeLists.txt
        if [ $? -ne 0 ]; then
            echo "copy cmake to template failed."
            return 1
        fi
        
        sed -i "s/example/$(basename ${flowdir})/g" ${flowdir}/CMakeLists.txt
        if [ $? -ne 0 ]; then
            echo "change cmakefile name failed."
            return 1
        fi
    done

    cp ${EXAMPLE_DIR}/project/base/src/graph/CMakeLists.txt ${TARGE_DIR}/src/graph/CMakeLists.txt
    if [ $? -ne 0 ]; then
        echo "copy cmake to graph failed."
        return 1
    fi

    mv ${TARGE_DIR}/src/graph/mnist.toml.in ${TARGE_DIR}/src/graph/mnist.toml
    sed -i "s#@DEMO_MNIST_FLOWUNIT_DIR@#@APPLICATION_PATH@/flowunit#g" ${TARGE_DIR}/src/graph/mnist.toml
    if [ $? -ne 0 ]; then
        echo "change graph path failed."
        return 1
    fi
}

main
