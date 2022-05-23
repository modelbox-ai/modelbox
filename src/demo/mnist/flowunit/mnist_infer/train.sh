#!/bin/bash
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
# See the License for the specific language 

CURRDIR=$(pwd)

setup_var() {
    TRAIN_DIR="${CURRDIR}/mnist-train"
    OS_ARCH=$(uname -m)
    mkdir ${TRAIN_DIR} -p
    if [ $? -ne 0 ]; then
        echo "create train dir failed"
        return 1
    fi
    TRAIN_FILE="${CURRDIR}/train.py"
}

download_package() {
    pip list |grep "tensorflow " >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        $SUDO pip install tensorflow tensorflow-datasets
        if [ $? -ne 0 ]; then
            echo "install python train package failed"
            return 1
        fi
    fi

    return 0
}

train() {
    python ${TRAIN_FILE}
    if [ $? -ne 0 ]; then
            echo "train failed"
            return 1
    fi

    return 0
}

export_model() {
    cp ${TRAIN_DIR}/saved_model.pb ${CURRDIR}/mnist_model.pb
    return $?
}


main() {
    setup_var
    if [ $? -ne 0 ]; then
            return 1
    fi

    echo "working directory: ${TRAIN_DIR}"

    download_package
    if [ $? -ne 0 ]; then
            return 1
    fi

    train
    if [ $? -ne 0 ]; then
            return 1
    fi

    export_model
    if [ $? -ne 0 ]; then
            return 1
    fi

    rm -fr "${TRAIN_DIR}"

    echo "train success."
    echo "quick start: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple"
}

main
