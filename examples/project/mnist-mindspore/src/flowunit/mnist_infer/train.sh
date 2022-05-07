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
VERSION="1.7.0"
TRAIN_VER="r1.6"

get_package_archname() {
  case $(uname -m) in
   x86_64)
     echo "linux-x64"
     ;;
   aarch64)
     echo "linux-aarch64"
     ;;
   armv7l | armv8l)
     echo "linux-aarch32"
     ;;
   *)
     echo ""
     ;;
  esac
}

setup_var() {
    TRAIN_DIR="${CURRDIR}/mnist-train"
    OS_ARCH=$(uname -m)
    mkdir ${TRAIN_DIR} -p
    if [ $? -ne 0 ]; then
        echo "create train dir failed"
        return 1
    fi
    TRAIN_FILE="${TRAIN_DIR}/train.py"
    MINDSPORE_LITE_NAME="mindspore-lite-${VERSION}-$(get_package_archname)"
    MINDSPORE_LITE_DIR="${TRAIN_DIR}/${MINDSPORE_LITE_NAME}"
    MINDSPORE_LITE_FILE="${MINDSPORE_LITE_NAME}.tar.gz"
    MINDSPORE_TRAIN_DOWNLOAD_URL="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/${TRAIN_VER}/tutorials/zh_cn/mindspore_quick_start.py"
    MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION}/MindSpore/lite/release/linux/${OS_ARCH}/${MINDSPORE_LITE_FILE}"

    pip list |grep mindspore >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        PYVER=$(python --version | awk '{print $2}' | awk -F. '{print $1$2}')
        PYVER_STR="cp${PYVER}-cp${PYVER}"
        MINDSPORE_WHEEL_FILE="mindspore-${VERSION}-${PYVER_STR}-linux_${OS_ARCH}.whl"
        MINDSPORE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION}/MindSpore/cpu/${OS_ARCH}/${MINDSPORE_WHEEL_FILE}"

        curl -Is ${MINDSPORE_DOWNLOAD_URL} | head -1 | grep "200 OK" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            PYVER_STR="cp${PYVER}-cp${PYVER}m"
            MINDSPORE_WHEEL_FILE="mindspore-${VERSION}-${PYVER_STR}-linux_${OS_ARCH}.whl"
            MINDSPORE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION}/MindSpore/cpu/${OS_ARCH}/${MINDSPORE_WHEEL_FILE}"
            curl -Is ${MINDSPORE_DOWNLOAD_URL} | head -1 | grep "200 OK" >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                echo "cannot find URL for mindspore, please download manually";
                return 1
            fi
        fi
    fi

    rm ${TRAIN_DIR}/checkpoint_lenet* -f
    cd ${TRAIN_DIR}
}

download_package() {
    if [ ! -f "${TRAIN_FILE}" ]; then
        wget ${MINDSPORE_TRAIN_DOWNLOAD_URL} -O ${TRAIN_FILE}
        if [ $? -ne 0 ]; then
            echo "download train script failed"
            return 1
        fi
    fi

    if [ ! -f "${MINDSPORE_LITE_FILE}" ]; then
        wget ${MINDSPORE_LITE_DOWNLOAD_URL}
        if [ $? -ne 0 ]; then
            echo "download lite failed"
            return 1
        fi
    fi

    if [ ! -d ${MINDSPORE_LITE_DIR} ]; then
        tar xf ${MINDSPORE_LITE_FILE}
        if [ $? -ne 0 ]; then
            echo "extract failed"
            return 1
        fi
    fi

    pip list |grep mindspore >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ ! -f "${MINDSPORE_WHEEL_FILE}" ]; then
            wget ${MINDSPORE_DOWNLOAD_URL}
            if [ $? -ne 0 ]; then
                echo "download mindspore train package failed"
                return 1
            fi
        fi
        SUDO=""
        if [ "$(id -u)" != 0 ]; then
            SUDO="sudo"
        fi

        $SUDO pip install ${MINDSPORE_WHEEL_FILE} --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
        if [ $? -ne 0 ]; then
            echo "install python train package failed"
            return 1
        fi
    fi

    return 0
}

setup_export_code() {
        echo "
from mindspore import export, load_checkpoint, load_param_into_net
import glob

ckpt = glob.glob('*.ckpt')

net = LeNet5()
param_dict = load_checkpoint(ckpt[0])
load_param_into_net(net, param_dict)
input = np.random.uniform(0.0, 1.0, size=[1, 1, 32, 32]).astype(np.float32)
export(net, Tensor(input), file_name='mnist', file_format='MINDIR')
" >> ${TRAIN_FILE}
}

train() {
    setup_export_code
    if [ $? -ne 0 ]; then
        echo "setup export python script failed."
        return 1
    fi

    python ${TRAIN_FILE}
    if [ $? -ne 0 ]; then
            echo "train failed"
            return 1
    fi

    return 0
}

export_lite_model() {
    export LD_LIBRARY_PATH=${MINDSPORE_LITE_DIR}/tools/converter/lib
    ${MINDSPORE_LITE_DIR}/tools/converter/converter/converter_lite --fmk=MINDIR --modelFile=${TRAIN_DIR}/mnist.mindir --outputFile=${TRAIN_DIR}/mnist
    if [ $? -ne 0 ]; then
            echo "export lite model failed"
            return 1
    fi

    cp ${TRAIN_DIR}/mnist.ms ${CURRDIR}/

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

    export_lite_model
    if [ $? -ne 0 ]; then
            return 1
    fi

    echo "train success."
    echo "model dir: ${TRAIN_DIR}"
    echo "quick start: https://www.mindspore.cn/tutorial/en/r0.5/quick_start/quick_start.html"
}

main
