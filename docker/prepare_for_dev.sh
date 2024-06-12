#!/bin/bash
CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
OS_NAME=$(sed -nr '/NAME/s/^NAME="(.*)"$/\1/gp' /etc/os-release)
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
PLATFROM=$(arch)
echo "OS_NAME:$OS_NAME"
echo "PLATFROM:$PLATFROM"
echo "VERSION_ID:$VERSION_ID"

download() {
    url="$1"
    softName=${url##*/}
    echo -e "\n\nBegin to download ${softName}"

    times=0
    while true
    do
        curl -k -L -O ${url}
        if [ $(ls -l ${softName}|awk '{print $5}') -ge 50000 ]; then
            echo "${softName} download complete"
            break
        else
            times=$[${times}+1]
            if [ ${times} -gt 3 ]; then
                echo "package ${softName} download failed,pls check"
                exit 1
            fi
            echo "package ${softName} download failed, retry $times in 3 seconds......"
            sleep 3
        fi
    done
}

if [ "${PLATFROM}" == "x86_64" ];then
    if [ "$OS_NAME" == "Ubuntu" ];then
        if [ "$VERSION_ID" == "20.04" ];then
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/glog_0.6.0_dev_ubuntu.tar.gz
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.5.5_dev_ubuntu.tar.gz
        elif [ "$VERSION_ID" == "22.04" ];then
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_dev_ubuntu.tar.gz
        fi
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3_dev_ubuntu.tar.gz
    elif [ "$OS_NAME" == "openEuler" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3_dev.tar.gz
    elif [ "$OS_NAME" == "android" ];then
        download https://dl.google.com/android/repository/android-ndk-r25b-linux.zip
        download https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_android/src.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_android/deb_files.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_android/cpprestsdk.tar.gz
        download https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/lite/release/linux/x86_64/mindspore-lite-1.9.0-linux-x64.tar.gz
        download https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/lite/release/android/gpu/mindspore-lite-1.9.0-android-aarch64.tar.gz
        ls -lh .
    fi
    download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/nlohmann-json_3.7.3.tar.gz
    download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/Video_Codec_SDK_9.1.23.tar.gz
elif [ "${PLATFROM}" == "aarch64" ];then
    if [ "$OS_NAME" == "Ubuntu" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/glog_0.6.0_dev_ubuntu.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/opencv_4.5.5_dev_ubuntu.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/obssdk_3.22.3_dev_ubuntu.tar.gz
    elif [ "$OS_NAME" == "openEuler" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/cpprestsdk_2.10.18_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/duktape_2.6.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/ffmpeg_4.4_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/opencv_4.2.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive_aarch64/obssdk_3.22.3_dev.tar.gz
    fi
    download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/nlohmann-json_3.7.3.tar.gz
else
    echo "build error"
    exit 1
fi

ls -lh *.tar.gz
ls -lh release
