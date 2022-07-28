#!/bin/bash
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
echo "VERSION_ID:$VERSION_ID"

download() {
    url="$1"
    softName=${url##*/}
    echo -e "\n\nBegin to download ${softName}"
    curl -k -L -O ${url}

    times=1
    until [ $times -gt 3 ]; do
        if [ $(ls -l ${softName}|awk '{print $5}') -lt 50000 ]; then
            ls -lh ${softName}
            echo "package ${softName} download failed, retry $times ......"
            curl -k -L -O ${url}
        fi
        let "times++"
    done

    if [ $(ls -l ${softName}|awk '{print $5}') -lt 50000 ]; then
        echo "package ${softName} download failed,pls check"
        exit 1
    fi
}

if [ "$(arch)" == "x86_64" ];then
    if [ "$VERSION_ID" == "18.04" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_dev-ubuntu.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3_dev-ubuntu.tar.gz
    elif [ "$VERSION_ID" == "20.04" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_dev.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3_dev.tar.gz
    fi
    download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/nlohmann-json_3.7.3.tar.gz
    download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/Video_Codec_SDK_9.1.23.tar.gz
elif [ "$(arch)" == "aarch64" ];then
    if [ "$VERSION_ID" == "18.04" ];then
        download http://download.modelbox-ai.com/third-party/aarch64/opencv_4.2.0_dev-ubuntu.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/obssdk_3.22.3_dev-ubuntu.tar.gz
    elif [ "$VERSION_ID" == "20.03" ];then
        download http://download.modelbox-ai.com/third-party/aarch64/cpprestsdk_2.10.18_dev.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/duktape_2.6.0_dev.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/ffmpeg_4.4_dev.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/opencv_4.2.0_dev.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/obssdk_3.22.3_dev.tar.gz
    fi
    download http://download.modelbox-ai.com/third-party/sdk/nlohmann-json_3.7.3.tar.gz
fi

ls -lh *.tar.gz
ls -lh release
