#!/bin/bash
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
echo "VERSION_ID:$VERSION_ID"

download() {
    url="$1"
    softName=${url##*/}
    echo -e "\n\n\Begen to download ${softName}"
    curl -LO ${url}

    if [ "$(ls -lh ${softName}|awk '{print $5}')" == "0" ]; then
        echo "package download failed"
        exit 1
    fi
}

if [ "$(arch)" == "x86_64" ];then
    if [ "$VERSION_ID" == "18.04" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3-ubuntu.tar.gz
    elif [ "$VERSION_ID" == "20.04" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3.tar.gz
    fi
elif [ "$(arch)" == "aarch64" ];then
    if [ "$VERSION_ID" == "18.04" ];then
        download http://download.modelbox-ai.com/third-party/aarch64/obssdk_3.22.3-ubuntu.tar.gz
    elif [ "$VERSION_ID" == "20.03" ];then
        download http://download.modelbox-ai.com/third-party/aarch64/cpprestsdk_2.10.15.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/duktape_2.6.0.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/ffmpeg_4.4.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/opencv_4.2.0.tar.gz
        download http://download.modelbox-ai.com/third-party/aarch64/obssdk_3.22.3.tar.gz
    fi
else
    echo "build error"
    exit 1
fi

ls -lh *.tar.gz

ls release|egrep 'devel|document|solution|demo'|xargs -i rm -f release/{}
ls -lh release
