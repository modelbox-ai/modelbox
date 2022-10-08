#!/bin/bash
CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
OS_NAME=$(sed -nr '/NAME/s/^NAME="(.*)"$/\1/gp' /etc/os-release)
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
echo "OS_NAME:$OS_NAME"
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
    if [ "$OS_NAME" == "Ubuntu" ];then
        if [ "$VERSION_ID" == "20.04" ];then
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/glog_0.6.0_ubuntu.tar.gz
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.5.5_ubuntu.tar.gz
        elif [ "$VERSION_ID" == "22.04" ];then
            download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_ubuntu.tar.gz
        fi
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3_ubuntu.tar.gz
    elif [ "$OS_NAME" == "openEuler" ];then
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0.tar.gz
        download https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obssdk_3.22.3.tar.gz
    fi
elif [ "$(arch)" == "aarch64" ];then
    if [ "$OS_NAME" == "Ubuntu" ];then
        cp -af /opt/ubuntu/* .
        sed -i '1d;/COPY --from=0/d' ${CUR_DIR}/Dockerfile.ascend.runtime.ubuntu
        sed -i '/ADD/i\COPY Ascend_run /usr/local/Ascend' ${CUR_DIR}/Dockerfile.ascend.runtime.ubuntu
        sed -i '/ADD/i\COPY npu-smi /usr/local/sbin/npu-smi' ${CUR_DIR}/Dockerfile.ascend.runtime.ubuntu
        if [ "$VERSION_ID" == "18.04" ];then
            download http://download.modelbox-ai.com/third-party/aarch64/opencv_4.2.0_ubuntu.tar.gz
        elif [ "$VERSION_ID" == "20.04" ];then
            download http://download.modelbox-ai.com/third-party/aarch64/glog_0.6.0_ubuntu.tar.gz
            download http://download.modelbox-ai.com/third-party/aarch64/opencv_4.5.5_ubuntu.tar.gz
        fi
        download http://download.modelbox-ai.com/third-party/aarch64/obssdk_3.22.3_ubuntu.tar.gz
    elif [ "$OS_NAME" == "openEuler" ];then
        cp -af /opt/openeuler/* .
        sed -i '1d;/COPY --from=0/d' ${CUR_DIR}/Dockerfile.ascend.runtime.openeuler
        sed -i '/ADD/i\COPY Ascend_run /usr/local/Ascend' ${CUR_DIR}/Dockerfile.ascend.runtime.openeuler
        sed -i '/ADD/i\COPY npu-smi /usr/local/sbin/npu-smi' ${CUR_DIR}/Dockerfile.ascend.runtime.openeuler
        download http://download.modelbox-ai.com/third-party/aarch64/cpprestsdk_2.10.18.tar.gz
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
