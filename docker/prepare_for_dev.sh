#!/bin/bash
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
if [ "$VERSION_ID" == "18.04" ];then
    ExecStar=/sbin/ldconfig
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obs_3.21.8_dev-ubuntu.tar.gz
elif [ "$VERSION_ID" == "20.04" ];then
    ExecStar=/usr/sbin/ldconfig
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15_dev.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0_dev.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4_dev.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obs_3.21.8_dev.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0_dev.tar.gz
fi
curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/nlohmann-json_3.7.3.tar.gz
curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/Video_Codec_SDK_9.1.23.tar.gz

cat << EOF >ldconfig.service
[Unit]
Description=run ldconfig
[Service]
type=oneshot
ExecStar=$ExecStar
[Install]
WantedBy=multi-user.target
EOF

cat ldconfig.service
chmod 755 ldconfig.service

ls -lh ./
ls -lh ./release
