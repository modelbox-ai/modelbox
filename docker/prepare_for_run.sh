#!/bin/bash
VERSION_ID=$(sed -nr '/VERSION_ID/s/^VERSION_ID="(.*)"$/\1/gp' /etc/os-release)
if [ "$VERSION_ID" == "18.04" ];then
    ExecStar=/sbin/ldconfig
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obs_3.21.8-ubuntu.tar.gz
elif [ "$VERSION_ID" == "20.04" ];then
    ExecStar=/usr/sbin/ldconfig
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/cpprestsdk_2.10.15.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/duktape_2.6.0.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/ffmpeg_4.4.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/obs_3.21.8.tar.gz
    curl -LJO https://github.com/modelbox-ai/modelbox-binary/releases/download/BinaryArchive/opencv_4.2.0.tar.gz
fi

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
