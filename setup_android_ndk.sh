#!/bin/bash
CODE_DIR=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
export NDK_ROOT=/data/ndk/android-ndk-r25b
export USER_ROOT=/data/devel/thirdparty/deb
export LIBRARY_RUNPATH=/data/user/0/com.mbox_ai/files/lib
export MINDSPORE_LITE_PATH=/data/mindspore/mindspore-lite-1.9.0
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH
export DEBIAN_FRONTEND=noninteractive
export $(dbus-launch)
export RUNLEVEL=3

apt update
apt install -y --no-install-recommends \
    vim curl zip unzip git maven make strace iproute2 \
    pkg-config openjdk-8-jdk ca-certificates patchelf \
    python3-pip ubuntu-desktop clang-tidy-14 cmake
ln -sf clang-tidy-14 /usr/bin/clang-tidy
ln -sf run-clang-tidy-14 /usr/bin/run-clang-tidy

curl --proto '=https' --tlsv1.2 -Sf https://repo.waydro.id/waydroid.gpg --output /usr/share/keyrings/waydroid.gpg
echo "deb [signed-by=/usr/share/keyrings/waydroid.gpg] https://repo.waydro.id/ jammy main" | tee /etc/apt/sources.list.d/waydroid.list
apt update
apt install -y waydroid=1.3.3

mkdir -p ~/.pip
echo "[global]" > ~/.pip/pip.conf
echo "index-url = https://pypi.python.org/simple" >> ~/.pip/pip.conf
echo "trusted-host = pypi.python.org" >> ~/.pip/pip.conf
echo "timeout = 120" >> ~/.pip/pip.conf
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir setuptools pyclip

sed -i '3s/OS_NAME=.*/OS_NAME=android/' ${CODE_DIR}/docker/prepare_for_dev.sh
sed -i '/ls -lh release/d' ${CODE_DIR}/docker/prepare_for_dev.sh
bash -x ${CODE_DIR}/docker/prepare_for_dev.sh
ls -lh ${CODE_DIR}

mkdir -p /data/ndk
unzip -o android-ndk-r25b-linux.zip -d /data/ndk
ls -lh /data/ndk

tar zxf cpprestsdk.tar.gz -C /data
ls -lh /data/cpprestsdk/{x86_64,arm64-v8a}/lib/*

mkdir -p /data/mindspore
tar zxf mindspore-lite-1.9.0-linux-x64.tar.gz -C /data/mindspore
tar zxf mindspore-lite-1.9.0-android-aarch64.tar.gz -C /data/mindspore
ls -lh /data/mindspore

mkdir -p /data/devel/thirdparty
tar zxf src.tar.gz -C /data/devel/thirdparty
tar zxf deb_files.tar.gz -C /data/devel/thirdparty
ls -lh /data/devel/thirdparty

ln -sf data/data/com.termux/files/usr /data/devel/thirdparty/deb/x86_64/usr
ln -sf data/data/com.termux/files/usr /data/devel/thirdparty/deb/arm64-v8a/usr
ln -sf opencv4/opencv2 /data/devel/thirdparty/deb/x86_64/usr/include/opencv2
ln -sf opencv4/opencv2 /data/devel/thirdparty/deb/arm64-v8a/usr/include/opencv2
cp -af /data/cpprestsdk/arm64-v8a/lib/libcpprest.so /data/devel/thirdparty/deb/arm64-v8a/usr/lib/
cp -af /data/cpprestsdk/x86_64/lib/libcpprest.so /data/devel/thirdparty/deb/x86_64/usr/lib/
ln -sf /data/cpprestsdk/include/cpprest /data/devel/thirdparty/deb/arm64-v8a/usr/include/cpprest
ln -sf /data/cpprestsdk/include/cpprest /data/devel/thirdparty/deb/x86_64/usr/include/cpprest
ln -sf /data/cpprestsdk/include/pplx /data/devel/thirdparty/deb/arm64-v8a/usr/include/pplx
ln -sf /data/cpprestsdk/include/pplx /data/devel/thirdparty/deb/x86_64/usr/include/pplx

mkdir -p /data/devel/stub
cd /data/devel/stub
touch test.c
/data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang -fPIC -shared -Wl,--build-id -Wl,-soname,libpthread.so test.c -o libpthread.so.a64
/data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang -fPIC -shared -Wl,--build-id -Wl,-soname,librt.so test.c -o librt.so.a64
/data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android28-clang -fPIC -shared -Wl,--build-id -Wl,-soname,libpthread.so test.c -o libpthread.so.x64
/data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android28-clang -fPIC -shared -Wl,--build-id -Wl,-soname,librt.so test.c -o librt.so.x64
cp libpthread.so.a64 /data/devel/thirdparty/deb/arm64-v8a/usr/lib/libpthread.so
cp librt.so.a64 /data/devel/thirdparty/deb/arm64-v8a/usr/lib/librt.so
cp libpthread.so.x64 /data/devel/thirdparty/deb/x86_64/usr/lib/libpthread.so
cp librt.so.x64 /data/devel/thirdparty/deb/x86_64/usr/lib/librt.so

echo "***android ndk build setup completed***"
