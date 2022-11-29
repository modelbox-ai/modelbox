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

compile() {
    rm -f /data/data /data/mindspore/mindspore-lite-1.9.0
    if [ "$1" == "a64" ];then
        export BUILD_DIR=build_a64
        export ANDROID_ABI=arm64-v8a
        export ARCH=aarch64
        ln -s mindspore-lite-1.9.0-android-aarch64 /data/mindspore/mindspore-lite-1.9.0
    elif [ "$1" == "x64" ];then
        export BUILD_DIR=build_x64
        export ANDROID_ABI=x86_64
        export ARCH=x86_64
        ln -s mindspore-lite-1.9.0-linux-x64 /data/mindspore/mindspore-lite-1.9.0
    fi
    ln -s /data/devel/thirdparty/deb/${ANDROID_ABI}/data/data /data/data

    ls -lh /data/data
    ls -lh /data/mindspore/mindspore-lite-1.9.0

    if [ -d ${CODE_DIR}/${BUILD_DIR} ];then
        rm -rf ${CODE_DIR}/${BUILD_DIR}/*
    else
        mkdir -p ${CODE_DIR}/${BUILD_DIR}
    fi

    cd ${CODE_DIR}/${BUILD_DIR}
    cmake .. \
        -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_ANDROID_API=28 \
        -DLOCAL_PACKAGE_PATH=/data/devel/thirdparty/src \
        -DCLANG_TIDY=OFF \
        -DCLANG_TIDY_AS_ERROR=OFF \
        -DWITH_MINDSPORE=on \
        -DWITH_JAVA=ON \
        -DTEST_WORKING_DIR=/data/user/test \
        -DPYTHON_VERSION_ANDROID=3.10 \
        -DANDROID_ABI=${ANDROID_ABI} \
        -DANDROID_PLATFORM=android-28 \
        -DANDROID_NDK=${NDK_ROOT} \
        -DANDROID_TOOLCHAIN=clang \
        -DANDROID_STL=c++_shared \
        -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE \
        -DBUILD_SHARED_LIBS=1 \
        -Wno-deprecated \
        -Wno-dev

    if [ $? -ne 0 ];then
        echo "cmake failed"
        cat CMakeFiles/CMakeError.log
        exit 1
    fi

    make package -j8
}

prepare(){
    cd ${CODE_DIR}/${BUILD_DIR}
    mkdir -p ${ANDROID_ABI}/jni/${ANDROID_ABI}
    mkdir -p ${ANDROID_ABI}/{libmodelbox-kernel,libmodelbox-drivers,assets}

    cp cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/java/packages/lib/libmodelbox-jni.so ${ANDROID_ABI}/jni/${ANDROID_ABI}/

    ls -lh cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/local/lib/

    cp /data/devel/thirdparty/deb/${ANDROID_ABI}/usr/lib/{libssl.so.1.1,libcrypto.so.1.1} ${ANDROID_ABI}/libmodelbox-kernel/
    cp cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/local/lib/{libmodelbox.so,libsecurec.so} ${ANDROID_ABI}/libmodelbox-kernel/
    cp /data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${ARCH}-linux-android/libc++_shared.so ${ANDROID_ABI}/libmodelbox-kernel/

    cp /data/mindspore/mindspore-lite-1.9.0/runtime/lib/libmindspore-lite.so ${ANDROID_ABI}/libmodelbox-drivers/
    cp /data/devel/thirdparty/deb/${ANDROID_ABI}/usr/lib/{libpython3.10.so.1.0,libcgraph.so,libcdt.so,libz.so} ${ANDROID_ABI}/libmodelbox-drivers/
    cp /data/devel/thirdparty/deb/${ANDROID_ABI}/usr/lib/{libandroid-support.so,libjpeg.so,liblzma.so,libopenblas.so,libopenjp2.so,libpng16.so,libprotobuf.so,libtiff.so,libwebp.so,libzstd.so,libopencv*.so} ${ANDROID_ABI}/libmodelbox-drivers/
    cp cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/local/lib/{libmbox-engine-mindspore-lite.so,libmodelbox-device-cpu.so,libmodelbox-graphconf-graphviz.so,libmodelbox-unit-cpu-mindspore-lite-inference.so,libmodelbox-unit-cpu-python.so,libmodelbox-virtualdriver-inference.so,libmodelbox-virtualdriver-python.so} ${ANDROID_ABI}/libmodelbox-drivers/

    cd ${ANDROID_ABI}/libmodelbox-drivers
    find . -name "lib*" | xargs patchelf --set-rpath /data/user/0/com.mbox_ai/files/lib
    zip -r ../assets/libmodelbox-drivers_${ANDROID_ABI}.zip ./
    md5sum ../assets/libmodelbox-drivers_${ANDROID_ABI}.zip > ../assets/libmodelbox-drivers_${ANDROID_ABI}.zip.md5

    cd ../libmodelbox-kernel
    patchelf --set-rpath /data/user/0/com.mbox_ai/files/lib libssl.so.1.1
    zip -r ../assets/libmodelbox-kernel_${ANDROID_ABI}.zip ./
    md5sum ../assets/libmodelbox-kernel_${ANDROID_ABI}.zip > ../assets/libmodelbox-kernel_${ANDROID_ABI}.zip.md5

    cd ..
    cp -r /data/devel/thirdparty/deb/${ANDROID_ABI}/usr/lib/python3.10 .
    cp -r python3.10/site-packages/numpy-1.23.3-py3.10-linux-${ARCH}.egg/numpy python3.10/site-packages/
    rm -rf python3.10/site-packages/numpy-1.23.3-py3.10-linux-${ARCH}.egg
    cp -r /data/devel/thirdparty/deb/${ANDROID_ABI}/data/data/com.termux/files/usr/lib/python3.10/site-packages/cv2 python3.10/site-packages/

    cd python3.10
    find . -name "*.so" | xargs patchelf --set-rpath /data/user/0/com.mbox_ai/files/lib

    cd ..
    zip -r assets/python3.10_${ANDROID_ABI}.zip python3.10
    md5sum assets/python3.10_${ANDROID_ABI}.zip > assets/python3.10_${ANDROID_ABI}.zip.md5

    rm -rf libmodelbox-drivers libmodelbox-kernel python3.10

    ls -lh ${CODE_DIR}/${BUILD_DIR}/${ANDROID_ABI}/*
}

package(){
    mkdir ${CODE_DIR}/aar_pkg
    cat << 'EOF'>${CODE_DIR}/aar_pkg/AndroidManifest.xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.modelbox"
    android:versionCode="1">

    <uses-sdk
        android:minSdkVersion="28"
        android:targetSdkVersion="30"/>

</manifest>
EOF

    cp ${CODE_DIR}/build_a64/src/java/modelbox-1.0.0.jar ${CODE_DIR}/aar_pkg/classes.jar
    cp -af ${CODE_DIR}/build_x64/x86_64/* ${CODE_DIR}/aar_pkg/
    cp -af ${CODE_DIR}/build_a64/arm64-v8a/* ${CODE_DIR}/aar_pkg/
    chown -R root.root ${CODE_DIR}/aar_pkg
    find ${CODE_DIR}/aar_pkg -type d|xargs chmod -R 755
    find ${CODE_DIR}/aar_pkg -type f|xargs chmod -R 644
    cd ${CODE_DIR}/aar_pkg
    jar -cvf modelbox.aar -C ${CODE_DIR}/aar_pkg .
}

buildtest(){
    cd ${CODE_DIR}/build_x64
    make unit -j8
    mkdir -p /data/user/test/{dep,drivers}
    cp lib/{libgmock.so,libgtest.so} /data/user/test/dep/
    cp cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/local/lib/{libmodelbox.so,libsecurec.so,libmanager-client.so} /data/user/test/dep/
    cp cpack/_CPack_Packages/Android/TGZ/modelbox-1.0.0-Android/usr/local/lib/{libmodelbox-unit-cpu-python.so,libmodelbox-device-cpu.so,libmodelbox-virtualdriver-python.so,libmodelbox-virtualdriver-inference.so,libmodelbox-graphconf-graphviz.so} /data/user/test/drivers/
    cp test/mock/minimodelbox/libflowmock-lib.so /data/user/test/dep/
    cp test/mock/drivers/libmock-driver-ctrl-lib.so /data/user/test/dep/
    cp test/mock/drivers/device_mockdevice/libmodelbox-device-mockdevice.so /data/user/test/dep/
    cp test/mock/drivers/flowunit_mockflowunit/libmodelbox-unit-mockdevice-mockflowunit.so /data/user/test/dep/
    cp test/mock/drivers/graph_conf_mockgraphconf/libmodelbox-graphconf-mockgraphconf.so /data/user/test/dep/
    cp test/unit/unit /data/user/test/bin/
    cp /data/devel/thirdparty/deb/x86_64/usr/lib/{libssl.so.1.1,libcrypto.so.1.1,libcgraph.so,libcdt.so,libpython3.10.so.1.0} /data/user/test/dep/
    cp /data/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android/libc++_shared.so /data/user/test/dep/
    cp -r ../test/assets /data/user/test/
    cp -r /data/devel/thirdparty/deb/x86_64/usr/lib/python3.10 /data/user/test/ 
    cp ../docker/unit.sh /data/user/test/bin/
    chmod +x /data/user/test/bin/*
    cd /data/user/
    tar zcf unittest.tar.gz test
    cp unittest.tar.gz /data/devel/modelbox
}

unittest(){
    apt update
    apt install -y linux-modules-extra-$(uname -r) waydroid dbus dbus-x11 #weston

    if [ $(ls /usr/share/waydroid-extra/images|grep img|wc -l) -eq 2 ];then
        waydroid init -f -i /usr/share/waydroid-extra/images
    else
        waydroid init
    fi

    mkdir -p /tmp/runtime/wayland-0
    chmod 700 /tmp/runtime
    /etc/init.d/dbus start
    eval $(dbus-launch --sh-syntax)
    export $(dbus-launch)
    #/usr/lib/waydroid/data/scripts/waydroid-net.sh start
    waydroid container start &
    waydroid session start &

    mkdir -p ~/.local/share/waydroid/data/user
    ls -lh /data/devel/modelbox/unittest.tar.gz
    tar zxf /data/devel/modelbox/unittest.tar.gz -C ~/.local/share/waydroid/data/user/
    ls -lh ~/.local/share/waydroid/data/user/test
    waydroid shell ./data/user/test/bin/unit.sh | tee unittest.log

    times=0
    while [ $(cat unittest.log|grep -c AesEncryptPass) -lt 1 ];do
        cat /var/lib/waydroid/waydroid.log
        if [ $(cat /var/lib/waydroid/waydroid.log|grep -c "waiting for session to load") -lt 1 ];then
            waydroid session start &
        fi
        waydroid container restart &
        waydroid shell ./data/user/test/bin/unit.sh | tee unittest.log
        times=$[${times}+1]
        if [ ${times} -gt 3 ]; then
            echo "waydroid container start failed,pls check"
            exit 1
        fi
    done
}

main(){
    if [ "$1" == "x64" ];then
        compile x64
    elif [ "$1" == "a64" ];then
        compile a64
    elif [ "$1" == "buildtest" ];then
        buildtest
    elif [ "$1" == "unittest" ];then
        unittest
    else
        for arch in x64 a64;do
            compile $arch
            prepare
        done
        package
    fi
}

main $@
