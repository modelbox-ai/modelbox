#!/bin/bash
CODE_DIR=$(cd $(dirname $0)/..;pwd)

echo ${CODE_DIR}
PLATFORM_NAME=$1
if [ "${PLATFORM_NAME}" == "356x" ] || [ "${PLATFORM_NAME}" == "3588" ]; then
  echo "build ${PLATFORM_NAME}"
else
  echo "no support ${PLATFORM_NAME}"
  exit 1
fi

mkdir -p ${CODE_DIR}/rockchip
unset https_proxy
unset http_proxy
apt update
apt-get install -y gnutls-bin
git config --global http.sslVerify false

# download lib rga
function download_rga() {
  mkdir -p ${CODE_DIR}/githubrga
  cd ${CODE_DIR}/githubrga
  if [ "${PLATFORM_NAME}" == "356x" ]; then
    wget https://ghproxy.com/github.com/airockchip/librga/archive/refs/heads/1.3.2_release.zip
    unzip 1.3.2_release.zip
    LIB_RGA_PATH=${CODE_DIR}/githubrga/librga-1.3.2_release
  elif [ "${PLATFORM_NAME}" == "3588" ]; then
    wget https://ghproxy.com/github.com/airockchip/librga/archive/refs/heads/main.zip
    unzip main.zip
    LIB_RGA_PATH=${CODE_DIR}/githubrga/librga-main
  else
    echo "no support ${PLATFORM_NAME}"
    exit 1
  fi

  echo "download rga finish"

  mkdir -p ${CODE_DIR}/rga/
  mkdir -p ${CODE_DIR}/rga/libs/Linux/gcc-aarch64
  cp -rfd ${LIB_RGA_PATH}/include ${CODE_DIR}/rga
  cp -rfd ${LIB_RGA_PATH}/libs/Linux/gcc-aarch64/* ${CODE_DIR}/rga/libs/Linux/gcc-aarch64/

  cd ${CODE_DIR}
  tar -czf rga.tar.gz rga
  mv rga.tar.gz ${CODE_DIR}/rockchip/rga.tar.gz

  rm -rf githubrga rga
}

# download lib rknpu
download_rknpu() {
  mkdir -p ${CODE_DIR}/githubrknpu
  cd ${CODE_DIR}/githubrknpu
  wget https://ghproxy.com/github.com/airockchip/RK3399Pro_npu/archive/refs/heads/main.zip
  unzip main.zip

  echo "download rknpu finish"

  RKNPU_PATH=rknn-api/librknn_api/
  mkdir -p ${CODE_DIR}/rknpu
  mkdir -p ${CODE_DIR}/rknpu/${RKNPU_PATH}/Linux/lib64
  cp -rfd ${CODE_DIR}/githubrknpu/RK3399Pro_npu-main/${RKNPU_PATH}/include ${CODE_DIR}/rknpu/${RKNPU_PATH}
  cp -rfd ${CODE_DIR}/githubrknpu/RK3399Pro_npu-main/${RKNPU_PATH}/Linux/lib64/* ${CODE_DIR}/rknpu/${RKNPU_PATH}/Linux/lib64

  cd ${CODE_DIR}
  tar -czf rknpu.tar.gz rknpu
  mv rknpu.tar.gz ${CODE_DIR}/rockchip/rknpu.tar.gz

  rm -rf rknpu githubrknpu
}

# download lib rknpu2
download_rknpu2() {
  mkdir -p ${CODE_DIR}/githubrknpu2
  cd ${CODE_DIR}/githubrknpu2
  wget https://ghproxy.com/github.com/rockchip-linux/rknpu2/archive/refs/heads/master.zip
  unzip master.zip
  mkdir -p ${CODE_DIR}/rknpu2
  if [ "${PLATFORM_NAME}" == "356x" ]; then
    RKNPU2_PATH=runtime/RK356X/Linux/librknn_api
    mkdir -p ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
    cp -rfd ${CODE_DIR}/githubrknpu2/rknpu2-master/${RKNPU2_PATH}/include ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
    cp -rfd ${CODE_DIR}/githubrknpu2/rknpu2-master/${RKNPU2_PATH}/aarch64 ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
  elif [ "${PLATFORM_NAME}" == "3588" ]; then
    RKNPU2_PATH=runtime/RK3588/Linux/librknn_api
    mkdir -p ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
    cp -rfd ${CODE_DIR}/githubrknpu2/rknpu2-master/${RKNPU2_PATH}/include ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
    cp -rfd ${CODE_DIR}/githubrknpu2/rknpu2-master/${RKNPU2_PATH}/aarch64 ${CODE_DIR}/rknpu2/${RKNPU2_PATH}
  else
    echo "no support ${PLATFORM_NAME}"
    exit 1
  fi
  
  echo "download rknpu2 finish"
 
  cd ${CODE_DIR}
  tar -czf rknpu2.tar.gz rknpu2
  mv rknpu2.tar.gz ${CODE_DIR}/rockchip/rknpu2.tar.gz

  rm -rf rknpu2 githubrknpu2
}

# download mpp and build lib mpp
download_rkmpp() {
  mkdir -p ${CODE_DIR}/mpp
  cd ${CODE_DIR}/mpp
  wget https://ghproxy.com/github.com/rockchip-linux/mpp/archive/refs/heads/develop.zip
  unzip develop.zip
  echo "download mpp finish"

  if [ -d "${CODE_DIR}/mpp/mpp-develop/mpp/release" ]; then
    rm -rf ${CODE_DIR}/mpp/mpp-develop/mpp/release
  fi

  cd ${CODE_DIR}/mpp/mpp-develop/build/linux/aarch64
  ./make-Makefiles.bash
  make -j4

  mkdir -p ${CODE_DIR}/rkmpp
  mkdir -p ${CODE_DIR}/rkmpp/include
  mkdir -p ${CODE_DIR}/rkmpp/lib
  
  cp -rfd ${CODE_DIR}/mpp/mpp-develop/inc/* ${CODE_DIR}/rkmpp/include
  cp -rfd ${CODE_DIR}/mpp/mpp-develop/utils/camera_source.h ${CODE_DIR}/rkmpp/include
  cp -rfd ${CODE_DIR}/mpp/mpp-develop/build/linux/aarch64/mpp/librockchip* ${CODE_DIR}/rkmpp/lib
  cp -rfd ${CODE_DIR}/mpp/mpp-develop/build/linux/aarch64/utils/libutils.a ${CODE_DIR}/rkmpp/lib/librk_utils.a
  
  cd ${CODE_DIR}
  tar -czf rkmpp.tar.gz rkmpp
  mv rkmpp.tar.gz ${CODE_DIR}/rockchip/rkmpp.tar.gz
  
  rm -rf mpp rkmpp
}

download_rga
download_rknpu
download_rknpu2
download_rkmpp