if(DEFINED ENV{ROCKCHIP_PATH})
    set(HINTS_ROCKCHIP_PATH $ENV{ROCKCHIP_PATH})

    message(STATUS "DEFINED ${HINTS_ROCKCHIP_PATH}")
else()
    set(HINTS_ROCKCHIP_PATH "/usr/local/rockchip")

    message(STATUS "set default search path: /usr/local/rockchip")
endif()

find_path(ROCKCHIP_RGA_INCLUDE NAMES im2d.h rga.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rga/include)
mark_as_advanced(ROCKCHIP_RGA_INCLUDE)

find_path(ROCKCHIP_MPP_INCLUDE NAMES rk_mpi.h rk_type.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rkmpp/include)
mark_as_advanced(ROCKCHIP_MPP_INCLUDE)

find_path(RKNPU_INCLUDE NAMES rknn_api.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rknpu/rknn-api/librknn_api/include)
mark_as_advanced(RKNPU_INCLUDE)

find_library(RKNPU_LIBRARY NAMES rknn_api HINTS ${HINTS_ROCKCHIP_PATH}/rknpu/rknn-api/librknn_api/Linux/lib64/)
mark_as_advanced(RKNPU_LIBRARY)

if(EXISTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK3588)
    find_path(RKNPU2_INCLUDE NAMES rknn_api.h
        HINTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK3588/Linux/librknn_api/include)
    mark_as_advanced(RKNPU2_INCLUDE)

    find_library(RKNPU2_LIBRARY NAMES rknnrt HINTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64)
    mark_as_advanced(RKNPU2_LIBRARY)
    message(STATUS "rk3588 platform")
elseif(EXISTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK356X)
    find_path(RKNPU2_INCLUDE NAMES rknn_api.h
        HINTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK356X/Linux/librknn_api/include)
    mark_as_advanced(RKNPU2_INCLUDE)

    find_library(RKNPU2_LIBRARY NAMES rknnrt HINTS ${HINTS_ROCKCHIP_PATH}/rknpu2/runtime/RK356X/Linux/librknn_api/aarch64)
    mark_as_advanced(RKNPU2_LIBRARY)
    message(STATUS "rk356x platform")
else()
    message(STATUS "other platform")
endif()

find_library(RKRGA_LIBRARY NAMES rga HINTS ${HINTS_ROCKCHIP_PATH}/rga/libs/Linux/gcc-aarch64)
mark_as_advanced(RKRGA_LIBRARY)

find_library(RKMPP_LIBRARY NAMES rockchip_mpp HINTS ${HINTS_ROCKCHIP_PATH}/rkmpp/lib)
mark_as_advanced(RKMPP_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ROCKCHIP
    REQUIRED_VARS ROCKCHIP_RGA_INCLUDE ROCKCHIP_MPP_INCLUDE
    VERSION_VAR ROCKCHIP_VERSION_STRING)

if(ROCKCHIP_FOUND)
    set(RKNPU_LIBRARIES ${RKNPU_LIBRARY})
    set(RKNPU_INCLUDE_DIR ${RKNPU_INCLUDE})
    set(RKNPU2_INCLUDE_DIR ${RKNPU2_INCLUDE})
    set(RKNPU2_LIBRARIES ${RKNPU2_LIBRARY})
    set(RKRGA_LIBRARIES ${RKRGA_LIBRARY})
    set(RKMPP_LIBRARIES ${RKMPP_LIBRARY})
    set(ROCKCHIP_INCLUDE_DIR ${ROCKCHIP_RGA_INCLUDE} ${ROCKCHIP_MPP_INCLUDE})

    message(STATUS "rockchip dependency found, ${ROCKCHIP_INCLUDE_DIR}")
endif()