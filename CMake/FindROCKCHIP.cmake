if(DEFINED ENV{ROCKCHIP_PATH})
    set(HINTS_ROCKCHIP_PATH $ENV{ROCKCHIP_PATH})

    message(STATUS "DEFINED ${HINTS_ROCKCHIP_PATH}")
else()
    set(HINTS_ROCKCHIP_PATH "/usr/local/rockchip")

    message(STATUS "set default search path: /usr/local/rockchip")
endif()

find_path(ROCKCHIP_RGA_INCLUDE NAMES im2d.h rga.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rk-rga/include)
mark_as_advanced(ROCKCHIP_RGA_INCLUDE)

find_path(ROCKCHIP_MPP_INCLUDE NAMES rk_mpi.h rk_type.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rkmpp/include/rockchip)
mark_as_advanced(ROCKCHIP_MPP_INCLUDE)

find_path(RKNN_INCLUDE NAMES rknn_api.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rknn/include)
mark_as_advanced(RKNN_INCLUDE)

find_path(RKNPU2_INCLUDE NAMES rknn_api.h
    HINTS ${HINTS_ROCKCHIP_PATH}/rknnrt/include)
mark_as_advanced(RKNPU2_INCLUDE)

find_library(RKNN_LIBRARY NAMES rknn_api HINTS ${HINTS_ROCKCHIP_PATH}/rknn/lib)
mark_as_advanced(RKNN_LIBRARY)

find_library(RKNPU2_LIBRARY NAMES rknnrt HINTS ${HINTS_ROCKCHIP_PATH}/rknnrt/lib)
mark_as_advanced(RKNPU2_LIBRARY)

find_library(RKRGA_LIBRARY NAMES rga HINTS ${HINTS_ROCKCHIP_PATH}/rk-rga/lib)
mark_as_advanced(RKRGA_LIBRARY)

find_library(RKMPP_LIBRARY NAMES rockchip_mpp HINTS ${HINTS_ROCKCHIP_PATH}/rkmpp/lib)
mark_as_advanced(RKMPP_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ROCKCHIP
    REQUIRED_VARS ROCKCHIP_RGA_INCLUDE ROCKCHIP_MPP_INCLUDE
    VERSION_VAR ROCKCHIP_VERSION_STRING)

if(ROCKCHIP_FOUND)
    set(RKNN_LIBRARIES ${RKNN_LIBRARY})
    set(RKNN_INCLUDE_DIR ${RKNN_INCLUDE})
    set(RKNPU2_INCLUDE_DIR ${RKNPU2_INCLUDE})
    set(RKNPU2_LIBRARIES ${RKNPU2_LIBRARY})
    set(RKRGA_LIBRARIES ${RKRGA_LIBRARY})
    set(RKMPP_LIBRARIES ${RKMPP_LIBRARY})
    set(ROCKCHIP_INCLUDE_DIR ${ROCKCHIP_RGA_INCLUDE} ${ROCKCHIP_MPP_INCLUDE})

    message(STATUS "rockchip dependency found, ${ROCKCHIP_INCLUDE_DIR}")
endif()