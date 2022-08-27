if(DEFINED ENV{ROCKCHIP_PATH})
    set(HINTS_ROCKCHIP_PATH $ENV{ROCKCHIP_PATH})

    message(STATUS "DEFINED ${HINTS_ROCKCHIP_PATH}")
else()
    set(HINTS_ROCKCHIP_PATH "/opt/rockchip")

    message(STATUS "set default search path: /opt/rockchip")
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

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ROCKCHIP
    REQUIRED_VARS ROCKCHIP_RGA_INCLUDE ROCKCHIP_MPP_INCLUDE
    VERSION_VAR ROCKCHIP_VERSION_STRING)

if(ROCKCHIP_FOUND)
    set(RKNN_INCLUDE_DIR ${RKNN_INCLUDE})
    set(RKNPU2_INCLUDE_DIR ${RKNPU2_INCLUDE})
    set(ROCKCHIP_INCLUDE_DIR ${ROCKCHIP_RGA_INCLUDE} ${ROCKCHIP_MPP_INCLUDE})

    message(STATUS "rockchip dependency found, ${ROCKCHIP_INCLUDE_DIR}")
endif()