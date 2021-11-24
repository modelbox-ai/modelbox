set(TRT_ROOT /tensorrt)
set(HINT_DIRS $ENV{TRT_RELEASE}/include ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${TRT_ROOT}/include)
set(HINT_LIBS $ENV{TRT_RELEASE}/lib ${CMAKE_INSTALL_FULL_LIBDIR} ${TRT_ROOT}/lib)
find_path(TENSORRT_INCLUDE 
    NAMES NvCaffeParser.h
          NvInfer.h
          NvInferPlugin.h
          NvOnnxConfig.h
          NvOnnxParser.h
          NvOnnxParserRuntime.h
          NvUffParser.h
          NvUtils.h
          NvInferPluginUtils.h
          NvInferRuntimeCommon.h
          NvInferRuntime.h
          NvInferVersion.h
    HINTS ${HINT_DIRS})

mark_as_advanced(TENSORRT_INCLUDE)

find_library(TENSORRT_LIBRARY NAMES nvinfer HINTS ${HINT_LIBS})
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin HINTS ${HINT_LIBS})
find_library(TRT_CAFFEPARSER_LIBRARY NAMES nvcaffe_parser nvparsers HINTS ${HINT_LIBS})
find_library(TRT_ONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_runtime HINTS ${HINT_LIBS})

mark_as_advanced(TENSORRT_LIBRARY)

if(TENSORRT_INCLUDE AND EXISTS "${TENSORRT_INCLUDE}/NvInfer.h")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInfer.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInfer.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInfer.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    
    set(TENSORRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
    message("find tensorrt version: " ${TENSORRT_VERSION})
    set(TENSORRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

if(TENSORRT_INCLUDE AND EXISTS "${TENSORRT_INCLUDE}/NvInferVersion.h")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TENSORRT_INCLUDE}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    
    set(TENSORRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
    message("find tensorrt version: " ${TENSORRT_VERSION})
    set(TENSORRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TENSORRT
                                  REQUIRED_VARS 
                                  TENSORRT_LIBRARY 
                                  TENSORRT_INCLUDE 
                                  TRT_CAFFEPARSER_LIBRARY 
                                  TRT_ONNXPARSER_LIBRARY 
                                  TENSORRT_PLUGIN_LIBRARY
                                  TENSORRT_VERSION_STRING
                                  VERSION_VAR TENSORRT_VERSION_STRING)

if(TENSORRT_FOUND)
  set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY} ${TRT_CAFFEPARSER_LIBRARY} ${TRT_ONNXPARSER_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})
  set(TENSORRT_INCLUDE_DIR ${TENSORRT_INCLUDE})
  set(TENSORRT_VERSION ${TENSORRT_VERSION_STRING})
endif()