set(HITS_MINDSPORE_LITE_PATH $ENV{MINDSPORE_LITE_PATH})
find_path(MINDSPORE_LITE_DIR NAMES 
  runtime/include/api/context.h 
  runtime/include/api/graph.h 
  runtime/include/api/model.h 
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${HITS_MINDSPORE_LITE_PATH} /usr/local/mindspore-lite)
mark_as_advanced(MINDSPORE_LITE_DIR)

# Look for the library (sorted from most current/relevant entry to least).
find_library(MINDSPORE_LITE_LIBRARY NAMES
    mindspore-lite
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${HITS_MINDSPORE_LITE_PATH}/runtime/lib /usr/local/mindspore-lite/runtime/lib
)
mark_as_advanced(MINDSPORE_LITE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MINDSPORE_LITE
                                  REQUIRED_VARS MINDSPORE_LITE_LIBRARY MINDSPORE_LITE_DIR
                                  VERSION_VAR MINDSPORE_VERSION_STRING)

if(MINDSPORE_LITE_FOUND)
  set(MINDSPORE_LITE_LIBRARIES ${MINDSPORE_LITE_LIBRARY})
  set(MINDSPORE_LITE_INCLUDE_DIR ${MINDSPORE_LITE_DIR}/runtime)
  set(MINDSOPRE_LITE_LIB_DIR ${MINDSPORE_LITE_DIR}/runtime/lib)
endif()
