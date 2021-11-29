set(HITS_MINDSPORE_PATH $ENV{MINDSPORE_PATH})
find_path(MINDSPORE_INCLUDE NAMES include/api/context.h include/api/graph.h include/api/model.h HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${HITS_MINDSPORE_PATH})
mark_as_advanced(MINDSPORE_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(MINDSPORE_LIBRARY NAMES
    mindspore
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${HITS_MINDSPORE_PATH}/lib
)
mark_as_advanced(MINDSPORE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MINDSPORE
                                  REQUIRED_VARS MINDSPORE_LIBRARY MINDSPORE_INCLUDE
                                  VERSION_VAR MINDSPORE_VERSION_STRING)

if(MINDSPORE_FOUND)
  set(MINDSPORE_LIBRARIES ${MINDSPORE_LIBRARY})
  set(MINDSPORE_INCLUDE_DIR ${MINDSPORE_INCLUDE})
endif()
