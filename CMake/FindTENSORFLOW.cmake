find_path(TENSORFLOW_INCLUDE NAMES tensorflow/c/c_api.h HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR})
mark_as_advanced(TENSORFLOW_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(TENSORFLOW_LIBRARY NAMES
    tensorflow
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR}
)
mark_as_advanced(TENSORFLOW_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TENSORFLOW
                                  REQUIRED_VARS TENSORFLOW_LIBRARY TENSORFLOW_INCLUDE
                                  VERSION_VAR TENSORFLOW_VERSION_STRING)

if(TENSORFLOW_FOUND)
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})
  set(TENSORFLOW_INCLUDE_DIR ${TENSORFLOW_INCLUDE})
endif()
