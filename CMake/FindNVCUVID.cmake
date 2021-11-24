find_path(NVCUVID_INCLUDE NAMES nvcuvid.h HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR})
mark_as_advanced(NVCUVID_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
find_library(NVCUVID_LIBRARY NAMES
    nvcuvid
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR}
)
mark_as_advanced(NVCUVID_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NVCUVID
                                  REQUIRED_VARS NVCUVID_LIBRARY NVCUVID_INCLUDE
                                  VERSION_VAR NVCUVID_VERSION_STRING)

if(NVCUVID_FOUND)
  set(NVCUVID_LIBRARIES ${NVCUVID_LIBRARY})
  set(NVCUVID_INCLUDE_DIR ${NVCUVID_INCLUDE})
endif()
