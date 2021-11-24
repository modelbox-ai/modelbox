find_path(FUSE_INCLUDE 
  NAMES fuse.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR}
)
mark_as_advanced(FUSE_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
set(FUSE_LIBRARY_NAME fuse)
find_library(FUSE_LIBRARY NAMES fuse HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
set(FUSE_LIBRARY ${FUSE_LIBRARY})
mark_as_advanced(FUSE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FUSE
                                  REQUIRED_VARS FUSE_LIBRARY FUSE_INCLUDE
                                  VERSION_VAR FUSE_VERSION_STRING)

if(FUSE_FOUND)
  set(FUSE_LIBRARIES ${FUSE_LIBRARY})
  set(FUSE_INCLUDE_DIR ${FUSE_INCLUDE})
endif()