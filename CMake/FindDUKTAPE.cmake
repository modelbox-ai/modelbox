find_path(DUKTAPE_INCLUDE NAMES duktape.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR}
)
mark_as_advanced(DUKTAPE_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
set(DRIVER_LIB_PATH ${HITS_DRIVER_PATH}/lib64)
find_library(DUKTAPE_LIBRARY NAMES
    duktape
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR}
)
mark_as_advanced(DUKTAPE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DUKTAPE
                                  REQUIRED_VARS DUKTAPE_LIBRARY DUKTAPE_INCLUDE
                                  VERSION_VAR DUKTAPE_VERSION_STRING)

if(DUKTAPE_FOUND)
  set(DUKTAPE_LIBRARIES ${DUKTAPE_LIBRARY})
  set(DUKTAPE_INCLUDE_DIR ${DUKTAPE_INCLUDE})
  message(STATUS "Duktape dependency found, ${DUKTAPE_LIBRARIES} ${DUKTAPE_INCLUDE_DIR}")
endif()