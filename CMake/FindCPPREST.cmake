find_path(CPPREST_INCLUDE NAMES cpprest/http_client.h 
    HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR}
)
mark_as_advanced(CPPREST_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(CPPREST_LIBRARY NAMES
    cpprest
    cpprestlib
    libcpprest_imp
    cpprestlib_static
    libcpprest
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR}
)
mark_as_advanced(CPPREST_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPPREST
                                  REQUIRED_VARS CPPREST_LIBRARY CPPREST_INCLUDE
                                  VERSION_VAR CPPREST_VERSION_STRING)

if(CPPREST_FOUND)
  set(CPPREST_LIBRARIES ${CPPREST_LIBRARY})
  set(CPPREST_INCLUDE_DIR ${CPPREST_INCLUDE})
endif()
