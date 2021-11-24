find_path(DIS_INCLUDE NAMES dis/dis.h 
    HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR}
)
mark_as_advanced(DIS_INCLUDE)
 
set(HINT_LIBS ${CMAKE_INSTALL_FULL_LIBDIR} /usr/lib64)
 
find_library(DIS_LIBRARY NAMES
    DISSDK
    HINTS ${HINT_LIBS}
)
mark_as_advanced(DIS_LIBRARY)

find_library(CURL_LIBRARY NAMES
    curl
    HINTS ${HINT_LIBS}
)
mark_as_advanced(CURL_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DIS
                                  REQUIRED_VARS DIS_LIBRARY CURL_LIBRARY DIS_INCLUDE
                                  VERSION_VAR DIS_VERSION_STRING)
 
if(DIS_FOUND)
  set(DIS_LIBRARIES ${DIS_LIBRARY} ${CURL_LIBRARY})
  set(DIS_INCLUDE_DIR ${DIS_INCLUDE})
  message(STATUS "Dis dependency found, ${DIS_LIBRARIES} ${DIS_INCLUDE_DIR}")
endif()