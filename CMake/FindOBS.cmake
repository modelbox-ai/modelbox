find_path(OBS_INCLUDE NAMES eSDKOBS.h HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR})
mark_as_advanced(OBS_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(OBS_LIBRARY NAMES 
                eSDKOBS 
                HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
find_library(OBSAPI_LIBRARY NAMES 
                eSDKLogAPI 
                HINTS ${CMAKE_INSTALL_FULL_LIBDIR})

mark_as_advanced(OBSAPI_LIBRARY OBS_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OBS
                                  REQUIRED_VARS OBS_LIBRARY OBSAPI_LIBRARY OBS_INCLUDE
                                  VERSION_VAR OBS_VERSION_STRING)

if(OBS_FOUND)
  set(OBS_LIBRARIES ${OBS_LIBRARY} ${OBSAPI_LIBRARY})
  set(OBS_INCLUDE_DIR ${OBS_INCLUDE})
  message(STATUS "OBS dependency found, ${OBS_LIBRARIES} ${OBS_INCLUDE_DIR}")
endif()
