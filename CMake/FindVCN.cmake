find_path(VCN_INCLUDE 
    NAMES IVS_SDK.h 
        hwsdk.h
        ivs_error.h
    HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR})
mark_as_advanced(VCN_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
find_library(VCN_LIBRARY NAMES 
                IVS_SDK 
                HINTS ${CMAKE_INSTALL_FULL_LIBDIR})

mark_as_advanced(VCN_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VCN
                                  REQUIRED_VARS VCN_LIBRARY VCN_INCLUDE
                                  VERSION_VAR VCN_VERSION_STRING)

if(VCN_FOUND)
  set(VCN_LIBRARIES ${VCN_LIBRARY})
  set(VCN_INCLUDE_DIR ${VCN_INCLUDE})
  message(STATUS "VCN dependency found, ${VCN_LIBRARIES} ${VCN_INCLUDE_DIR}")
endif()
