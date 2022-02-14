set(HITS_DDK_PATH $ENV{DDK_PATH})
find_path(ACL_INCLUDE
  NAMES acl/acl.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${HITS_DDK_PATH}/include
)
mark_as_advanced(ACL_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
set(ACL_LIBRARY_NAME ascendcl)
set(DDK_LIB_PATH ${HITS_DDK_PATH}/lib64)
find_library(ACL_LIBRARY NAMES ascendcl HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${DDK_LIB_PATH})
find_library(ACL_DVPP_LIBRARY NAMES acl_dvpp HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${DDK_LIB_PATH})
find_library(ACL_CBLAS_LIBRARY NAMES acl_cblas HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${DDK_LIB_PATH})
find_library(ACL_RT_LIBRARY NAMES runtime HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${DDK_LIB_PATH})

mark_as_advanced(ACL_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ACL
                                  REQUIRED_VARS ACL_LIBRARY ACL_DVPP_LIBRARY ACL_CBLAS_LIBRARY ACL_RT_LIBRARY
                                  ACL_INCLUDE
                                  VERSION_VAR ACL_VERSION_STRING)

if(ACL_FOUND)
  set(ACL_LIBRARIES ${ACL_LIBRARY} ${ACL_DVPP_LIBRARY} ${ACL_CBLAS_LIBRARY} ${ACL_RT_LIBRARY})
  set(ACL_INCLUDE_DIR ${ACL_INCLUDE})
endif()
