set(HITS_MINDSPORE_PATH $ENV{MINDSPORE_PATH})

if (NOT WITH_MINDSPORE) 
  message(STATUS "not build with mindspore, to enable please add -DWITH_MINDSPORE=on")
  return()
endif()

set(PYTHON_VER 3.5)
find_package(PythonInterp ${PYTHON_VER} QUIET)
if (PYTHON_EXECUTABLE)
  execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(' '.join([s + '/mindspore' for s in site.getsitepackages()]))" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE " " ";" PYTHON_SITE_PACKAGE_LIST ${PYTHON_SITE_PACKAGES})
else()
  set(PYTHON_SITE_PACKAGE_LIST "")
endif()
find_path(MINDSPORE_DIR NAMES include/api/context.h include/api/graph.h include/api/model.h HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${HITS_MINDSPORE_PATH} ${PYTHON_SITE_PACKAGE_LIST})
mark_as_advanced(MINDSPORE_DIR)

# Look for the library (sorted from most current/relevant entry to least).
find_library(MINDSPORE_LIBRARY NAMES
    mindspore
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR} ${HITS_MINDSPORE_PATH}/lib ${MINDSPORE_DIR}/lib
)
mark_as_advanced(MINDSPORE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MINDSPORE
                                  REQUIRED_VARS MINDSPORE_LIBRARY MINDSPORE_DIR
                                  VERSION_VAR MINDSPORE_VERSION_STRING)

if(MINDSPORE_FOUND)
  set(MINDSPORE_LIBRARIES ${MINDSPORE_LIBRARY})
  set(MINDSPORE_INCLUDE_DIR ${MINDSPORE_DIR})
  set(MINDSOPRE_LIB_DIR ${MINDSPORE_DIR}/lib)
endif()
