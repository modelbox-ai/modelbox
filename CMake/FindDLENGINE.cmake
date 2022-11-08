set(DLENGINE_PATH $ENV{DLENGINE_PATH})

if(NOT DLENGINE_PATH)
  find_package(PythonInterp QUIET)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-c" "import re, dlengine; print(re.compile('/__init__.py.*').sub('', dlengine.__file__))"
    RESULT_VARIABLE DLENGINE_STATUS
    OUTPUT_VARIABLE DLENGINE_PATH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

find_path(DLENGINE_INCLUDE
  NAMES dlengine.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} ${DLENGINE_PATH}/include
)
mark_as_advanced(DLENGINE_INCLUDE)

find_library(DLENGINE_LIBRARY
  NAMES dlengine
  HINTS ${DLENGINE_PATH}
)
mark_as_advanced(DLENGINE_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DLENGINE
                                  REQUIRED_VARS DLENGINE_PATH DLENGINE_LIBRARY DLENGINE_INCLUDE
                                  VERSION_VAR DLENGINE_VERSION_STRING)

if(DLENGINE_FOUND)
  set(DLENGINE_LIBRARIES ${DLENGINE_LIBRARY})
  set(DLENGINE_INCLUDE_DIR ${DLENGINE_INCLUDE})
  set(DLENGINE_BACKEND_ZOO_DIR ${DLENGINE_PATH}/backend_zoo)
endif()
