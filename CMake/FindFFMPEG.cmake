find_path(FFMPEG_INCLUDE 
  NAMES libavformat/avformat.h libavcodec/avcodec.h libavutil/avutil.h libswscale/swscale.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR}
)
mark_as_advanced(FFMPEG_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
set(FFMPEG_LIBRARY_NAME avutil avcodec avformat)
find_library(AVCODEC_LIBRARY NAMES avcodec HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
find_library(AVUTIL_LIBRARY NAMES avutil HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
find_library(AVFORMAT_LIBRARY NAMES avformat HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
find_library(SWSCALE_LIBRARY NAMES swscale HINTS ${CMAKE_INSTALL_FULL_LIBDIR})
set(FFMPEG_LIBRARY ${AVCODEC_LIBRARY}
                   ${AVUTIL_LIBRARY}
                   ${AVFORMAT_LIBRARY}
                   ${SWSCALE_LIBRARY})
mark_as_advanced(FFMPEG_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFMPEG
                                  REQUIRED_VARS FFMPEG_LIBRARY AVCODEC_LIBRARY AVUTIL_LIBRARY AVFORMAT_LIBRARY SWSCALE_LIBRARY FFMPEG_INCLUDE
                                  VERSION_VAR FFMPEG_VERSION_STRING)

if(FFMPEG_FOUND)
  set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARY})
  set(FFMPEG_INCLUDE_DIR ${FFMPEG_INCLUDE})
endif()
