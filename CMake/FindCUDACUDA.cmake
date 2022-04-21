if (CUDA_CUDA_LIBRARY) 
  return()
endif()

find_library(CUDA_CUDA_LIBRARY
  NAMES cuda
  HINTS ${CMAKE_INSTALL_FULL_LIBDIR} /usr/local/cuda/lib64/stubs /usr/local/cuda/lib/stubs 
)
mark_as_advanced(CUDA_CUDA_LIBRARY)
