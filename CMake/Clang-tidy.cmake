
find_program(CLANG_TIDY_BIN clang-tidy)

if(NOT CLANG_TIDY_BIN)
    message(STATUS "No clang-tidy found, skip lint")
    return()
endif()

if (NOT CLANG_TIDY_BIN)
    message(STATUS "clang-tidy disabled, skip lint")
    return()
endif()

if (NOT CLANG_TIDY)
    message(STATUS "disable clang-tidy")
    return()
endif()

if (CLANG_TIDY_FIX)
    set(CLANG_TIDY_FLAG "${CLANG_TIDY_FLAG};-fix;")
endif()

message(STATUS "enable clang-tidy lint")
set(CMAKE_CXX_CLANG_TIDY 
   ${CMAKE_CURRENT_LIST_DIR}/clang-tidy-warp;${CLANG_TIDY_FLAG})
