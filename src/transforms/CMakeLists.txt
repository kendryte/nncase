cmake_minimum_required (VERSION 3.8)
include(${NNCASE_TARGET}/src.cmake)

set(SRCS transform.cpp
         neutral/fold_transpose.cpp
         neutral/fold_pad.cpp
         neutral/fold_quantize.cpp
         neutral/fuse_pad.cpp
         neutral/transpose_motion.cpp)

add_library(transforms STATIC ${SRCS} ${TARGET_SRCS})
target_include_directories(transforms PUBLIC include)
target_link_libraries(transforms PUBLIC ir evaluator)
target_link_libraries(transforms PRIVATE CONAN_PKG::range-v3)