cmake_minimum_required (VERSION 3.8)

set(SRCS transform.cpp
         pass.cpp
         neutral/fold_memorycopy.cpp)

add_library(llir_transforms STATIC ${SRCS})
target_include_directories(llir_transforms PUBLIC ../include)
target_link_libraries(llir_transforms PUBLIC llir evaluator)