cmake_minimum_required (VERSION 3.8)

set(SRCS cli.cpp
         compile.cpp
         targets/cpu/target.cpp)

add_executable (ncc ${SRCS})
target_include_directories(ncc PRIVATE ${THIRD_PARTY}/clipp/include)
target_link_libraries(ncc PRIVATE importer data scheduler codegen evaluator transforms)
