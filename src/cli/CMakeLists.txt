cmake_minimum_required (VERSION 3.8)

set(SRCS cli.cpp
         compile.cpp
         registry.cpp)

add_executable (cli ${SRCS})
target_include_directories(cli PRIVATE ${THIRD_PARTY}/clipp/include)
target_link_libraries(cli PRIVATE importer data scheduler codegen evaluator transforms)
#set_target_properties(cli PROPERTIES OUTPUT_NAME "ncc")